"""
pipeline/ear.py — вычисление Eye Aspect Ratio (EAR) через MediaPipe Face Landmarker.

EAR — классическая 6-точечная метрика Soukupova & Cech (2016):
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    где p1,p4 — уголки глаза, p2,p3,p5,p6 — верхнее/нижнее веко.

EAR ≈ 0.0–0.15  → закрытый/прищуренный глаз
EAR ≈ 0.25–0.35 → открытый глаз

Нормированный Q3-score: min(1.0, EAR / 0.30)
При мин. пороге min_ear=0.58: требуется EAR >= 0.174.
"""

import os
from typing import Optional, Tuple

import cv2
import numpy as np


# ── Индексы MediaPipe Face Mesh (478 точек) ────────────────────────────────────
# Порядок: [outer_corner, upper1, upper2, inner_corner, lower2, lower1]
# — правый глаз (с точки зрения камеры = левый на экране зрителя)
_RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
# — левый глаз
_LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]

# Нормировка EAR → [0, 1]
_EAR_SCALE = 0.30          # EAR полностью открытого глаза ≈ 0.28–0.33


def _ear_from_6pts(pts: np.ndarray) -> float:
    """Классическая формула EAR из 6 точек (N×2 float)."""
    return float(
        (np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4]))
        / (2.0 * np.linalg.norm(pts[0] - pts[3]) + 1e-6)
    )


class EarEstimator:
    """
    Обёртка над MediaPipe FaceLandmarker для вычисления EAR.

    Если mediapipe не установлен или модель не найдена — gracefully unavailable.
    """

    def __init__(self, model_path: str):
        self._available = False
        self._landmarker = None
        self._mp = None

        if not model_path or not os.path.exists(model_path):
            print(f'[EarEstimator] модель не найдена: {model_path!r} — Q3 будет pixel-based')
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python.vision.face_landmarker import (
                FaceLandmarker, FaceLandmarkerOptions,
            )

            base_opts = mp_python.BaseOptions(model_asset_path=model_path)
            opts = FaceLandmarkerOptions(
                base_options=base_opts,
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                min_face_detection_confidence=0.3,
                min_face_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )
            self._landmarker = FaceLandmarker.create_from_options(opts)
            self._mp = mp
            self._available = True
            print(f'[EarEstimator] OK — MediaPipe FaceLandmarker загружен')
        except Exception as e:
            print(f'[EarEstimator] ошибка инициализации: {e} — Q3 будет pixel-based')

    @property
    def available(self) -> bool:
        return self._available

    def analyze(
        self,
        img_bgr: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[dict]:
        """
        Разобрать лицо через MediaPipe FaceLandmarker.

        img_bgr    — исходное BGR изображение
        face_bbox  — (x, y, w, h) для обрезки ROI (ускоряет инференс)

        Возвращает dict или None при ошибке:
          ear          — среднее EAR двух глаз (0≈closed, 0.30≈open)
          blink        — среднее eyeBlinkLeft/Right [0..1] (1 = глаз закрыт)
          jaw_open     — jawOpen [0..1]
          smile        — среднее mouthSmileLeft/Right [0..1]
          yaw/pitch/roll — поворот головы в градусах (из матрицы трансформации)
        """
        if not self._available:
            return None

        # Вырезаем ROI с запасом 35% для надёжного детектирования
        if face_bbox is not None:
            x, y, w, h = face_bbox
            pad = int(max(w, h) * 0.35)
            H, W = img_bgr.shape[:2]
            x1 = max(0, x - pad);  y1 = max(0, y - pad)
            x2 = min(W, x + w + pad); y2 = min(H, y + h + pad)
            roi = img_bgr[y1:y2, x1:x2]
        else:
            roi = img_bgr

        try:
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect(mp_img)
        except Exception:
            return None

        if not result.face_landmarks:
            return None

        H_roi, W_roi = roi.shape[:2]
        lms = result.face_landmarks[0]
        pts = np.array([[lm.x * W_roi, lm.y * H_roi] for lm in lms], dtype=np.float32)

        out = {
            'ear': (_ear_from_6pts(pts[_RIGHT_EYE_IDX]) + _ear_from_6pts(pts[_LEFT_EYE_IDX])) / 2.0,
            'blink': None, 'jaw_open': None, 'smile': None,
            'yaw': None, 'pitch': None, 'roll': None,
        }

        if result.face_blendshapes:
            bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
            out['blink'] = (bs.get('eyeBlinkLeft', 0.0) + bs.get('eyeBlinkRight', 0.0)) / 2.0
            out['jaw_open'] = bs.get('jawOpen', 0.0)
            out['smile'] = (bs.get('mouthSmileLeft', 0.0) + bs.get('mouthSmileRight', 0.0)) / 2.0

        if result.facial_transformation_matrixes:
            R = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)[:3, :3]
            R = R / np.maximum(np.linalg.norm(R, axis=0), 1e-9)
            out['yaw']   = float(np.degrees(np.arctan2(R[0, 2], R[2, 2])))
            out['pitch'] = float(np.degrees(np.arcsin(np.clip(-R[1, 2], -1.0, 1.0))))
            out['roll']  = float(np.degrees(np.arctan2(R[1, 0], R[1, 1])))

        return out

    def estimate(
        self,
        img_bgr: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[float]:
        """Среднее EAR двух глаз или None (совместимость; см. analyze())."""
        info = self.analyze(img_bgr, face_bbox)
        return None if info is None else info['ear']

    def ear_to_score(self, ear_value: float) -> float:
        """Нормировать EAR → [0, 1]. EAR=0.30 → 1.0, EAR=0.174 → 0.58 (порог)."""
        return min(1.0, max(0.0, ear_value / _EAR_SCALE))
