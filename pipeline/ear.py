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
# Центры радужек (face mesh с iris): 468 — для глаза 33/133, 473 — для глаза 362/263
_IRIS_RIGHT = 468
_IRIS_LEFT  = 473

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
          squint       — среднее eyeSquintLeft/Right [0..1] (прищур: полузакрытые глаза, улыбка)
          jaw_open     — jawOpen [0..1]
          smile        — среднее mouthSmileLeft/Right [0..1]
          yaw/pitch/roll — поворот головы в градусах (из матрицы трансформации)
          gaze_x/gaze_y  — смещение зрачка от центра глаза в долях полуширины глаза (по радужке)
          look_h/look_v  — отклонение взгляда по blendshapes eyeLook* [0..1]
          expr           — суммарная «ненейтральность» выражения (0 = серьёзное лицо)
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
            'blink': None, 'squint': None, 'jaw_open': None, 'smile': None,
            'yaw': None, 'pitch': None, 'roll': None,
            'gaze_x': None, 'gaze_y': None, 'look_h': None, 'look_v': None, 'expr': None,
        }

        # Взгляд по радужке (ландмарки 468-477): смещение центра зрачка от центра глаза
        # в долях полуширины глаза. 0 — зрачок по центру (взгляд прямо относительно головы).
        if len(pts) >= 478:
            gx, gy = [], []
            for iris, c_out, c_in in ((_IRIS_RIGHT, 33, 133), (_IRIS_LEFT, 362, 263)):
                a, b = pts[c_out], pts[c_in]
                half = max(float(np.linalg.norm(a - b)) / 2.0, 1e-6)
                mid = (a + b) / 2.0
                gx.append(float((pts[iris][0] - mid[0]) / half))
                gy.append(float((pts[iris][1] - mid[1]) / half))
            # смещения в системе изображения: знак у двух глаз общий, поэтому усредняем
            out['gaze_x'], out['gaze_y'] = float(np.mean(gx)), float(np.mean(gy))

        if result.face_blendshapes:
            bs = {c.category_name: c.score for c in result.face_blendshapes[0]}

            def pair(name):
                return (bs.get(name + 'Left', 0.0) + bs.get(name + 'Right', 0.0)) / 2.0

            out['blink'] = pair('eyeBlink')
            out['squint'] = pair('eyeSquint')
            out['jaw_open'] = bs.get('jawOpen', 0.0)
            out['smile'] = pair('mouthSmile')
            # Взгляд по blendshapes: отклонение в стороны (in/out) и вверх/вниз
            out['look_h'] = max(abs(bs.get('eyeLookInLeft', 0.0) - bs.get('eyeLookOutLeft', 0.0)),
                                abs(bs.get('eyeLookInRight', 0.0) - bs.get('eyeLookOutRight', 0.0)))
            out['look_v'] = max(abs(bs.get('eyeLookUpLeft', 0.0) - bs.get('eyeLookDownLeft', 0.0)),
                                abs(bs.get('eyeLookUpRight', 0.0) - bs.get('eyeLookDownRight', 0.0)))
            # «Ненейтральность» выражения: улыбка, открытый рот, растянутые губы, губы
            # трубочкой, нахмуренные (browDown) или поднятые брови. Серьёзное лицо → близко к 0.
            out['expr'] = float(
                1.0 * out['smile']
                + 0.5 * pair('mouthDimple')
                + 1.0 * out['jaw_open']
                + 0.7 * pair('mouthStretch')
                + 1.0 * bs.get('mouthPucker', 0.0)
                + 0.7 * bs.get('mouthFunnel', 0.0)
                + 0.4 * max(bs.get('browInnerUp', 0.0), pair('browOuterUp'))
                + 2.0 * pair('browDown')   # нахмуренные брови заметно хуже слегка поднятых
                + 0.5 * pair('cheekSquint')
                + 0.5 * pair('mouthFrown')
                + 0.3 * pair('mouthUpperUp')
            )

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
