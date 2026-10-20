"""
pipeline/detect.py — детекция лиц через cv2.FaceDetectorYN (YuNet).

Возвращает для каждого лица:
  bbox:  (x, y, w, h)  — в пикселях исходного изображения
  kps:   5×2 float32   — 5 ключевых точек: right_eye, left_eye, nose, right_mouth, left_mouth
  conf:  float          — confidence detector

Примечание: «right/left» — с точки зрения наблюдателя (камеры).
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class FaceDetectionResult:
    __slots__ = ('bbox', 'kps', 'conf')

    def __init__(self, bbox: Tuple[int, int, int, int],
                 kps: np.ndarray, conf: float):
        self.bbox = bbox    # (x, y, w, h) int
        self.kps  = kps     # (5, 2) float32: re, le, nose, rm, lm
        self.conf = conf


class FaceDetector:
    """Тонкая обёртка над cv2.FaceDetectorYN."""

    def __init__(self, model_path: str,
                 conf_threshold: float = 0.5,
                 nms_threshold: float = 0.4,
                 max_det_size: int = 1280):
        self._model_path = model_path
        self._conf       = conf_threshold
        self._nms        = nms_threshold
        self._max_size   = max_det_size
        self._detector   = None
        self._cur_size   = (0, 0)

    def _ensure_detector(self, w: int, h: int):
        if self._detector is None or self._cur_size != (w, h):
            self._detector = cv2.FaceDetectorYN.create(
                self._model_path, '', (w, h),
                self._conf, self._nms, top_k=1000
            )
            self._cur_size = (w, h)

    def detect(self, img_bgr: np.ndarray,
               min_face_px: int = 50) -> List[FaceDetectionResult]:
        """
        Детектировать лица в изображении BGR.

        img_bgr     — numpy BGR (H, W, 3)
        min_face_px — минимальный размер лица (меньшая сторона bbox)
        Возвращает список FaceDetectionResult, отсортированных по убыванию площади.
        """
        h_orig, w_orig = img_bgr.shape[:2]

        # Масштабирование для ускорения инференса
        scale = 1.0
        if max(w_orig, h_orig) > self._max_size:
            scale = self._max_size / max(w_orig, h_orig)
            new_w = int(w_orig * scale)
            new_h = int(h_orig * scale)
            img_small = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            new_w, new_h = w_orig, h_orig
            img_small = img_bgr

        self._ensure_detector(new_w, new_h)
        _, faces_raw = self._detector.detect(img_small)

        if faces_raw is None:
            return []

        results = []
        for row in faces_raw:
            # row: x1,y1,w,h, re_x,re_y, le_x,le_y, nt_x,nt_y, rm_x,rm_y, lm_x,lm_y, score
            x, y, w, h = row[0], row[1], row[2], row[3]
            conf = float(row[14])

            # Отмасштабировать обратно к оригинальному размеру
            if scale != 1.0:
                x /= scale; y /= scale; w /= scale; h /= scale

            if min(w, h) < min_face_px:
                continue

            # Ключевые точки (5 шт.)
            kps_raw = row[4:14].reshape(5, 2)
            if scale != 1.0:
                kps_raw = kps_raw / scale

            bbox = (int(x), int(y), int(w), int(h))
            kps  = kps_raw.astype(np.float32)

            results.append(FaceDetectionResult(bbox, kps, conf))

        # Сортировка по убыванию площади (самое крупное лицо — первым)
        results.sort(key=lambda r: r.bbox[2] * r.bbox[3], reverse=True)
        return results
