"""
pipeline/align.py — 5-точечный аффинный alignment лица (ArcFace template).

Принимает 5 keypoints от YuNet (right_eye, left_eye, nose, right_mouth, left_mouth)
и возвращает выровненный кроп лица нужного размера.

Опорные точки ArcFace взяты из официальной реализации InsightFace.
"""

import cv2
import numpy as np
from typing import Optional


# Опорные точки ArcFace для 112×112
# Порядок: right_eye, left_eye, nose_tip, right_mouth, left_mouth
# «right/left» — с точки зрения камеры (правое = слева в кадре у зрителя)
_ARCFACE_DST_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

# Масштабированные на 224×224 (для Human FaceRes)
_ARCFACE_DST_224 = _ARCFACE_DST_112 * (224.0 / 112.0)


def get_arcface_ref(size: int) -> np.ndarray:
    """Возвращает опорные точки ArcFace для заданного размера кропа."""
    if size == 112:
        return _ARCFACE_DST_112
    elif size == 224:
        return _ARCFACE_DST_224
    else:
        return _ARCFACE_DST_112 * (size / 112.0)


def align_face(img_bgr: np.ndarray,
               kps: np.ndarray,
               output_size: int = 112) -> Optional[np.ndarray]:
    """
    Выровнять лицо по 5 ключевым точкам (similarity transform).

    img_bgr     — исходное BGR изображение
    kps         — (5, 2) float32: right_eye, left_eye, nose, right_mouth, left_mouth
    output_size — размер выходного кропа (квадрат)

    Возвращает BGR кроп (output_size, output_size, 3) или None при ошибке.
    """
    dst = get_arcface_ref(output_size)
    src = kps.astype(np.float32)

    # estimateAffinePartial2D = similarity transform (4 DOF: rotation, scale, tx, ty)
    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        return None

    aligned = cv2.warpAffine(
        img_bgr, M, (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    return aligned


def estimate_yaw_deg(kps: np.ndarray) -> float:
    """
    Грубая оценка поворота головы в градусах из 5 точек.

    Использует горизонтальное смещение кончика носа относительно
    середины межглазной линии, нормированное на межглазное расстояние.
    >0 = лицо повернуто вправо (от камеры), <0 = влево.
    """
    right_eye  = kps[0]
    left_eye   = kps[1]
    nose       = kps[2]

    eye_center = (right_eye + left_eye) / 2.0
    inter_eye  = float(np.linalg.norm(left_eye - right_eye))
    if inter_eye < 1e-6:
        return 0.0

    offset = float(nose[0] - eye_center[0])
    # Линейное приближение: ±1 межглазное расстояние ≈ ±45°
    yaw_deg = (offset / inter_eye) * 45.0
    return yaw_deg


def estimate_frontal_ratio(kps: np.ndarray) -> float:
    """
    Оценка фронтальности по метрике Face-ID Kiosk: min(d_re_nose, d_le_nose) / max(...).
    1.0 = полный анфас, 0.0 = полный профиль.
    """
    right_eye = kps[0]
    left_eye  = kps[1]
    nose      = kps[2]

    d_re = float(np.linalg.norm(right_eye - nose))
    d_le = float(np.linalg.norm(left_eye  - nose))

    dmin = min(d_re, d_le)
    dmax = max(d_re, d_le)
    if dmax < 1e-6:
        return 1.0
    return dmin / dmax
