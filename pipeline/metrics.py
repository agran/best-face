"""
pipeline/metrics.py — расчёт метрик качества кадра (Q1-Q8).

Метрики рассчитываются на этапе обработки кадра (до кластеризации).
Q8 (централность эмбеддинга) рассчитывается позже в score.py.

Все метрики возвращают float в диапазоне [0, 1] (до нормировки по серии).
"""

import cv2
import numpy as np
from typing import Tuple


def q1_sharpness(face_crop: np.ndarray) -> float:
    """
    Q1: Резкость / фокус.
    Variance of Laplacian на area лица в grayscale.
    Высокое значение = резкое, низкое = смазанное.
    Возвращает сырое (ненормированное) значение — нормировка в score.py по серии.
    """
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def q2_frontality(kps: np.ndarray) -> float:
    """
    Q2: Фронтальность. Уже рассчитывается в align.estimate_frontal_ratio().
    Принимает готовое значение [0, 1].
    """
    # Передаётся как параметр, здесь просто паспорт для совместимости
    raise NotImplementedError("используй pipeline.align.estimate_frontal_ratio()")


def q3_eye_openness(face_crop: np.ndarray, kps_112=None, ear_value: float = None) -> float:
    """
    Q3: Открытость глаз.

    Если передан ear_value (из MediaPipe FaceLandmarker, EAR-метрика):
        score = min(1.0, ear_value / 0.30)
        EAR~0.28-0.33 = открытый, EAR~0.09-0.15 = закрытый/прищуренный
        Порог min_ear=0.58 → требуется EAR >= 0.174

    Иначе — fallback pixel-based (менее надёжен при прищуривании):
    A) pupil_score  — min пикселя в патче: open~10-15 → 0.75-0.83; closed~30-50 → 0.17-0.50.
       Нормировка: max(0, 1 - min_val / 60).  Вес: 0.50.
    B) darkness_score — mean центральной зоны 50×50%: open→low, closed→skin.
       max(0, 1 - mean_center / 160).  Вес: 0.25.
    C) contrast_score — std патча: open→высокий, closed→низкий.
       min(1, std / 35).  Вес: 0.25.
    """
    if ear_value is not None:
        return min(1.0, max(0.0, ear_value / 0.30))
    h, w = face_crop.shape[:2]
    if h < 32 or w < 32:
        return 0.5

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

    scale = h / 112.0
    # ArcFace template positions: (x, y)
    eye_re = np.array([38.2946 * scale, 51.6963 * scale])
    eye_le = np.array([73.5318 * scale, 51.5014 * scale])

    pw = max(10, int(18 * scale))   # полуширина патча
    ph = max(6,  int(11 * scale))   # полувысота патча

    scores = []
    for ex, ey in [eye_re, eye_le]:
        x1 = max(0, int(ex - pw)); x2 = min(w, int(ex + pw))
        y1 = max(0, int(ey - ph)); y2 = min(h, int(ey + ph))
        if x2 <= x1 or y2 <= y1:
            continue
        patch = gray[y1:y2, x1:x2]
        ph_act, pw_act = patch.shape

        # A) Зрачок (ГЛАВНЫЙ признак): открытый глаз → тёмный зрачок виден →
        #    минимальный пиксель в патче очень тёмный (~10-15).
        #    Закрытый глаз: только ресницы/тени → min~30-50.
        #    Нормировка: min=0 → 1.0, min=60 → 0.0
        min_val = float(patch.min())
        pupil_score = max(0.0, 1.0 - min_val / 60.0)

        # B) Тёмность центра (зрачок открытого глаза → низкий mean_center)
        cy1, cy2 = ph_act // 4, 3 * ph_act // 4
        cx1, cx2 = pw_act // 4, 3 * pw_act // 4
        mean_center = float(patch[cy1:cy2, cx1:cx2].mean())
        darkness_score = max(0.0, 1.0 - mean_center / 160.0)

        # C) Контрастность патча (открытый глаз → высокий std)
        contrast_score = min(1.0, float(patch.std()) / 35.0)

        score = 0.50 * pupil_score + 0.25 * darkness_score + 0.25 * contrast_score
        scores.append(score)

    normalized = float(np.mean(scores)) if scores else 0.5
    return normalized


def q4_expression_neutrality(face_crop: np.ndarray, kps_orig: np.ndarray) -> float:
    """
    Q4: Нейтральность выражения (прокси).
    Оцениваем по геометрии рта: большое расстояние между уголками рта
    относительно расстояния глаза-нос → широкая улыбка / открытый рот.
    Нейтральное лицо: рот относительно закрыт.

    kps_orig — 5 точек в оригинальных координатах кадра:
               [re, le, nose, right_mouth, left_mouth]
    """
    right_eye   = kps_orig[0]
    left_eye    = kps_orig[1]
    nose        = kps_orig[2]
    right_mouth = kps_orig[3]
    left_mouth  = kps_orig[4]

    inter_eye    = float(np.linalg.norm(left_eye - right_eye)) + 1e-6
    mouth_width  = float(np.linalg.norm(left_mouth - right_mouth))

    # Нормализованная ширина рта (относительно расстояния между глазами)
    mouth_ratio = mouth_width / inter_eye

    # Нейтральное лицо: ~0.7-0.9, широкая улыбка/открытый рот: >1.0
    # Штрафуем за слишком большой рот
    if mouth_ratio < 1.0:
        return 1.0
    else:
        # Плавный штраф: 1.0 → 0.5 при mouth_ratio 1.0 → 1.5
        return max(0.3, 1.0 - (mouth_ratio - 1.0) * 1.4)


def q5_face_size(bbox: Tuple[int, int, int, int],
                 img_shape: Tuple[int, int]) -> float:
    """
    Q5: Относительный размер лица в кадре.
    Доля площади bbox лица от площади изображения, нормированная [0, 1].
    """
    x, y, bw, bh = bbox
    img_h, img_w = img_shape[:2]
    img_area  = float(img_w * img_h)
    face_area = float(bw * bh)
    ratio = face_area / (img_area + 1e-6)
    # Нормировать: 5% площади = хорошо, ≥30% = отлично
    return float(min(1.0, ratio / 0.10))


def q6_exposure(face_crop: np.ndarray) -> float:
    """
    Q6: Качество экспозиции / освещения лица.
    Штрафуем за:
    - пересвет: много пикселей >240
    - недосвет: много пикселей <20
    - низкий контраст: маленький диапазон гистограммы
    Возвращает 1.0 = хорошо, 0.0 = плохо.
    """
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    total = float(gray.size)

    overexposed  = float(np.sum(gray > 240)) / total
    underexposed = float(np.sum(gray < 20))  / total

    # Пенальти за клиппинг
    clip_penalty = min(1.0, (overexposed + underexposed) * 5.0)

    # Контраст: std яркости (хорошее фото ~40-80)
    std = float(gray.std())
    contrast_score = min(1.0, std / 50.0)

    score = (1.0 - clip_penalty) * 0.6 + contrast_score * 0.4
    return max(0.0, min(1.0, score))


def q9_face_illumination(face_crop: np.ndarray) -> float:
    """
    Q9: Качество освещения лица (равномерность, тени, пересветы).

    Применяется к выровненному кропу (112×112 или 224×224).
    Оценивает пригодность снимка для распознавания лица по освещению:
    равномерный мягкий свет → высокий балл; жёсткие тени или пересветы → низкий.

    A) Оптимальность средней яркости (диапазон ~100-160 из 255):
       score_A = max(0, 1 - |mean - 130| / 130)

    B) Тени: доля очень тёмных пикселей (< 55). Допустимо ≤ 8%.
       score_B = max(0, 1 - shadow_frac / 0.08)

    C) Пересветы: доля выбитых пикселей (> 235). Допустимо ≤ 6%.
       score_C = max(0, 1 - highlight_frac / 0.06)

    D) Равномерность: std средних значений зон 3×3 по кропу.
       Равномерный свет → низкий std (< 15 → отлично, > 60 → плохо).
       score_D = max(0, 1 - zone_std / 60)

    Итого: 0.20*A + 0.25*B + 0.25*C + 0.30*D → [0, 1].

    Ожидаемые значения: равномерный свет ≈ 0.70-0.90,
                        жёсткие тени / пересвет ≈ 0.30-0.60.
    """
    h, w = face_crop.shape[:2]
    if h < 32 or w < 32:
        return 0.5

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # A) Средняя яркость
    mean_val = float(gray.mean())
    score_a = max(0.0, 1.0 - abs(mean_val - 130.0) / 130.0)

    # B) Доля теней (очень тёмные пиксели)
    total = float(gray.size)
    shadow_frac = float(np.sum(gray < 55)) / total
    score_b = max(0.0, 1.0 - shadow_frac / 0.08)

    # C) Доля пересветов (выбитые пиксели)
    highlight_frac = float(np.sum(gray > 235)) / total
    score_c = max(0.0, 1.0 - highlight_frac / 0.06)

    # D) Равномерность освещения по зонам 3×3
    zh, zw = max(1, h // 3), max(1, w // 3)
    zone_means = []
    for r in range(3):
        for c in range(3):
            zone = gray[r * zh:(r + 1) * zh, c * zw:(c + 1) * zw]
            if zone.size > 0:
                zone_means.append(float(zone.mean()))
    zone_std = float(np.std(zone_means)) if len(zone_means) > 1 else 0.0
    score_d = max(0.0, 1.0 - zone_std / 60.0)

    return 0.20 * score_a + 0.25 * score_b + 0.25 * score_c + 0.30 * score_d


def compute_all_metrics(
    img_bgr: np.ndarray,
    face_crop_112: np.ndarray,
    kps_orig: np.ndarray,
    bbox: Tuple[int, int, int, int],
    det_conf: float,
    frontal_ratio: float,
    ear_value: float = None,
) -> dict:
    """
    Рассчитать все метрики Q1-Q9 для одного кадра.
    Q8 (централность) рассчитывается позже в score.py.
    Q9 (NIMA) может быть переопределён в best_face.py если доступна NIMA модель.

    Возвращает dict с ключами Q1..Q9 и вспомогательными полями.
    """
    return {
        'Q1_sharpness':     q1_sharpness(face_crop_112),      # сырое
        'Q2_frontality':    float(frontal_ratio),             # [0,1]
        'Q3_eye_open':      q3_eye_openness(face_crop_112, None, ear_value),
        'Q4_expression':    q4_expression_neutrality(face_crop_112, kps_orig),
        'Q5_face_size':     q5_face_size(bbox, img_bgr.shape),
        'Q6_exposure':      q6_exposure(face_crop_112),
        'Q7_confidence':    float(det_conf),
        'Q8_centrality':    0.0,  # заполняется позже в score.py
        'Q9_nima':          q9_face_illumination(face_crop_112),  # базовый; NIMA перезапишет
        'EAR_valid':        1.0 if ear_value is not None else 0.0,
    }
