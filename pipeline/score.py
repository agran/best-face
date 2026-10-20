"""
pipeline/score.py — скоринг кадров и выбор лучшего в каждой серии.

Порядок действий:
  1. Для каждого кластера вычислить Q8 (централность FaceRes-эмбеддинга).
  2. Нормировать Q1 и Q5 по серии (min-max).
  3. Применить жёсткие фильтры (hard gates) — отбросить явно непригодные кадры.
  4. Взвешенная сумма Q1-Q8 → итоговый score.
  5. Выбрать кадр с максимальным score.
"""

import os
import numpy as np
from typing import List, Dict, Optional
from .cluster import FrameRecord, PersonCluster


_DEFAULT_WEIGHTS = {
    'Q1_sharpness':   0.20,
    'Q2_frontality':  0.20,
    'Q3_eye_open':    0.12,
    'Q4_expression':  0.05,
    'Q5_face_size':   0.08,
    'Q6_exposure':    0.07,
    'Q7_confidence':  0.03,
    'Q8_centrality':  0.15,
    'Q9_nima':        0.10,
}

_DEFAULT_GATES = {
    'max_yaw_deg':       35.0,
    'max_pitch_deg':     30.0,
    'min_frontal_ratio': 0.45,
    'min_ear':           0.58,
    'min_sharpness_rel': 0.15,
}

# Нормировка Q3 по человеку: если у человека в норме Q3 ниже _EYE_BASELINE_MIN
# (узкий разрез глаз), его открытые глаза масштабируются к уровню _EYE_TARGET
_EYE_BASELINE_MIN = 0.75
_EYE_TARGET       = 0.90


def mark_manual(scored: List[dict], file_name: str) -> bool:
    """Пометить кадр с этим именем файла как выбранный вручную. Возвращает True, если найден."""
    found = False
    for r in scored:
        r['manual'] = (os.path.basename(r['frame'].path) == file_name)
        found = found or r['manual']
    return found


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return values
    mn, mx = min(values), max(values)
    rng = mx - mn
    if rng < 1e-9:
        return [1.0] * len(values)
    return [(v - mn) / rng for v in values]


def score_cluster(
    cluster: PersonCluster,
    weights: Dict[str, float] = None,
    gates:   Dict[str, float] = None,
) -> List[dict]:
    """
    Рассчитать scores для всех кадров кластера и выбрать лучший.

    Возвращает список dict с полями:
      frame, score, metrics_norm, passed_gates, rejected_reason
    Список отсортирован по убыванию score.
    """
    if weights is None:
        weights = _DEFAULT_WEIGHTS
    if gates is None:
        gates = _DEFAULT_GATES

    frames = cluster.frames
    if not frames:
        return []

    # ── 1. Q8: централность FaceRes-эмбеддинга к центроиду кластера ──────────
    centroid = cluster.centroid_res
    for f in frames:
        if centroid is not None and f.emb_res is not None:
            q8 = float(np.dot(f.emb_res, centroid))
            q8 = (q8 + 1.0) / 2.0  # [-1,1] → [0,1]
        else:
            q8 = 0.5  # нет данных → нейтрально
        f.metrics['Q8_centrality'] = q8

    # ── 2. Нормировка Q1, Q5 и Q9 по серии (min-max) ─────────────────────
    q1_raw = [f.metrics.get('Q1_sharpness', 0.0) for f in frames]
    q5_raw = [f.metrics.get('Q5_face_size', 0.0) for f in frames]
    q9_raw = [f.metrics.get('Q9_nima', 0.5)      for f in frames]

    q1_norm = _minmax(q1_raw)
    q5_norm = _minmax(q5_raw)
    # Q9 нормируем только если модель дала разные значения (иначе все 0.5 → minmax=1.0)
    q9_all_same = max(q9_raw) - min(q9_raw) < 0.01
    q9_norm = [0.5] * len(q9_raw) if q9_all_same else _minmax(q9_raw)

    max_sharpness = max(q1_raw) if q1_raw else 1.0

    # Q3 относительно самого человека: у людей с узким разрезом глаз открытый глаз
    # даёт стабильно низкий Q3, и абсолютный порог принял бы их за «моргнувших».
    # Базовый уровень — 75-й перцентиль Q3 по кадрам с надёжным EAR.
    q3_scale = 1.0
    ear_q3 = [f.metrics.get('Q3_eye_open', 1.0) for f in frames
              if f.metrics.get('EAR_valid', 0.0) > 0.5]
    if len(ear_q3) >= 3:
        baseline = float(np.percentile(ear_q3, 75))
        if 0.15 < baseline < _EYE_BASELINE_MIN:
            q3_scale = _EYE_TARGET / baseline

    results = []
    for i, f in enumerate(frames):
        m = f.metrics.copy()
        m['Q1_sharpness'] = q1_norm[i]
        m['Q5_face_size']  = q5_norm[i]
        m['Q9_nima']       = q9_norm[i]
        if q3_scale != 1.0 and m.get('EAR_valid', 0.0) > 0.5:
            m['Q3_eye_open'] = min(1.0, m.get('Q3_eye_open', 1.0) * q3_scale)

        # ── 3. Жёсткие фильтры ───────────────────────────────────────────────
        rej = None

        if abs(f.yaw_deg) > gates.get('max_yaw_deg', 35.0):
            rej = 'yaw_too_large'

        if f.pitch_deg is not None and abs(f.pitch_deg) > gates.get('max_pitch_deg', 30.0):
            rej = rej or 'pitch_too_large'

        if f.frontal_ratio < gates.get('min_frontal_ratio', 0.45):
            rej = rej or 'not_frontal'

        # Порог Q3 откалиброван под EAR; pixel-based fallback на другой шкале — не гейтим
        if m.get('EAR_valid', 1.0) > 0.5 and m.get('Q3_eye_open', 1.0) < gates.get('min_ear', 0.58):
            rej = rej or 'eyes_closed'

        min_sharp_rel = gates.get('min_sharpness_rel', 0.15)
        if max_sharpness > 0 and q1_raw[i] < max_sharpness * min_sharp_rel:
            rej = rej or 'blurry'

        if f.det_conf < 0.5:
            rej = rej or 'low_confidence'

        # ── 4. Итоговый score ────────────────────────────────────────────────
        score = 0.0
        for key, w in weights.items():
            score += w * m.get(key, 0.5)

        results.append({
            'frame':          f,
            'score':          float(score),
            'metrics_norm':   m,
            'passed_gates':   rej is None,
            'rejected_reason': rej,
        })

    # Сортировка: сначала прошедшие фильтры, затем по убыванию score
    results.sort(key=lambda r: (int(r['passed_gates']), r['score']), reverse=True)

    # Лучший = первый. Если все отфильтрованы — лучший всё равно первый
    # (с пометкой 'all_filtered' для серии)
    return results


def pick_best(scored: List[dict]) -> Optional[dict]:
    """Вернуть лучший кадр из результатов score_cluster(); ручной выбор имеет приоритет."""
    if not scored:
        return None
    for r in scored:
        if r.get('manual'):
            return r
    # Предпочесть прошедший gates, иначе первый
    for r in scored:
        if r['passed_gates']:
            return r
    return scored[0]  # все плохие — берём лучшее из худших
