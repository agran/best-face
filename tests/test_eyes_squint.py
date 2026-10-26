import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.cluster import FrameRecord, PersonCluster  # noqa: E402
from pipeline.metrics import eye_signals_score  # noqa: E402
from pipeline.score import pick_best, score_cluster  # noqa: E402


def _frame(idx, ear, blink, squint, sharp):
    f = FrameRecord(f'f{idx}.jpg', idx)
    f.det_conf = 0.9
    f.emb_res = np.ones(4, np.float32) / 2
    f.metrics = {
        'Q1_sharpness': sharp, 'Q2_frontality': 0.95, 'Q3_eye_open': 1.0,
        'Q4_expression': 1.0, 'Q5_face_size': 0.5, 'Q6_exposure': 0.8,
        'Q7_confidence': 0.9, 'Q9_nima': 0.5, 'EAR_valid': 1.0,
        'EAR_raw': ear, 'BLINK_raw': blink, 'SQUINT_raw': squint,
    }
    return f


def test_squint_lowers_eye_score():
    open_eyes = eye_signals_score(0.33, 0.05, 0.15)
    squinting = eye_signals_score(0.29, 0.15, 0.50)
    closed = eye_signals_score(0.10, 0.90, 0.40)
    assert open_eyes > 0.95
    assert squinting < open_eyes - 0.15
    assert closed < 0.3


def test_squinting_sharper_frame_does_not_beat_open_eyes():
    cluster = PersonCluster(1)
    # кадр 0 чуть резче, но глаза прищурены; кадры 1-3 открыты
    cluster.frames = [_frame(0, 0.29, 0.15, 0.50, 120.0),
                      _frame(1, 0.33, 0.05, 0.15, 100.0),
                      _frame(2, 0.33, 0.05, 0.16, 98.0),
                      _frame(3, 0.34, 0.04, 0.15, 99.0)]
    cluster.compute_centroids()
    scored = score_cluster(cluster)
    best = pick_best(scored)
    assert best['frame'].file_idx != 0
    squint = next(r for r in scored if r['frame'].file_idx == 0)
    assert squint['rejected_reason'] == 'eyes_half_closed'


def test_half_closed_kept_when_no_alternative():
    cluster = PersonCluster(1)
    cluster.frames = [_frame(0, 0.25, 0.15, 0.45, 100.0),
                      _frame(1, 0.24, 0.16, 0.47, 100.0)]
    cluster.compute_centroids()
    scored = score_cluster(cluster)
    assert pick_best(scored) is not None
    assert any(r['passed_gates'] for r in scored)


def test_sharpness_is_saturating_not_minmax():
    cluster = PersonCluster(1)
    # резкость различается на 5% — это не должно перевешивать лучший кадр по глазам
    cluster.frames = [_frame(0, 0.33, 0.05, 0.15, 100.0),
                      _frame(1, 0.33, 0.05, 0.15, 105.0)]
    cluster.compute_centroids()
    scored = score_cluster(cluster)
    q1 = [r['metrics_norm']['Q1_sharpness'] for r in scored]
    assert max(q1) - min(q1) < 1e-9
