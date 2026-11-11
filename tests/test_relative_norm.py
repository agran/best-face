import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.cluster import FrameRecord, PersonCluster  # noqa: E402
from pipeline.score import pick_best, score_cluster  # noqa: E402


def _frame(idx, expr, size, nima):
    f = FrameRecord(f'f{idx}.jpg', idx)
    f.det_conf = 0.9
    f.emb_res = np.ones(4, np.float32) / 2
    f.yaw_deg, f.pitch_deg, f.roll_deg = 0.0, 0.0, 0.0
    f.metrics = {
        'Q1_sharpness': 100.0, 'Q2_frontality': 0.95, 'Q3_eye_open': 1.0,
        'Q4_expression': 1.0, 'Q5_face_size': size, 'Q6_exposure': 0.8,
        'Q7_confidence': 0.9, 'Q9_nima': nima, 'EAR_valid': 1.0,
        'EAR_raw': 0.33, 'BLINK_raw': 0.05, 'SQUINT_raw': 0.15,
        'GAZE_X_raw': 0.0, 'GAZE_Y_raw': -0.14, 'LOOK_H_raw': 0.05, 'LOOK_V_raw': 0.1,
        'EXPR_raw': expr, 'JAW_raw': 0.0, 'SMILE_raw': 0.0,
    }
    return f


def test_tiny_size_and_nima_differences_do_not_decide_the_pick():
    # Кадр 0 — рот приоткрыт (губы трубочкой), чуть крупнее и эстетичнее на ~3%
    # (раньше min-max растягивал это в разницу 0..1 и кадр 0 выигрывал)
    cluster = PersonCluster(1)
    cluster.frames = [_frame(0, expr=0.24, size=0.724, nima=0.482),
                      _frame(1, expr=0.02, size=0.703, nima=0.469)]
    cluster.compute_centroids()
    scored = score_cluster(cluster)
    assert pick_best(scored)['frame'].file_idx == 1
    assert next(r for r in scored if r['frame'].file_idx == 0)['rejected_reason'] == 'not_neutral'
    q5 = {r['frame'].file_idx: r['metrics_norm']['Q5_face_size'] for r in scored}
    assert abs(q5[0] - q5[1]) < 0.05


def test_big_nima_difference_still_counts():
    cluster = PersonCluster(1)
    cluster.frames = [_frame(0, expr=0.02, size=0.7, nima=0.70),
                      _frame(1, expr=0.02, size=0.7, nima=0.40)]
    cluster.compute_centroids()
    assert pick_best(score_cluster(cluster))['frame'].file_idx == 0
