import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.cluster import FrameRecord, PersonCluster  # noqa: E402
from pipeline.metrics import gaze_deviation, gaze_score, neutral_score, pose_score  # noqa: E402
from pipeline.score import pick_best, score_cluster  # noqa: E402


def _frame(idx, expr=0.1, gaze_x=0.0, yaw=0.0, pitch=0.0, roll=0.0, jaw=0.0, sharp=100.0):
    f = FrameRecord(f'f{idx}.jpg', idx)
    f.det_conf = 0.9
    f.emb_res = np.ones(4, np.float32) / 2
    f.yaw_deg, f.pitch_deg, f.roll_deg = yaw, pitch, roll
    f.metrics = {
        'Q1_sharpness': sharp, 'Q2_frontality': 0.95, 'Q3_eye_open': 1.0,
        'Q4_expression': 1.0, 'Q5_face_size': 0.5, 'Q6_exposure': 0.8,
        'Q7_confidence': 0.9, 'Q9_nima': 0.5, 'EAR_valid': 1.0,
        'EAR_raw': 0.33, 'BLINK_raw': 0.05, 'SQUINT_raw': 0.15,
        'GAZE_X_raw': gaze_x, 'GAZE_Y_raw': -0.14, 'LOOK_H_raw': 0.05, 'LOOK_V_raw': 0.1,
        'EXPR_raw': expr, 'JAW_raw': jaw, 'SMILE_raw': 0.0,
    }
    return f


def test_gaze_accounts_for_head_turn():
    # голова повёрнута на 8°, зрачки компенсируют поворот → взгляд в камеру
    assert gaze_score(gaze_deviation(-0.0103 * 8, 8.0, 0.05)) > 0.95
    # зрачок смещён в сторону при прямой голове
    assert gaze_score(gaze_deviation(0.25, 0.0, 0.05)) < 0.2
    assert gaze_score(None) == 0.5


def test_pose_and_neutral_scores():
    assert pose_score(0, 0, 0) > 0.99
    assert pose_score(0, 0, 12) < pose_score(0, 0, 3) < 1.0   # наклон головы штрафуется
    assert pose_score(15, 0, 0) < 0.3
    assert neutral_score(0.05) == 1.0
    assert neutral_score(0.5) > neutral_score(0.9) > 0.0   # не насыщается даже при сильной улыбке


def test_serious_straight_frame_beats_smiling_sharper_one():
    cluster = PersonCluster(1)
    cluster.frames = [_frame(0, expr=0.9, sharp=130.0),               # улыбка, чуть резче
                      _frame(1, expr=0.05, sharp=100.0),               # серьёзное
                      _frame(2, expr=0.06, gaze_x=0.30, sharp=105.0),  # серьёзное, но смотрит в сторону
                      _frame(3, expr=0.08, roll=11.0, sharp=104.0)]    # серьёзное, голова наклонена
    cluster.compute_centroids()
    scored = score_cluster(cluster)
    assert pick_best(scored)['frame'].file_idx == 1
    reasons = {r['frame'].file_idx: r['rejected_reason'] for r in scored}
    assert reasons[0] == 'not_neutral'
    assert reasons[2] == 'gaze_off'


def test_open_mouth_is_hard_gate_when_alternatives_exist():
    cluster = PersonCluster(1)
    cluster.frames = [_frame(0, expr=0.5, jaw=0.3), _frame(1, expr=0.5, jaw=0.0)]
    cluster.compute_centroids()
    scored = score_cluster(cluster)
    assert pick_best(scored)['frame'].file_idx == 1
    assert next(r for r in scored if r['frame'].file_idx == 0)['rejected_reason'] == 'mouth_open'


def test_all_smiling_series_still_picks_the_least_smiling():
    cluster = PersonCluster(1)
    cluster.frames = [_frame(0, expr=0.9), _frame(1, expr=0.8), _frame(2, expr=0.85)]
    cluster.compute_centroids()
    scored = score_cluster(cluster)
    assert pick_best(scored)['frame'].file_idx == 1
