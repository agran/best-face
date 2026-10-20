import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from best_face import pick_face  # noqa: E402
from pipeline.cluster import FrameRecord, PersonCluster  # noqa: E402
from pipeline.detect import FaceDetectionResult  # noqa: E402
from pipeline.score import score_cluster  # noqa: E402


def _face(x, y, w, h, conf=0.9):
    return FaceDetectionResult((x, y, w, h), np.zeros((5, 2), np.float32), conf)


def test_pick_face_policies():
    big_edge = _face(0, 0, 200, 200, conf=0.6)
    small_center = _face(450, 450, 120, 120, conf=0.95)
    faces = sorted([big_edge, small_center], key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
    shape = (1000, 1000, 3)

    assert pick_face(faces, shape, 'largest') is big_edge
    assert pick_face(faces, shape, 'best_conf') is small_center
    assert pick_face(faces, shape, 'central') is small_center


def _record(idx, ear_valid, q3):
    f = FrameRecord(f'f{idx}.jpg', idx)
    f.det_conf = 0.9
    f.emb_res = np.ones(4, np.float32) / 2
    f.metrics = {
        'Q1_sharpness': 100.0, 'Q2_frontality': 0.9, 'Q3_eye_open': q3,
        'Q4_expression': 1.0, 'Q5_face_size': 0.5, 'Q6_exposure': 0.8,
        'Q7_confidence': 0.9, 'Q9_nima': 0.5, 'EAR_valid': ear_valid,
    }
    return f


def test_eye_gate_only_applies_to_real_ear():
    cluster = PersonCluster(1)
    cluster.frames = [_record(0, 1.0, 0.3), _record(1, 0.0, 0.3)]
    cluster.compute_centroids()
    res = {r['frame'].file_idx: r for r in score_cluster(cluster)}
    assert res[0]['rejected_reason'] == 'eyes_closed'
    assert res[1]['passed_gates']
