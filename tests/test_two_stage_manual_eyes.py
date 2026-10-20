import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from best_face import apply_manual_choices  # noqa: E402
from pipeline.cluster import FrameRecord, PersonCluster, cluster_frames  # noqa: E402
from pipeline.export import copy_best_photos  # noqa: E402
from pipeline.score import mark_manual, pick_best, score_cluster  # noqa: E402


def _unit(v):
    return v / np.linalg.norm(v)


def _frame(idx, emb, t=None):
    f = FrameRecord(f'f{idx}.jpg', idx)
    f.emb_arc = emb.astype(np.float32)
    f.emb_res = f.emb_arc
    f.shot_time = t
    return f


def test_two_stage_matches_dense_clustering():
    rng = np.random.default_rng(10)
    people = [_unit(rng.standard_normal(64)) for _ in range(3)]
    frames, idx = [], 0
    for visit in range(4):          # каждый человек приходит несколько раз
        for p in people:
            for _ in range(5):
                frames.append(_frame(idx, _unit(p + 0.05 * rng.standard_normal(64)),
                                     1000.0 * idx))
                idx += 1

    dense = cluster_frames(frames, threshold=0.5, mode='person', time_far_penalty=0.0)
    staged = cluster_frames(frames, threshold=0.5, mode='person', time_far_penalty=0.0,
                            dense_limit=20)
    assert sorted(len(c.frames) for c in dense) == [20, 20, 20]
    assert sorted(len(c.frames) for c in staged) == [20, 20, 20]


def test_two_stage_without_exif_chunks_by_order():
    rng = np.random.default_rng(11)
    a = _unit(rng.standard_normal(64))
    b = _unit(rng.standard_normal(64))
    frames = [_frame(i, _unit((a if i % 2 else b) + 0.05 * rng.standard_normal(64)))
              for i in range(40)]
    staged = cluster_frames(frames, threshold=0.5, mode='person', dense_limit=10)
    assert sorted(len(c.frames) for c in staged) == [20, 20]


def _record(idx, q3=0.9, ear_valid=1.0):
    f = FrameRecord(f'f{idx}.jpg', idx)
    f.det_conf = 0.9
    f.emb_res = np.ones(4, np.float32) / 2
    f.metrics = {
        'Q1_sharpness': 100.0 + idx, 'Q2_frontality': 0.9, 'Q3_eye_open': q3,
        'Q4_expression': 1.0, 'Q5_face_size': 0.5, 'Q6_exposure': 0.8,
        'Q7_confidence': 0.9, 'Q9_nima': 0.5, 'EAR_valid': ear_valid,
    }
    return f


def test_narrow_eyes_person_is_not_flagged_as_closed():
    cluster = PersonCluster(1)
    cluster.frames = [_record(i, q3=0.50) for i in range(4)] + [_record(4, q3=0.15)]
    cluster.compute_centroids()
    res = {r['frame'].file_idx: r for r in score_cluster(cluster)}
    assert all(res[i]['passed_gates'] for i in range(4))
    assert res[4]['rejected_reason'] == 'eyes_closed'


def test_manual_choice_overrides_auto_pick(tmp_path):
    cluster = PersonCluster(1)
    cluster.frames = [_record(i) for i in range(3)]
    cluster.compute_centroids()
    scored = score_cluster(cluster)
    auto = pick_best(scored)['frame'].path
    other = next(r['frame'].path for r in scored if r['frame'].path != auto)

    choices = tmp_path / 'manual_choices.json'
    choices.write_text(json.dumps({'choices': {'1': other}}), encoding='utf-8')
    apply_manual_choices(str(choices), [scored])
    assert pick_best(scored)['frame'].path == other
    assert not mark_manual(scored, 'нет_такого.jpg')


def test_copy_best_removes_stale_results(tmp_path):
    photo = tmp_path / 'a.jpg'
    photo.write_bytes(b'x')
    best = tmp_path / 'best'
    best.mkdir()
    (best / 'person0099_old.jpg').write_bytes(b'old')
    (best / 'keep.txt').write_bytes(b'k')

    rec = _record(0)
    rec.path = str(photo)
    cluster = PersonCluster(1)
    cluster.frames = [rec]
    cluster.compute_centroids()
    scored = score_cluster(cluster)
    copy_best_photos([cluster], {1: scored}, str(best), str(tmp_path / 'rej'), [])

    names = sorted(p.name for p in best.iterdir())
    assert names == ['keep.txt', 'person0001_a.jpg']
