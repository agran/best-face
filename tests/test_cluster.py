import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.cluster import FrameRecord, cluster_frames, _robust_centroid  # noqa: E402


def _unit(v):
    return v / np.linalg.norm(v)


def _frame(idx, emb, t=None):
    f = FrameRecord(f'f{idx}.jpg', idx)
    f.emb_arc = emb.astype(np.float32)
    f.emb_res = emb.astype(np.float32)
    f.shot_time = t
    return f


def _person_frames(rng, base, start_idx, count, noise=0.05, t0=None):
    frames = []
    for i in range(count):
        emb = _unit(base + noise * rng.standard_normal(base.shape))
        t = None if t0 is None else t0 + i * 1.5
        frames.append(_frame(start_idx + i, emb, t))
    return frames


def test_separates_people_and_bridge_does_not_merge():
    rng = np.random.default_rng(0)
    a = _unit(rng.standard_normal(64))
    b = _unit(rng.standard_normal(64))
    frames = _person_frames(rng, a, 0, 8) + _person_frames(rng, b, 8, 8)
    # одиночный «мост» между двумя разными людьми
    frames.append(_frame(16, _unit(a + b)))

    clusters = cluster_frames(frames, threshold=0.6, threshold_time=0.5)
    big = [c for c in clusters if len(c.frames) >= 8]
    assert len(big) == 2
    for c in big:
        ids = {f.file_idx // 8 for f in c.frames if f.file_idx < 16}
        assert len(ids) == 1


def test_far_in_time_series_are_split():
    rng = np.random.default_rng(1)
    a = _unit(rng.standard_normal(64))
    frames = (_person_frames(rng, a, 0, 5, t0=1000.0)
              + _person_frames(rng, a, 5, 5, t0=5000.0))
    clusters = cluster_frames(frames, threshold=0.6, time_far_sec=120, time_far_penalty=0.9)
    assert len(clusters) == 2

    merged = cluster_frames(frames, threshold=0.6, time_far_sec=0)
    assert len(merged) == 1


def test_low_cohesion_cluster_is_split():
    rng = np.random.default_rng(2)
    base = _unit(rng.standard_normal(64))
    # два близких, но различимых «человека»
    other = _unit(base + 0.9 * _unit(rng.standard_normal(64)))
    frames = _person_frames(rng, base, 0, 10, noise=0.05) + _person_frames(rng, other, 10, 10, noise=0.05)

    loose = cluster_frames(frames, threshold=0.3)
    assert len(loose) == 1

    split = cluster_frames(frames, threshold=0.3, min_cohesion=0.8, split_step=0.1)
    assert len(split) == 2
    assert not any(c.low_cohesion for c in split)


def test_large_cohesive_series_is_not_split():
    rng = np.random.default_rng(4)
    a = _unit(rng.standard_normal(64))
    frames = _person_frames(rng, a, 0, 60)
    clusters = cluster_frames(frames, threshold=0.5, min_cohesion=0.6)
    assert len(clusters) == 1
    assert not clusters[0].low_cohesion


def test_robust_centroid_ignores_outlier():
    rng = np.random.default_rng(3)
    base = _unit(rng.standard_normal(64))
    embs = [_unit(base + 0.05 * rng.standard_normal(64)) for _ in range(9)]
    outlier = _unit(rng.standard_normal(64))
    plain = _unit(np.mean(embs + [outlier], axis=0))
    robust = _robust_centroid(embs + [outlier])
    assert float(robust @ base) > float(plain @ base)
