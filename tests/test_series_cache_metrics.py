import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.cache import FrameCache, make_signature  # noqa: E402
from pipeline.cluster import FrameRecord, cluster_frames  # noqa: E402
from pipeline.metrics import q3_eye_openness, q4_expression_neutrality  # noqa: E402


def _unit(v):
    return v / np.linalg.norm(v)


def _frame(idx, emb, t):
    f = FrameRecord(f'f{idx}.jpg', idx)
    f.emb_arc = emb.astype(np.float32)
    f.emb_res = f.emb_arc
    f.shot_time = t
    return f


def test_series_mode_splits_same_person_by_pause():
    rng = np.random.default_rng(5)
    a = _unit(rng.standard_normal(64))
    frames = []
    for i in range(6):
        frames.append(_frame(i, _unit(a + 0.05 * rng.standard_normal(64)), 100.0 + i * 1.5))
    for i in range(6):
        frames.append(_frame(6 + i, _unit(a + 0.05 * rng.standard_normal(64)), 160.0 + i * 1.5))

    person = cluster_frames(frames, threshold=0.5, time_far_sec=0, mode='person')
    series = cluster_frames(frames, threshold=0.5, mode='series', series_gap_sec=14.0)
    assert len(person) == 1
    assert len(series) == 2
    assert sorted(len(c.frames) for c in series) == [6, 6]


def test_series_mode_separates_interleaved_people():
    rng = np.random.default_rng(6)
    a = _unit(rng.standard_normal(64))
    b = _unit(rng.standard_normal(64))
    frames = []
    for i in range(8):
        base = a if i % 2 == 0 else b
        frames.append(_frame(i, _unit(base + 0.05 * rng.standard_normal(64)), 100.0 + i))
    series = cluster_frames(frames, threshold=0.5, mode='series', series_gap_sec=14.0)
    assert sorted(len(c.frames) for c in series) == [4, 4]


def test_frame_cache_roundtrip_and_invalidation(tmp_path):
    photo = tmp_path / 'a.jpg'
    photo.write_bytes(b'x')
    cache_file = str(tmp_path / 'cache.pkl')

    rec = FrameRecord(str(photo), 0)
    rec.det_conf = 0.9
    rec.emb_arc = np.ones(4, np.float32)

    cache = FrameCache(cache_file, make_signature(m=1))
    assert cache.get(str(photo)) is None
    cache.put(str(photo), rec)
    cache.save()

    again = FrameCache(cache_file, make_signature(m=1))
    got = again.get(str(photo))
    assert got is not None and got.det_conf == 0.9

    # другая сигнатура (смена модели) сбрасывает кэш
    assert FrameCache(cache_file, make_signature(m=2)).get(str(photo)) is None

    # изменение файла инвалидирует запись
    photo.write_bytes(b'xx')
    assert again.get(str(photo)) is None


def test_blendshapes_improve_eye_and_expression_scores():
    crop = np.zeros((112, 112, 3), np.uint8)
    kps = np.array([[38, 52], [74, 52], [56, 72], [42, 92], [70, 92]], np.float32)

    open_eyes = q3_eye_openness(crop, None, ear_value=0.30, blink=0.02)
    closed = q3_eye_openness(crop, None, ear_value=0.10, blink=0.95)
    assert open_eyes > 0.9 and closed < 0.3

    neutral = q4_expression_neutrality(crop, kps, jaw_open=0.0, smile=0.1)
    open_mouth = q4_expression_neutrality(crop, kps, jaw_open=0.5, smile=0.0)
    assert neutral > 0.9 and open_mouth < 0.2
