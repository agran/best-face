import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import cache as cache_mod  # noqa: E402
from pipeline.cache import FrameCache, make_signature  # noqa: E402
from pipeline.cluster import FrameRecord, PersonCluster  # noqa: E402
from pipeline.export import copy_best_photos  # noqa: E402
from pipeline.score import score_cluster  # noqa: E402


def _record(path, idx=0):
    r = FrameRecord(str(path), idx)
    r.det_conf = 0.93
    r.bbox = (10, 20, 100, 120)
    r.kps = np.arange(10, dtype=np.float32).reshape(5, 2)
    r.emb_arc = np.linspace(0, 1, 8, dtype=np.float32)
    r.emb_res = np.linspace(1, 2, 16, dtype=np.float32)
    r.metrics = {'Q1_sharpness': 12.5, 'EAR_raw': 0.31}
    r.shot_time = 1234.5
    r.pitch_deg, r.roll_deg = 1.5, -2.0
    r.gender, r.age = 0.7, 11.5
    return r


def test_cache_roundtrips_without_pickle(tmp_path):
    photo = tmp_path / 'a.jpg'
    photo.write_bytes(b'x')
    rejected = tmp_path / 'b.jpg'
    rejected.write_bytes(b'yy')
    f = str(tmp_path / 'cache.npz')
    sig = make_signature(m=1)

    c = FrameCache(f, sig)
    c.put(str(photo), _record(photo))
    bad = FrameRecord(str(rejected), 1)
    bad.rejected_reason = 'no_face'
    c.put(str(rejected), bad)
    c.save()

    # файл читается numpy без pickle (чужой кэш не может выполнить код)
    with np.load(f, allow_pickle=False) as z:
        assert set(z.files) == {'header', 'kps', 'arc', 'res'}

    got = FrameCache(f, sig).get(str(photo))
    assert got.bbox == (10, 20, 100, 120) and got.det_conf == 0.93
    assert np.allclose(got.emb_arc, np.linspace(0, 1, 8)) and got.emb_res.shape == (16,)
    assert np.allclose(got.kps, np.arange(10).reshape(5, 2))
    assert got.metrics['EAR_raw'] == 0.31 and got.pitch_deg == 1.5 and got.age == 11.5
    rej = FrameCache(f, sig).get(str(rejected))
    assert rej.rejected_reason == 'no_face' and rej.emb_arc is None and rej.kps is None


def test_cache_ignores_garbage_and_foreign_pickle(tmp_path):
    photo = tmp_path / 'a.jpg'
    photo.write_bytes(b'x')
    f = tmp_path / 'cache.npz'
    f.write_bytes(b'\x80\x04\x95not-a-real-cache')   # мусор / pickle-подобные байты
    c = FrameCache(str(f), make_signature(m=1))
    assert c.get(str(photo)) is None and len(c) == 0


def test_signature_changes_with_pipeline_code(monkeypatch):
    before = make_signature(m=1)
    monkeypatch.setattr(cache_mod, 'code_fingerprint', lambda: 'другой-код')
    assert make_signature(m=1) != before


def test_read_errors_are_not_cached(tmp_path):
    photo = tmp_path / 'a.jpg'
    photo.write_bytes(b'x')
    c = FrameCache(str(tmp_path / 'c.npz'), make_signature(m=1))
    r = FrameRecord(str(photo), 0)
    r.rejected_reason = 'read_error'
    c.put(str(photo), r)
    assert c.get(str(photo)) is None


def _scored(path):
    f = FrameRecord(str(path), 0)
    f.det_conf = 0.9
    f.emb_res = np.ones(4, np.float32) / 2
    f.metrics = {'Q1_sharpness': 1.0, 'Q2_frontality': 1.0, 'Q3_eye_open': 1.0,
                 'Q4_expression': 1.0, 'Q5_face_size': 1.0, 'Q6_exposure': 1.0,
                 'Q7_confidence': 1.0, 'Q9_nima': 0.5}
    cl = PersonCluster(1)
    cl.frames = [f]
    cl.compute_centroids()
    return cl, score_cluster(cl)


def test_rerun_keeps_users_renamed_files(tmp_path):
    photo = tmp_path / 'IMG_1.jpg'
    photo.write_bytes(b'photo')
    best = tmp_path / 'out' / 'best'
    best.mkdir(parents=True)
    renamed = best / 'person0001_IMG_1 3Б Иванов Иван.jpg'   # имя дополнено вручную
    renamed.write_bytes(b'user-work')
    (best / 'notes.txt').write_text('keep')

    cl, scored = _scored(photo)
    copy_best_photos([cl], {1: scored}, str(best), str(tmp_path / 'out' / 'rejected'), [])

    names = sorted(p.name for p in best.iterdir())
    assert names == ['notes.txt', 'person0001_IMG_1.jpg']
    backups = list((tmp_path / 'out' / 'best_prev').rglob('*.jpg'))
    assert [b.name for b in backups] == [renamed.name]
    assert backups[0].read_bytes() == b'user-work'


def test_only_last_three_backups_are_kept(tmp_path):
    photo = tmp_path / 'IMG_1.jpg'
    photo.write_bytes(b'photo')
    out = tmp_path / 'out'
    best = out / 'best'
    best.mkdir(parents=True)
    cl, scored = _scored(photo)
    for i in range(5):
        copy_best_photos([cl], {1: scored}, str(best), str(out / 'rejected'), [])
        time.sleep(1.1)   # имя копии — секундная метка времени
    assert len([d for d in (out / 'best_prev').iterdir() if d.is_dir()]) == 3
