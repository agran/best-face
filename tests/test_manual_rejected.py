import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.cluster import FrameRecord, PersonCluster  # noqa: E402
from pipeline.export import REASON_RU, build_contact_sheet, build_report  # noqa: E402
from pipeline.score import pick_best, mark_manual, score_cluster  # noqa: E402


def _frame(idx, ear, blink=0.02):
    f = FrameRecord(f'f{idx}.jpg', idx)
    f.det_conf = 0.9
    f.emb_res = np.ones(4, np.float32) / 2
    f.metrics = {
        'Q1_sharpness': 100.0, 'Q2_frontality': 0.95, 'Q3_eye_open': 1.0,
        'Q4_expression': 1.0, 'Q5_face_size': 0.5, 'Q6_exposure': 0.8,
        'Q7_confidence': 0.9, 'Q9_nima': 0.5, 'EAR_valid': 1.0,
        'EAR_raw': ear, 'BLINK_raw': blink, 'SQUINT_raw': 0.15,
    }
    return f


def _cluster():
    cl = PersonCluster(1)
    cl.frames = [_frame(0, 0.33), _frame(1, 0.33), _frame(2, 0.10, blink=0.9)]   # кадр 2 — «глаза закрыты»
    cl.compute_centroids()
    return cl


def test_rejected_frame_can_be_chosen_manually_and_wins():
    cl = _cluster()
    scored = score_cluster(cl)
    blink = next(r for r in scored if r['frame'].file_idx == 2)
    assert not blink['passed_gates'] and blink['rejected_reason'] == 'eyes_closed'

    assert mark_manual(scored, 'f2.jpg')
    assert pick_best(scored)['frame'].file_idx == 2

    rep = build_report([cl], {1: scored}, [], {'input_dir': 'x', 'output_dir': 'y'})
    p = rep['persons'][0]
    assert p['best_photo'] == 'f2.jpg' and p['manual_choice'] is True
    assert p['low_quality_flag'] is False and p['close_call'] is False
    assert next(f for f in p['frames'] if f['file'] == 'f2.jpg')['selected'] is True


def test_contact_sheet_marks_rejected_frames_and_lets_chosen_state_win_css(tmp_path):
    cl = _cluster()
    scored = score_cluster(cl)
    rep = build_report([cl], {1: scored}, [], {'input_dir': 'x', 'output_dir': 'y'})
    out = tmp_path / 'sheet.html'
    build_contact_sheet([cl], {1: scored}, [], rep, str(out))
    html = out.read_text(encoding='utf-8')

    # причина показана по-русски, а не кодом
    assert REASON_RU['eyes_closed'] in html and '>eyes_closed' not in html
    # «выбрано вами» объявлено после правил best/rejected, иначе выбор забракованного кадра не виден
    css = html[:html.index('<script>')]
    assert css.index('.frame.rejected {') < css.index('.frame.chosen {')
    assert css.index('.frame.best {') < css.index('.frame.chosen {')
    assert 'ВАШ ВЫБОР' in html
