import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from best_face import FolderError, keep_person_ids, plan_swap  # noqa: E402
from pipeline.export import swap_best_photo  # noqa: E402


def _make_output(tmp_path, persons):
    photos = tmp_path / 'photos'
    photos.mkdir()
    out = tmp_path / 'out'
    (out / 'best').mkdir(parents=True)
    (out / 'report.json').write_text(json.dumps({
        'params': {'input_dir': str(photos)}, 'persons': persons}), encoding='utf-8')
    return out


def _person(pid, files, best, manual=False):
    return {'person_id': pid, 'best_photo': best, 'manual_choice': manual,
            'frames': [{'file': f} for f in files]}


def test_plan_swap_takes_next_in_ranking_and_keeps_manual_choices(tmp_path):
    out = _make_output(tmp_path, [_person(1, ['a.jpg', 'b.jpg'], 'a.jpg', manual=True),
                                  _person(2, ['c.jpg', 'd.jpg', 'e.jpg'], 'c.jpg')])
    f = out / 'best' / 'person0002_c.jpg'
    f.write_bytes(b'x')
    plan = plan_swap(str(f))
    assert plan['new_file'] == 'd.jpg' and plan['new_name'] == 'person0002_d.jpg'
    assert plan['choices'] == {'1': 'a.jpg', '2': 'd.jpg'}
    assert plan['output'] == str(out)


def test_plan_swap_keeps_manual_rename_prefix_and_suffix(tmp_path):
    out = _make_output(tmp_path, [_person(1, ['P1.JPG', 'P2.JPG'], 'P1.JPG')])
    f = out / 'best' / 'person0001_Иванов_P1.JPG'
    f.write_bytes(b'x')
    assert plan_swap(str(f))['new_name'] == 'person0001_Иванов_P2.JPG'
    g = out / 'best' / 'person0001_P1 11 Иванов Иван.JPG'
    g.write_bytes(b'x')
    assert plan_swap(str(g))['new_name'] == 'person0001_P2 11 Иванов Иван.JPG'


def test_plan_swap_goes_further_on_repeated_drag(tmp_path):
    out = _make_output(tmp_path, [_person(1, ['a.jpg', 'b.jpg', 'c.jpg'], 'b.jpg', manual=True)])
    f = out / 'best' / 'person0001_b.jpg'
    f.write_bytes(b'x')
    assert plan_swap(str(f))['new_file'] == 'c.jpg'


def test_plan_swap_last_frame_and_foreign_files_are_errors(tmp_path):
    out = _make_output(tmp_path, [_person(1, ['a.jpg'], 'a.jpg')])
    f = out / 'best' / 'person0001_a.jpg'
    f.write_bytes(b'x')
    with pytest.raises(FolderError, match='нет других кадров'):
        plan_swap(str(f))
    other = tmp_path / 'photo.jpg'
    other.write_bytes(b'x')
    with pytest.raises(FolderError):
        plan_swap(str(other))


def _cluster(*names):
    return SimpleNamespace(frames=[SimpleNamespace(path=os.path.join('d', n)) for n in names])


def test_keep_person_ids_restores_previous_numbering():
    ids = {1: frozenset({'a', 'b'}), 2: frozenset({'c'})}
    assert keep_person_ids([_cluster('c'), _cluster('b', 'a')], ids) == [1, 0]
    assert keep_person_ids([_cluster('c'), _cluster('a')], ids) is None


def test_swap_moves_only_old_file_to_backup(tmp_path):
    best = tmp_path / 'best'
    best.mkdir()
    old, other = best / 'person0001_a.jpg', best / 'person0002_Петров_x.jpg'
    old.write_bytes(b'old')
    other.write_bytes(b'other')
    src = tmp_path / 'b.jpg'
    src.write_bytes(b'new')
    new = swap_best_photo(str(best), str(old), str(src), 'person0001_b.jpg')
    assert open(new, 'rb').read() == b'new'
    assert other.exists() and not old.exists()
    assert list((tmp_path / 'best_prev').rglob('person0001_a.jpg'))
