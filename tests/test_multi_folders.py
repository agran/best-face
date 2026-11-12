import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from best_face import FolderError, main, resolve_job  # noqa: E402


def _args(**kw):
    return SimpleNamespace(**{'output': None, 'choices': None, **kw})


def test_default_output_is_none_so_each_folder_uses_its_own_subfolder(tmp_path):
    a = tmp_path / 'A (1)'
    a.mkdir()
    assert resolve_job(str(a), _args(), set(), multi=True) == (str(a), None, None)


def test_output_for_many_folders_gets_unique_subfolders(tmp_path):
    one, two = tmp_path / 'x' / 'class', tmp_path / 'y' / 'class'
    one.mkdir(parents=True)
    two.mkdir(parents=True)
    used: set = set()
    args = _args(output=str(tmp_path / 'out'))
    out1 = resolve_job(str(one), args, used, multi=True)[1]
    out2 = resolve_job(str(two), args, used, multi=True)[1]
    assert out1 == str(tmp_path / 'out' / 'class')
    assert out2 == str(tmp_path / 'out' / 'class_2')


def test_output_for_single_folder_is_used_as_is(tmp_path):
    a = tmp_path / 'a'
    a.mkdir()
    args = _args(output=str(tmp_path / 'out'))
    assert resolve_job(str(a), args, set(), multi=False)[1] == str(tmp_path / 'out')


def test_json_among_folders_brings_its_own_folders_and_choices(tmp_path):
    photos = tmp_path / 'photos'
    photos.mkdir()
    out = tmp_path / 'res'
    js = tmp_path / 'manual_choices.json'
    js.write_text(json.dumps({'input_dir': str(photos), 'output_dir': str(out),
                              'choices': {'1': 'a.jpg'}}), encoding='utf-8')
    assert resolve_job(str(js), _args(), set(), multi=True) == (str(photos), str(out), str(js))


def test_not_a_folder_nor_json_is_a_folder_error(tmp_path):
    f = tmp_path / 'x.txt'
    f.write_text('x')
    with pytest.raises(FolderError):
        resolve_job(str(f), _args(), set(), multi=True)


def test_choices_flag_with_several_folders_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['best_face.py', str(tmp_path), str(tmp_path), '--choices', 'c.json'])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 64
