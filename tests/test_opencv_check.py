import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import best_face  # noqa: E402


def test_duplicate_opencv_is_reported(monkeypatch, capsys):
    monkeypatch.setattr(best_face, 'installed_opencv_packages',
                        lambda: ['opencv-python', 'opencv-contrib-python'])
    best_face.warn_if_duplicate_opencv()
    out = capsys.readouterr().out
    assert 'конфликтуют' in out
    assert 'pip uninstall -y opencv-python opencv-contrib-python' in out
    assert 'pip install opencv-contrib-python' in out


def test_single_opencv_is_silent(monkeypatch, capsys):
    monkeypatch.setattr(best_face, 'installed_opencv_packages', lambda: ['opencv-contrib-python'])
    best_face.warn_if_duplicate_opencv()
    assert capsys.readouterr().out == ''
