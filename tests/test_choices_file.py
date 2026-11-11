import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from best_face import unpack_choices_file  # noqa: E402
from pipeline.export import render_html_tail  # noqa: E402


def _write(tmp_path, data):
    p = tmp_path / 'manual_choices.json'
    p.write_text(json.dumps(data), encoding='utf-8')
    return str(p)


def test_choices_file_gives_folders(tmp_path):
    photos = tmp_path / 'Папка (26)'
    photos.mkdir()
    out = tmp_path / 'out'
    path = _write(tmp_path, {'input_dir': str(photos), 'output_dir': str(out),
                             'choices': {'1': 'a.jpg'}})
    assert unpack_choices_file(path) == (str(photos), str(out))


def test_choices_file_errors_are_explained(tmp_path):
    with pytest.raises(ValueError, match='choices'):
        unpack_choices_file(_write(tmp_path, {'foo': 1}))
    with pytest.raises(ValueError, match='старой версией'):
        unpack_choices_file(_write(tmp_path, {'choices': {'1': 'a.jpg'}}))
    with pytest.raises(ValueError, match='недоступна'):
        unpack_choices_file(_write(tmp_path, {'input_dir': str(tmp_path / 'нет'),
                                              'choices': {}}))


def test_contact_sheet_embeds_folders_for_saved_json():
    meta = {'input_dir': r'F:\Фото\Новая папка (26)', 'output_dir': r'F:\Фото\out</script>'}
    html = render_html_tail(meta)
    m = re.search(r'const META = (\{.*?\});\n', html, re.S)
    assert m and json.loads(m.group(1).replace('<\\/', '</')) == meta
    assert '</script>' not in m.group(1)          # значение не должно ломать тег script
    assert '{...META, choices}' in html           # сохранённый JSON содержит META
