#!/usr/bin/env python3
"""
download_models.py — проверить модели в папке models/ и докачать недостающие.

  python download_models.py            # проверить и скачать недостающие модели
  python download_models.py --force    # перекачать заново

Основная задача — скачать `w600k_mbf.onnx` (ArcFace, InsightFace): эта модель не
распространяется в репозитории, потому что разрешена только для некоммерческого
исследовательского использования (см. THIRD_PARTY_LICENSES.md). Из архива buffalo_s.zip
(127 МБ) скачивается только нужный файл.

Остальные модели лежат в репозитории (models/); если файл удалён или повреждён, YuNet и
MediaPipe докачиваются из официальных источников (проверяется SHA-256).
NIMA и FaceRes — необязательные модели, у них нет внешнего источника: они берутся из репозитория.
"""

import argparse
import hashlib
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

MODELS_DIR = Path(__file__).resolve().parent / 'models'

_YUNET = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
_BUFFALO = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip'
_LANDMARKER = ('https://storage.googleapis.com/mediapipe-models/face_landmarker/'
               'face_landmarker/float16/1/face_landmarker.task')

# имя файла -> (источник, имя файла внутри zip или None, sha256)
MODELS = {
    'face_detection_yunet_2023mar.onnx': (
        _YUNET, None, '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'),
    'w600k_mbf.onnx': (
        _BUFFALO, 'w600k_mbf.onnx', '9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f'),
    'face_landmarker.task': (
        _LANDMARKER, None, '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'),
}

_UA = {'User-Agent': 'best-face-model-downloader'}


class _HttpRangeFile(io.RawIOBase):
    """Файлоподобный объект поверх HTTP Range: zipfile читает из большого архива только нужное."""

    def __init__(self, url: str):
        self._url = url
        self._pos = 0
        req = urllib.request.Request(url, headers={**_UA, 'Range': 'bytes=0-0'})
        with urllib.request.urlopen(req, timeout=60) as r:
            if r.status != 206:
                raise OSError('сервер не поддерживает Range')
            self._size = int(r.headers['Content-Range'].rsplit('/', 1)[1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def readinto(self, b):
        n = min(len(b), self._size - self._pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(
            self._url, headers={**_UA, 'Range': f'bytes={self._pos}-{self._pos + n - 1}'})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        b[:len(data)] = data
        self._pos += len(data)
        return len(data)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst: Path) -> None:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, 'wb') as out:
        total = int(r.headers.get('Content-Length') or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f'\r    {done / 1e6:.1f} / {total / 1e6:.1f} МБ', end='', flush=True)
    print()


def _fetch_from_zip(url: str, member: str, dst: Path) -> None:
    try:
        with zipfile.ZipFile(_HttpRangeFile(url)) as z, z.open(member) as src, open(dst, 'wb') as out:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    except OSError:
        # Range недоступен — качаем архив целиком во временный файл
        tmp = dst.with_suffix('.zip.part')
        print('    Range не поддерживается, скачиваю архив целиком (127 МБ)...')
        _download(url, tmp)
        try:
            with zipfile.ZipFile(tmp) as z, z.open(member) as src, open(dst, 'wb') as out:
                out.write(src.read())
        finally:
            tmp.unlink(missing_ok=True)


def ensure_model(name: str, force: bool = False, models_dir: Optional[Path] = None) -> bool:
    """
    Убедиться, что модель name есть в models/ и цела; при необходимости скачать.
    Создаёт папку models/, если её нет. Возвращает True при успехе.
    """
    if name not in MODELS:
        return False
    url, member, sha = MODELS[name]
    models_dir = models_dir or MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    dst = models_dir / name

    if dst.exists() and not force:
        if _sha256(dst) == sha:
            return True
        print(f'  {name}: контрольная сумма не совпала, скачиваю заново')

    print(f'  Скачиваю {name} ...')
    if name == 'w600k_mbf.onnx':
        print('  Внимание: эта модель (InsightFace) разрешена только для некоммерческого '
              'исследовательского использования, см. THIRD_PARTY_LICENSES.md')
    tmp = dst.with_name(dst.name + '.part')
    try:
        if member:
            _fetch_from_zip(url, member, tmp)
        else:
            _download(url, tmp)
        if _sha256(tmp) != sha:
            print(f'  ОШИБКА: {name}: контрольная сумма не совпала (файл повреждён или изменился на сервере)')
            tmp.unlink(missing_ok=True)
            return False
        os.replace(tmp, dst)
        print(f'  OK: {dst}')
        return True
    except Exception as e:
        print(f'  ОШИБКА: не удалось скачать {name}: {e}')
        tmp.unlink(missing_ok=True)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description='Скачать модели Best-Face в папку models/')
    ap.add_argument('--force', action='store_true', help='перекачать, даже если файлы есть')
    args = ap.parse_args()

    print(f'Папка моделей: {MODELS_DIR}')
    ok = all([ensure_model(name, args.force) for name in MODELS])
    if ok:
        print('Все модели на месте.')
        return 0
    print('Не все модели удалось скачать: проверьте интернет и повторите, '
          'либо положите файлы в папку models/ вручную (см. README.md).')
    return 1


if __name__ == '__main__':
    sys.exit(main())
