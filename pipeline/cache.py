"""
pipeline/cache.py — кэш результатов обработки кадров (детекция, эмбеддинги, метрики).

Первый этап пайплайна (чтение JPEG, детекция, эмбеддинги, MediaPipe) занимает
основное время. Кэш позволяет заново кластеризовать и подбирать пороги/веса за секунды.

Ключ записи — путь + mtime + размер файла. Весь кэш сбрасывается, если изменилась
сигнатура (модели, параметры детекции, версия формата).
"""

import hashlib
import json
import os
import pickle
from typing import Optional

from .cluster import FrameRecord

CACHE_VERSION = 4


def make_signature(**parts) -> str:
    payload = json.dumps({'v': CACHE_VERSION, **parts}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()


def file_fingerprint(path: Optional[str]) -> Optional[list]:
    """Размер и mtime файла модели — чтобы смена модели сбрасывала кэш."""
    if not path or not os.path.exists(path):
        return None
    st = os.stat(path)
    return [os.path.basename(path), st.st_size, st.st_mtime_ns]


class FrameCache:
    def __init__(self, path: str, signature: str, enabled: bool = True):
        self._path = path
        self._sig = signature
        self._enabled = enabled
        self._records = {}
        self._dirty = False
        if enabled and os.path.exists(path):
            try:
                with open(path, 'rb') as fh:
                    data = pickle.load(fh)
                if data.get('signature') == signature:
                    self._records = data.get('records', {})
            except Exception:
                self._records = {}

    @staticmethod
    def _key(path: str) -> Optional[str]:
        try:
            st = os.stat(path)
        except OSError:
            return None
        return f'{os.path.abspath(path)}|{st.st_mtime_ns}|{st.st_size}'

    def __len__(self) -> int:
        return len(self._records)

    def get(self, path: str) -> Optional[FrameRecord]:
        if not self._enabled:
            return None
        key = self._key(path)
        return self._records.get(key) if key else None

    def put(self, path: str, rec: FrameRecord) -> None:
        if not self._enabled:
            return
        key = self._key(path)
        if key:
            self._records[key] = rec
            self._dirty = True

    def save(self) -> None:
        if not (self._enabled and self._dirty):
            return
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        tmp = self._path + '.tmp'
        with open(tmp, 'wb') as fh:
            pickle.dump({'signature': self._sig, 'records': self._records}, fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self._path)
        self._dirty = False
