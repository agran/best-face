"""
pipeline/cache.py — кэш результатов обработки кадров (детекция, эмбеддинги, метрики).

Первый этап пайплайна (чтение JPEG, детекция, эмбеддинги, MediaPipe) занимает
основное время. Кэш позволяет заново кластеризовать и подбирать пороги/веса за секунды.

Ключ записи — путь + mtime + размер файла. Весь кэш сбрасывается, если изменилась
сигнатура: модели, параметры детекции, версия формата и исходный код этапов, от
которых зависят значения (detect/align/embed/ear/metrics/exif).

Формат — .npz без pickle (numpy.load(allow_pickle=False)): кэш лежит рядом с чужими
фотографиями, и загрузка pickle из недоверенной папки позволила бы выполнить произвольный код.
"""

import hashlib
import json
import os
from typing import Optional

import numpy as np

from .cluster import FrameRecord

CACHE_VERSION = 8

_CODE_FILES = ('detect.py', 'align.py', 'embed.py', 'ear.py', 'metrics.py', 'exif.py')


def code_fingerprint() -> str:
    """Хэш исходников этапов, определяющих кэшируемые значения (меняется код — кэш устаревает)."""
    h = hashlib.sha1()
    here = os.path.dirname(os.path.abspath(__file__))
    for name in _CODE_FILES:
        try:
            with open(os.path.join(here, name), 'rb') as f:
                # переводы строк не должны влиять на хэш (CRLF/LF)
                h.update(f.read().replace(b'\r\n', b'\n'))
        except OSError:
            h.update(name.encode())
    return h.hexdigest()


def make_signature(**parts) -> str:
    payload = json.dumps({'v': CACHE_VERSION, 'code': code_fingerprint(), **parts},
                         sort_keys=True, default=str)
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()


def file_fingerprint(path: Optional[str]) -> Optional[list]:
    """Размер и mtime файла модели — чтобы смена модели сбрасывала кэш."""
    if not path or not os.path.exists(path):
        return None
    st = os.stat(path)
    return [os.path.basename(path), st.st_size, st.st_mtime_ns]


def _num(x):
    return None if x is None else float(x)


def _dump_meta(key: str, r: FrameRecord) -> dict:
    return {
        'key': key, 'path': r.path, 'file_idx': int(r.file_idx),
        'shot_time': _num(r.shot_time),
        'bbox': None if r.bbox is None else [int(v) for v in r.bbox],
        'det_conf': float(r.det_conf),
        'gender': _num(r.gender), 'age': _num(r.age),
        'metrics': {k: float(v) for k, v in r.metrics.items()},
        'yaw_deg': float(r.yaw_deg), 'frontal_ratio': float(r.frontal_ratio),
        'pitch_deg': _num(r.pitch_deg), 'roll_deg': _num(r.roll_deg),
        'rejected_reason': r.rejected_reason,
        'has_kps': r.kps is not None, 'has_arc': r.emb_arc is not None,
        'has_res': r.emb_res is not None,
    }


class FrameCache:
    def __init__(self, path: str, signature: str, enabled: bool = True):
        self._path = path
        self._sig = signature
        self._enabled = enabled
        self._records = {}
        self._dirty = False
        if enabled and os.path.exists(path):
            try:
                self._records = self._load(path, signature)
            except Exception:
                # повреждённый или чужой формат кэша — просто считаем заново
                self._records = {}

    @staticmethod
    def _load(path: str, signature: str) -> dict:
        with np.load(path, allow_pickle=False) as z:
            header = json.loads(str(z['header']))
            if header.get('signature') != signature:
                return {}
            kps, arc, res = z['kps'], z['arc'], z['res']
            out = {}
            for i, m in enumerate(header['records']):
                r = FrameRecord(m['path'], m['file_idx'])
                r.shot_time = m['shot_time']
                r.bbox = None if m['bbox'] is None else tuple(m['bbox'])
                r.kps = kps[i].copy() if m['has_kps'] else None
                r.det_conf = m['det_conf']
                r.emb_arc = arc[i].copy() if m['has_arc'] else None
                r.emb_res = res[i].copy() if m['has_res'] else None
                r.gender, r.age = m['gender'], m['age']
                r.metrics = dict(m['metrics'])
                r.yaw_deg, r.frontal_ratio = m['yaw_deg'], m['frontal_ratio']
                r.pitch_deg, r.roll_deg = m['pitch_deg'], m['roll_deg']
                r.rejected_reason = m['rejected_reason']
                out[m['key']] = r
            return out

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
        # Ошибку чтения не запоминаем: файл мог быть временно занят
        if rec.rejected_reason == 'read_error':
            return
        key = self._key(path)
        if key:
            self._records[key] = rec
            self._dirty = True

    def save(self) -> None:
        if not (self._enabled and self._dirty):
            return
        items = list(self._records.items())
        arc_dim = next((r.emb_arc.shape[0] for _, r in items if r.emb_arc is not None), 0)
        res_dim = next((r.emb_res.shape[0] for _, r in items if r.emb_res is not None), 0)
        n = len(items)
        kps = np.zeros((n, 5, 2), np.float32)
        arc = np.zeros((n, arc_dim), np.float32)
        res = np.zeros((n, res_dim), np.float32)
        for i, (_, r) in enumerate(items):
            if r.kps is not None:
                kps[i] = r.kps
            if r.emb_arc is not None:
                arc[i] = r.emb_arc
            if r.emb_res is not None:
                res[i] = r.emb_res
        header = {'signature': self._sig,
                  'records': [_dump_meta(k, r) for k, r in items]}

        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        tmp = self._path + '.tmp.npz'
        np.savez_compressed(tmp, header=np.array(json.dumps(header)), kps=kps, arc=arc, res=res)
        os.replace(tmp, self._path)
        self._dirty = False
