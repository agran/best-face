"""
pipeline/exif.py — время съёмки кадра из EXIF (DateTimeOriginal + SubSecTimeOriginal).
"""

import io
from datetime import datetime
from typing import Optional

from PIL import Image

_EXIF_IFD          = 0x8769
_DATETIME_ORIGINAL = 0x9003
_SUBSEC_ORIGINAL   = 0x9291
_DATETIME          = 0x0132


def read_shot_time(data: bytes) -> Optional[float]:
    """Вернуть время съёмки (секунды, unix-timestamp локального времени) или None."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            exif = im.getexif()
            ifd = exif.get_ifd(_EXIF_IFD)
            raw = ifd.get(_DATETIME_ORIGINAL) or exif.get(_DATETIME)
            if not raw:
                return None
            ts = datetime.strptime(str(raw).strip(), '%Y:%m:%d %H:%M:%S').timestamp()
            sub = str(ifd.get(_SUBSEC_ORIGINAL, '')).strip()
            if sub.isdigit():
                ts += int(sub) / (10 ** len(sub))
            return ts
    except Exception:
        return None
