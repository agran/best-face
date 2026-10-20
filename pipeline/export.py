"""
pipeline/export.py — сохранение результатов.

Выходные данные:
  output/best/          — лучшие фото (копии оригиналов)
  output/rejected/      — кадры без лица / с ошибками
  output/report.json    — полный отчёт по каждому кадру
  output/contact_sheet.html — HTML-галерея серий (превью)
"""

import os
import json
import shutil
import html as html_mod
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .cluster import PersonCluster, FrameRecord


# ─── Копирование файлов ────────────────────────────────────────────────────────

def copy_best_photos(
    clusters: List[PersonCluster],
    scored_map: dict,   # person_id → List[score_dict]
    best_dir: str,
    rejected_dir: str,
    rejected_frames: List[FrameRecord],
) -> None:
    """Скопировать лучшие фото в best_dir, бракованные в rejected_dir."""
    Path(best_dir).mkdir(parents=True, exist_ok=True)
    Path(rejected_dir).mkdir(parents=True, exist_ok=True)

    for cluster in clusters:
        scored = scored_map.get(cluster.person_id, [])
        if not scored:
            continue

        # Лучший кадр — первый после сортировки в score_cluster
        best_result = None
        for r in scored:
            if r['passed_gates']:
                best_result = r
                break
        if best_result is None and scored:
            best_result = scored[0]  # все плохие

        if best_result:
            src = best_result['frame'].path
            fname = f"person{cluster.person_id:04d}_{Path(src).name}"
            dst = os.path.join(best_dir, fname)
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"  [WARN] Не удалось скопировать {src}: {e}")

    for f in rejected_frames:
        src = f.path
        fname = Path(src).name
        dst = os.path.join(rejected_dir, fname)
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"  [WARN] Не удалось скопировать в rejected {src}: {e}")


# ─── JSON-отчёт ────────────────────────────────────────────────────────────────

def build_report(
    clusters: List[PersonCluster],
    scored_map: dict,
    rejected_frames: List[FrameRecord],
    params: dict,
) -> dict:
    """Построить структуру отчёта."""
    persons = []
    for cluster in clusters:
        scored = scored_map.get(cluster.person_id, [])

        best_file = None
        best_score = 0.0
        all_passed = [r for r in scored if r['passed_gates']]
        if all_passed:
            best_file  = Path(all_passed[0]['frame'].path).name
            best_score = all_passed[0]['score']
        elif scored:
            best_file  = Path(scored[0]['frame'].path).name
            best_score = scored[0]['score']

        frames_info = []
        for r in scored:
            f = r['frame']
            frames_info.append({
                'file':            Path(f.path).name,
                'score':           round(r['score'], 4),
                'metrics':         {k: round(v, 4) for k, v in r['metrics_norm'].items()},
                'selected':        (f.path == (Path(cluster.frames[0].path).parent /
                                               best_file).as_posix()
                                    if best_file else False),
                'passed_gates':    r['passed_gates'],
                'rejected_reason': r['rejected_reason'],
                'yaw_deg':         round(f.yaw_deg, 1),
                'frontal_ratio':   round(f.frontal_ratio, 3),
                'det_conf':        round(f.det_conf, 3),
                'gender':          _gender_str(f.gender),
                'age':             round(f.age, 1) if f.age is not None else None,
            })

        # Пометить selected правильно
        for fi in frames_info:
            fi['selected'] = (fi['file'] == best_file)

        all_low_quality = len(all_passed) == 0 and len(scored) > 0

        persons.append({
            'person_id':       cluster.person_id,
            'frame_count':     len(cluster.frames),
            'best_photo':      best_file,
            'best_score':      round(best_score, 4),
            'low_quality_flag': all_low_quality,
            'mean_similarity': round(cluster.mean_sim, 4) if cluster.mean_sim is not None else None,
            'low_cohesion':    cluster.low_cohesion,
            'frames':          frames_info,
        })

    rejected_info = [{
        'file':   Path(f.path).name,
        'reason': f.rejected_reason or 'no_face',
    } for f in rejected_frames]

    return {
        'summary': {
            'total_photos':    sum(len(c.frames) for c in clusters) + len(rejected_frames),
            'total_persons':   len(clusters),
            'total_rejected':  len(rejected_frames),
            'low_quality_persons': sum(1 for p in persons if p['low_quality_flag']),
            'low_cohesion_persons': sum(1 for p in persons if p['low_cohesion']),
        },
        'params':   params,
        'persons':  persons,
        'rejected': rejected_info,
    }


def save_report(report: dict, report_file: str) -> None:
    Path(report_file).parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


# ─── HTML-контактный лист ──────────────────────────────────────────────────────

def _write_thumb(path: str, bbox, out_path: str, size: int = 240) -> bool:
    """
    Сохранить превью лица (кроп вокруг bbox с запасом, не весь кадр) в JPEG.
    Поддерживает кириллицу в путях на Windows.
    """
    try:
        with open(path, 'rb') as fh:
            buf = np.frombuffer(fh.read(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return False
        h, w = img.shape[:2]
        if bbox is not None:
            x, y, bw, bh = bbox
            side = max(bw, bh) * 1.9
            cx, cy = x + bw / 2.0, y + bh / 2.0
            x1 = int(max(0, cx - side / 2)); x2 = int(min(w, cx + side / 2))
            y1 = int(max(0, cy - side / 2)); y2 = int(min(h, cy + side / 2))
            if x2 - x1 > 8 and y2 - y1 > 8:
                img = img[y1:y2, x1:x2]
        h, w = img.shape[:2]
        scale = size / max(h, w)
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=interp)
        ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return False
        with open(out_path, 'wb') as fh:
            fh.write(enc.tobytes())
        return True
    except Exception:
        return False


def _gender_str(g: Optional[float]) -> Optional[str]:
    if g is None:
        return None
    return 'f' if g > 0 else 'm'


_HTML_HEAD = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Best-Face — контактный лист</title>
<style>
  body { font-family: sans-serif; background:#1a1a1a; color:#ddd; margin:0; padding:16px; }
  h1 { color:#eee; }
  .summary { background:#2a2a2a; padding:12px; border-radius:6px; margin-bottom:20px; }
  .person { border:1px solid #444; border-radius:8px; margin-bottom:24px;
            background:#222; overflow:hidden; }
  .person-header { background:#333; padding:10px 16px; font-weight:bold; font-size:14px; }
  .person-header .pid { color:#8cf; }
  .person-header .cnt { color:#aaa; font-weight:normal; font-size:12px; margin-left:8px; }
  .person-header .low { color:#f88; font-size:11px; margin-left:8px; }
  .frames { display:flex; flex-wrap:wrap; gap:8px; padding:12px; }
  .frame { position:relative; border-radius:4px; overflow:hidden; cursor:default; }
  .frame img { display:block; border-radius:4px; object-fit:cover; }
  .frame.best { outline:3px solid #4f4; }
  .frame.rejected { opacity:0.45; outline:1px solid #f44; }
  .frame .badge { position:absolute; bottom:0; left:0; right:0;
                  background:rgba(0,0,0,0.72); font-size:10px; padding:2px 4px;
                  text-align:center; }
  .frame.best .badge { color:#4f4; }
  .frame.rejected .badge { color:#f88; }
  .score { color:#fda; }
</style>
</head>
<body>
"""

_HTML_TAIL = "</body></html>\n"


def build_contact_sheet(
    clusters: List[PersonCluster],
    scored_map: dict,
    rejected_frames: List[FrameRecord],
    report: dict,
    output_file: str,
    thumb_size: int = 240,
) -> None:
    """
    Сгенерировать HTML-контактный лист с превью всех серий.
    Превью (кропы лиц) сохраняются файлами в <имя_листа>_thumbs/ рядом с HTML;
    клик по превью открывает исходное фото.
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    thumbs_dir = Path(output_file).with_suffix('')
    thumbs_dir = thumbs_dir.parent / (thumbs_dir.name + '_thumbs')
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    parts = [_HTML_HEAD]
    s = report.get('summary', {})
    parts.append(
        f'<h1>Best-Face — результаты</h1>'
        f'<div class="summary">'
        f'Всего фото: <b>{s.get("total_photos","?")}</b> | '
        f'Персон: <b>{s.get("total_persons","?")}</b> | '
        f'Отклонено (без лица): <b>{s.get("total_rejected","?")}</b>'
        f'</div>\n'
    )

    for cluster in clusters:
        scored = scored_map.get(cluster.person_id, [])
        person_data = next(
            (p for p in report['persons'] if p['person_id'] == cluster.person_id),
            None
        )
        best_file = person_data['best_photo'] if person_data else None
        low_q = person_data and person_data['low_quality_flag']

        low_badge = '<span class="low">⚠ низкое качество</span>' if low_q else ''
        parts.append(
            f'<div class="person">'
            f'<div class="person-header">'
            f'<span class="pid">Персона #{cluster.person_id}</span>'
            f'<span class="cnt">{len(cluster.frames)} кадров</span>'
            f'{low_badge}'
            f'</div>'
            f'<div class="frames">'
        )

        for r in scored:
            f_rec = r['frame']
            fname = Path(f_rec.path).name
            is_best = (fname == best_file)
            is_rej  = not r['passed_gates']

            css_cls = 'best' if is_best else ('rejected' if is_rej else 'frame')
            css_cls = 'frame ' + css_cls

            thumb_name = f'p{cluster.person_id:04d}_{f_rec.file_idx:05d}.jpg'
            has_thumb = _write_thumb(f_rec.path, f_rec.bbox,
                                     str(thumbs_dir / thumb_name), thumb_size)
            if has_thumb:
                rel = f'{thumbs_dir.name}/{thumb_name}'
                img_tag = (f'<img src="{rel}" width="{thumb_size}" '
                           f'height="{thumb_size}" loading="lazy">')
            else:
                img_tag = (f'<div style="width:{thumb_size}px;height:{thumb_size}px;'
                           f'background:#333;"></div>')
            href = Path(f_rec.path).resolve().as_uri()

            score_str = f'{r["score"]:.2f}'
            reason    = html_mod.escape(r['rejected_reason'] or '')
            label     = ('★ BEST' if is_best else
                         (reason if reason else f'<span class="score">{score_str}</span>'))

            parts.append(
                f'<div class="{css_cls}" title="{html_mod.escape(fname)}">'
                f'<a href="{href}" target="_blank">{img_tag}</a>'
                f'<div class="badge">{label}</div>'
                f'</div>'
            )

        parts.append('</div></div>\n')

    parts.append(_HTML_TAIL)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(''.join(parts))
