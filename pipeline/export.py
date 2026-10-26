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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .cluster import PersonCluster, FrameRecord
from .score import pick_best


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

    # Убрать результаты прошлого запуска: иначе в best/ останутся устаревшие personNNNN_*
    for old in Path(best_dir).glob('person[0-9]*_*'):
        if old.is_file():
            try:
                old.unlink()
            except OSError as e:
                print(f"  [WARN] Не удалось удалить устаревший {old.name}: {e}")

    for cluster in clusters:
        scored = scored_map.get(cluster.person_id, [])
        best_result = pick_best(scored)

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
    close_call_margin: float = 0.02,
    close_call_max_pool: int = 6,
) -> dict:
    """Построить структуру отчёта."""
    persons = []
    for cluster in clusters:
        scored = scored_map.get(cluster.person_id, [])

        best_file = None
        best_score = 0.0
        all_passed = [r for r in scored if r['passed_gates']]
        best_result = pick_best(scored)
        manual = bool(best_result and best_result.get('manual'))
        if best_result:
            best_file  = Path(best_result['frame'].path).name
            best_score = best_result['score']

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
                'pitch_deg':       round(f.pitch_deg, 1) if f.pitch_deg is not None else None,
                'det_conf':        round(f.det_conf, 3),
                'gender':          _gender_str(f.gender),
                'age':             round(f.age, 1) if f.age is not None else None,
            })

        # Пометить selected правильно
        for fi in frames_info:
            fi['selected'] = (fi['file'] == best_file)

        all_low_quality = len(all_passed) == 0 and len(scored) > 0 and not manual

        # Два лучших кадра почти равны по score при небольшом выборе — выбор неуверенный,
        # стоит глянуть руками. Если хороших кадров много, любой из верхних подойдёт.
        pool = all_passed if all_passed else scored
        close_call = (not manual and 2 <= len(pool) <= close_call_max_pool
                      and (pool[0]['score'] - pool[1]['score']) < close_call_margin)

        persons.append({
            'person_id':       cluster.person_id,
            'frame_count':     len(cluster.frames),
            'single_frame':    len(cluster.frames) == 1,
            'best_photo':      best_file,
            'best_score':      round(best_score, 4),
            'manual_choice':   manual,
            'low_quality_flag': all_low_quality,
            'close_call':      close_call,
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
            'close_call_persons': sum(1 for p in persons if p['close_call']),
            'single_frame_persons': sum(1 for p in persons if p['single_frame']),
            'manual_choices': sum(1 for p in persons if p['manual_choice']),
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
  .person-header .doubt { color:#fb4; font-size:11px; margin-left:8px; }
  .frames { display:flex; flex-wrap:wrap; gap:8px; padding:12px; }
  .frame { position:relative; border-radius:4px; overflow:hidden; cursor:pointer; }
  .frame img { display:block; border-radius:4px; object-fit:cover; }
  .frame.best { outline:3px solid #4f4; }
  .frame.chosen { outline:3px solid #4af; }
  .frame.chosen.best { outline:3px solid #4af; box-shadow:0 0 0 3px #4f4; }
  .frame.rejected { opacity:0.45; outline:1px solid #f44; }
  .frame .open { position:absolute; top:2px; right:2px; background:rgba(0,0,0,0.6);
                 color:#fff; text-decoration:none; font-size:12px; padding:1px 5px;
                 border-radius:3px; }
  .frame .badge { position:absolute; bottom:0; left:0; right:0;
                  background:rgba(0,0,0,0.72); font-size:10px; padding:2px 4px;
                  text-align:center; }
  .frame.best .badge { color:#4f4; }
  .frame.chosen .badge { color:#4af; }
  .frame.rejected .badge { color:#f88; }
  .score { color:#fda; }
  .toolbar { position:sticky; top:0; z-index:10; background:#111; padding:8px 12px;
             border-bottom:1px solid #444; margin:-16px -16px 16px; font-size:13px; }
  .toolbar button, .toolbar label { margin-right:14px; }
  body.only-doubt .person[data-doubt="0"] { display:none; }
</style>
</head>
<body>
<div class="toolbar">
  <label><input type="checkbox" id="onlyDoubt"> только сомнительные</label>
  <span>Выбрано вручную: <b id="cnt">0</b></span>
  <button id="save">Скачать manual_choices.json</button>
  <button id="reset">Сбросить выбор</button>
  <span style="color:#888">клик по превью — выбрать лучший кадр вручную, ↗ — открыть оригинал</span>
</div>
"""

_HTML_TAIL = """<script>
const KEY = 'bestface:' + location.pathname;
let choices = JSON.parse(localStorage.getItem(KEY) || '{}');
function apply() {
  document.querySelectorAll('.frame').forEach(f => {
    const p = f.closest('.person').dataset.person;
    f.classList.toggle('chosen', choices[p] === f.dataset.file);
  });
  document.getElementById('cnt').textContent = Object.keys(choices).length;
}
document.addEventListener('click', e => {
  const f = e.target.closest('.frame');
  if (!f || e.target.closest('a')) return;
  const p = f.closest('.person').dataset.person;
  if (choices[p] === f.dataset.file) delete choices[p]; else choices[p] = f.dataset.file;
  localStorage.setItem(KEY, JSON.stringify(choices));
  apply();
});
document.getElementById('onlyDoubt').addEventListener('change', e =>
  document.body.classList.toggle('only-doubt', e.target.checked));
document.getElementById('reset').addEventListener('click', () => {
  choices = {}; localStorage.removeItem(KEY); apply();
});
document.getElementById('save').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify({choices}, null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'manual_choices.json'; a.click();
});
apply();
</script>
</body></html>
"""


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
    Превью (кропы лиц) сохраняются файлами в <имя_листа>_thumbs/ рядом с HTML.
    Клик по превью выбирает кадр вручную (сохраняется в браузере; выбор можно
    скачать как manual_choices.json для tune_weights.py), ↗ открывает оригинал.
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    thumbs_dir = Path(output_file).with_suffix('')
    thumbs_dir = thumbs_dir.parent / (thumbs_dir.name + '_thumbs')
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    # Превью считаются параллельно: декодирование крупных JPEG — самая медленная часть
    jobs = []
    for cluster in clusters:
        for r in scored_map.get(cluster.person_id, []):
            f_rec = r['frame']
            name = f'p{cluster.person_id:04d}_{f_rec.file_idx:05d}.jpg'
            jobs.append((f_rec.path, f_rec.bbox, str(thumbs_dir / name)))
    with ThreadPoolExecutor(max_workers=8) as pool:
        thumb_ok = dict(zip(
            (j[2] for j in jobs),
            pool.map(lambda j: _write_thumb(j[0], j[1], j[2], thumb_size), jobs),
        ))

    parts = [_HTML_HEAD]
    s = report.get('summary', {})
    unit_label = 'Серия' if report.get('params', {}).get('cluster_mode') == 'series' else 'Персона'
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
        doubt_reasons = []
        if low_q:
            doubt_reasons.append('⚠ низкое качество')
        if person_data and person_data.get('low_cohesion'):
            doubt_reasons.append('⚠ возможна склейка разных людей')
        if person_data and person_data.get('close_call'):
            doubt_reasons.append('⚖ два кадра почти равны')

        badges = ''.join(f'<span class="doubt">{r}</span>' for r in doubt_reasons)
        parts.append(
            f'<div class="person" data-person="{cluster.person_id}" '
            f'data-doubt="{1 if doubt_reasons else 0}">'
            f'<div class="person-header">'
            f'<span class="pid">{unit_label} #{cluster.person_id}</span>'
            f'<span class="cnt">{len(cluster.frames)} кадров</span>'
            f'{badges}'
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
            has_thumb = thumb_ok.get(str(thumbs_dir / thumb_name), False)
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
                f'<div class="{css_cls}" title="{html_mod.escape(fname)}" '
                f'data-file="{html_mod.escape(fname)}">'
                f'{img_tag}'
                f'<a class="open" href="{href}" target="_blank" title="Открыть оригинал">↗</a>'
                f'<div class="badge">{label}</div>'
                f'</div>'
            )

        parts.append('</div></div>\n')

    parts.append(_HTML_TAIL)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(''.join(parts))
