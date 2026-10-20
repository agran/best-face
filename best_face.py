#!/usr/bin/env python3
"""
best_face.py — главный скрипт отбора лучших фото из серий.

Использование:
  python best_face.py <папка_с_фото> [--config config.yaml] [--output output/]
                     [--limit N] [--no-contact-sheet]

Примеры:
  python best_face.py "D:/photos/session1"
  python best_face.py photos/ --output results/ --limit 100
"""

import argparse
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from pipeline.detect  import FaceDetector
from pipeline.align   import align_face, estimate_yaw_deg, estimate_frontal_ratio
from pipeline.embed   import ArcFaceEmbedder, FaceResEmbedder, NimaScorer, estimate_age_from_probs
from pipeline.ear     import EarEstimator
from pipeline.metrics import compute_all_metrics
from pipeline.cluster import FrameRecord, cluster_frames
from pipeline.cache   import FrameCache, make_signature, file_fingerprint
from pipeline.exif    import read_shot_time
from pipeline.score   import score_cluster, pick_best
from pipeline.export  import (
    copy_best_photos, build_report, save_report, build_contact_sheet
)

_SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
_MULTI_FACE_POLICIES = ('largest', 'central', 'best_conf')


# ─── Загрузка конфига ──────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_model_path(path, label: str, required: bool = True):
    """
    Проверить путь к модели. Если файла нет (например, недоступен сетевой диск),
    искать одноимённый файл в локальной папке models/.
    """
    if not path:
        return path
    if os.path.exists(path):
        return path
    local = Path(__file__).parent / 'models' / Path(str(path).replace('\\', '/')).name
    if local.exists():
        print(f'  [WARN] {label}: {path} недоступен, использую {local}')
        return str(local)
    if required:
        print(f'ОШИБКА: модель {label} не найдена: {path}\n'
              f'  Скопируйте её в {local.parent} или поправьте путь в config.yaml.')
        sys.exit(2)
    print(f'  [WARN] {label}: {path} не найден — компонент отключён')
    return None


# ─── Выбор лица при нескольких в кадре ─────────────────────────────────────────

def pick_face(faces: list, img_shape, policy: str = 'largest'):
    """Выбрать «главное» лицо. faces отсортированы по убыванию площади."""
    if len(faces) == 1 or policy == 'largest':
        return faces[0]
    if policy == 'best_conf':
        return max(faces, key=lambda f: f.conf)
    # central: среди достаточно крупных лиц — ближайшее к центру кадра
    h, w = img_shape[:2]
    max_area = faces[0].bbox[2] * faces[0].bbox[3]
    candidates = [f for f in faces if f.bbox[2] * f.bbox[3] >= 0.25 * max_area]

    def off_center(f):
        x, y, bw, bh = f.bbox
        return ((x + bw / 2 - w / 2) / w) ** 2 + ((y + bh / 2 - h / 2) / h) ** 2

    return min(candidates, key=off_center)


# ─── Сканирование папки ────────────────────────────────────────────────────────

def scan_photos(folder: str) -> list:
    """Найти все фото в папке, отсортировать по имени файла."""
    paths = []
    for entry in sorted(Path(folder).iterdir()):
        if entry.is_file() and entry.suffix.lower() in _SUPPORTED_EXT:
            paths.append(str(entry))
    return paths


# ─── Обработка одного файла ────────────────────────────────────────────────────

def load_image(path: str):
    """Прочитать и декодировать файл. Возвращает (img_bgr, shot_time) или (None, None)."""
    try:
        with open(path, 'rb') as fh:
            raw = fh.read()
        # cv2.imread не поддерживает кириллицу в путях на Windows — декодируем из буфера
        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None, None
    if img is None:
        return None, None
    return img, read_shot_time(raw)


def process_one(
    path: str,
    file_idx: int,
    detector: FaceDetector,
    embedder_arc: ArcFaceEmbedder,
    embedder_res: Optional[FaceResEmbedder],
    embedder_nima: NimaScorer,
    ear_estimator: EarEstimator,
    cfg_det: dict,
    multi_face_policy: str = 'largest',
    loaded: tuple = None,
) -> FrameRecord:
    """
    Обработать один файл: детекция → alignment → embedding → метрики.
    Возвращает FrameRecord (rejected_reason != None если нет лица).
    loaded — уже декодированный (img, shot_time) из load_image (предзагрузка в потоках).
    """
    rec = FrameRecord(path, file_idx)

    img, shot_time = loaded if loaded is not None else load_image(path)
    if img is None:
        rec.rejected_reason = 'read_error'
        return rec

    rec.shot_time = shot_time

    # Детекция
    faces = detector.detect(img, min_face_px=cfg_det.get('min_face_px', 50))
    if not faces:
        rec.rejected_reason = 'no_face'
        return rec

    face = pick_face(faces, img.shape, multi_face_policy)

    rec.bbox     = face.bbox
    rec.kps      = face.kps
    rec.det_conf = face.conf

    # Геометрия позы (по 5 точкам; ниже уточняется MediaPipe, если доступен)
    rec.yaw_deg      = estimate_yaw_deg(face.kps)
    rec.frontal_ratio = estimate_frontal_ratio(face.kps)

    # Выравнивание
    crop_112 = align_face(img, face.kps, output_size=112)
    crop_224 = align_face(img, face.kps, output_size=224)

    if crop_112 is None or crop_224 is None:
        rec.rejected_reason = 'alignment_failed'
        return rec

    # ArcFace эмбеддинг (512-dim, для группировки)
    rec.emb_arc = embedder_arc(crop_112)

    # FaceRes эмбеддинг (1024-dim, целевая метрика); без модели Q8 считается по ArcFace
    if embedder_res is not None:
        emb_res, gender, age_probs = embedder_res(crop_224)
        rec.emb_res = emb_res
        rec.gender  = gender
        if age_probs is not None:
            rec.age = estimate_age_from_probs(age_probs)
    else:
        rec.emb_res = rec.emb_arc

    # MediaPipe: EAR, blendshapes (глаза/рот) и поза головы
    pose_info = ear_estimator.analyze(img, face.bbox) if ear_estimator.available else None
    ear_val = pose_info['ear'] if pose_info else None
    if pose_info and pose_info['yaw'] is not None:
        rec.yaw_deg   = pose_info['yaw']
        rec.pitch_deg = pose_info['pitch']
        rec.roll_deg  = pose_info['roll']

    # Метрики качества Q1-Q7
    rec.metrics = compute_all_metrics(
        img_bgr=img,
        face_crop_112=crop_112,
        kps_orig=face.kps,
        bbox=face.bbox,
        det_conf=face.conf,
        frontal_ratio=rec.frontal_ratio,
        ear_value=ear_val,
        face_crop_224=crop_224,
        pose_info=pose_info,
    )

    # Q9: NIMA aesthetic score если модель загружена, иначе q9_face_illumination из compute_all_metrics
    if embedder_nima.available:
        rec.metrics['Q9_nima'] = embedder_nima(crop_224)

    return rec


# ─── Прогресс-бар ─────────────────────────────────────────────────────────────

class _SimpleProgress:
    def __init__(self, total, desc=''):
        self.total = total
        self.n = 0
        self.desc = desc
        self._t0 = time.time()
        print(f'{desc}: 0/{total}', end='', flush=True)

    def update(self, n=1):
        self.n += n
        elapsed = time.time() - self._t0
        rate = self.n / max(elapsed, 0.001)
        eta  = (self.total - self.n) / rate if rate > 0 else 0
        print(f'\r{self.desc}: {self.n}/{self.total}  '
              f'{rate:.1f} img/s  ETA {eta:.0f}s    ', end='', flush=True)

    def close(self):
        elapsed = time.time() - self._t0
        print(f'\r{self.desc}: {self.total}/{self.total}  '
              f'за {elapsed:.1f}s          ')


# ─── Главная функция ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Отбор лучших фото из серий')
    parser.add_argument('input_dir',              help='Папка с фотографиями')
    parser.add_argument('--config',  default=str(Path(__file__).parent / 'config.yaml'),
                        help='Путь к config.yaml')
    parser.add_argument('--output',  default=None,
                        help='Папка для результатов (переопределяет output в конфиге)')
    parser.add_argument('--limit',   type=int, default=0,
                        help='Обработать только первые N фото (0 = все)')
    parser.add_argument('--no-contact-sheet', action='store_true',
                        help='Не генерировать HTML-контактный лист')
    parser.add_argument('--no-cache', action='store_true',
                        help='Не использовать кэш обработанных кадров (frame_cache.pkl)')
    args = parser.parse_args()

    # ── Конфиг ──────────────────────────────────────────────────────────────
    cfg = load_config(args.config)

    out_cfg = cfg.get('output', {})
    if args.output:
        base = args.output
    else:
        # По умолчанию: подпапка внутри папки с фото
        base = str(Path(args.input_dir) / 'best_face_output')

    best_dir     = os.path.join(base, 'best')
    rejected_dir = os.path.join(base, 'rejected')
    report_file  = os.path.join(base, 'report.json')
    contact_file = os.path.join(base, 'contact_sheet.html')

    if args.no_contact_sheet:
        contact_file = None

    cfg_det     = cfg.get('detection',  {})
    cfg_cluster = cfg.get('clustering', {})
    cfg_scoring = cfg.get('scoring',    {})
    cfg_models  = cfg.get('models',     {})

    # ── Сканирование ────────────────────────────────────────────────────────
    print(f'Сканируем: {args.input_dir}')
    all_photos = scan_photos(args.input_dir)
    if args.limit > 0:
        all_photos = all_photos[:args.limit]
    print(f'Найдено файлов: {len(all_photos)}')

    if not all_photos:
        print('Нет фото для обработки.')
        sys.exit(0)

    # ── Загрузка моделей ────────────────────────────────────────────────────
    print('Загружаем модели...')
    t0 = time.time()

    policy = cfg_det.get('multi_face_policy', 'largest')
    if policy not in _MULTI_FACE_POLICIES:
        print(f'ОШИБКА: multi_face_policy={policy!r}, допустимо: {_MULTI_FACE_POLICIES}')
        sys.exit(2)

    det_path      = resolve_model_path(cfg_models['detector'], 'detector')
    arc_path      = resolve_model_path(cfg_models['arcface'], 'arcface')
    faceres_path  = resolve_model_path(cfg_models.get('faceres'), 'faceres', required=False)
    nima_path     = resolve_model_path(cfg_models.get('nima'), 'nima', required=False)
    ear_path      = resolve_model_path(
        cfg_models.get('ear', 'models/face_landmarker.task'), 'ear', required=False)

    detector     = FaceDetector(
        model_path      = det_path,
        conf_threshold  = cfg_det.get('conf_threshold', 0.5),
        nms_threshold   = cfg_det.get('nms_threshold',  0.4),
        max_det_size    = cfg_det.get('max_det_size',   1280),
    )
    embedder_arc  = ArcFaceEmbedder(arc_path)
    embedder_res  = FaceResEmbedder(faceres_path) if faceres_path else None
    if embedder_res is None:
        print('  FaceResEmbedder:  отключён — Q8 считается по ArcFace')
    embedder_nima = NimaScorer(nima_path)
    ear_estimator = EarEstimator(ear_path)

    print(f'Модели загружены за {time.time()-t0:.1f}s')

    # ── Обработка кадров ────────────────────────────────────────────────────
    print(f'\nШаг 1/3: Детекция + эмбеддинги ({len(all_photos)} фото)...')

    all_records: list[FrameRecord] = []
    rejected_records: list[FrameRecord] = []

    cache_sig = make_signature(
        models=[file_fingerprint(p) for p in (det_path, arc_path, faceres_path, nima_path, ear_path)],
        detection=cfg_det, policy=policy,
    )
    cache = FrameCache(os.path.join(base, 'frame_cache.pkl'), cache_sig,
                       enabled=not args.no_cache)

    results: dict = {}
    for idx, path in enumerate(all_photos):
        cached = cache.get(path)
        if cached is not None:
            cached.file_idx = idx
            results[idx] = cached
    todo = [i for i in range(len(all_photos)) if i not in results]
    if results:
        print(f'  Из кэша: {len(results)} кадров, обработать: {len(todo)}')

    if _HAS_TQDM:
        progress = tqdm(total=len(all_photos), desc='Обработка', unit='фото')
    else:
        progress = _SimpleProgress(len(all_photos), 'Обработка')
    progress.update(len(results))

    # Чтение и декодирование JPEG идёт в потоках на несколько кадров вперёд,
    # пока основной поток считает эмбеддинги предыдущего кадра
    workers = max(1, int(cfg.get('performance', {}).get('workers', 4)))
    todo_iter = iter(todo)
    window = deque()

    def submit_next(pool):
        i = next(todo_iter, None)
        if i is not None:
            window.append((i, pool.submit(load_image, all_photos[i])))

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in range(workers * 2):
                submit_next(pool)
            done = 0
            while window:
                idx, fut = window.popleft()
                submit_next(pool)
                rec = process_one(
                    path=all_photos[idx],
                    file_idx=idx,
                    detector=detector,
                    embedder_arc=embedder_arc,
                    embedder_res=embedder_res,
                    embedder_nima=embedder_nima,
                    ear_estimator=ear_estimator,
                    cfg_det=cfg_det,
                    multi_face_policy=policy,
                    loaded=fut.result(),
                )
                results[idx] = rec
                cache.put(all_photos[idx], rec)
                progress.update(1)
                done += 1
                if done % 200 == 0:
                    cache.save()
    finally:
        cache.save()

    progress.close()

    for idx in range(len(all_photos)):
        rec = results[idx]
        if rec.rejected_reason is not None:
            rejected_records.append(rec)
        else:
            all_records.append(rec)

    n_valid    = len(all_records)
    n_rejected = len(rejected_records)
    print(f'  Валидных кадров с лицами: {n_valid}')
    print(f'  Отклонено (нет лица / ошибка): {n_rejected}')

    if n_valid == 0:
        print('Лица не найдены. Проверьте входные данные.')
        sys.exit(1)

    # ── Кластеризация ───────────────────────────────────────────────────────
    print(f'\nШаг 2/3: Кластеризация серий...')
    t1 = time.time()
    cluster_mode = cfg_cluster.get('mode', 'person')
    if cluster_mode not in ('person', 'series'):
        print(f"ОШИБКА: clustering.mode={cluster_mode!r}, допустимо: 'person' | 'series'")
        sys.exit(2)

    clusters = cluster_frames(
        frames           = all_records,
        threshold        = cfg_cluster.get('threshold',           0.50),
        time_window      = cfg_cluster.get('time_window_frames',  8),
        threshold_time   = cfg_cluster.get('threshold_time_hint', 0.40),
        time_window_sec  = cfg_cluster.get('time_window_sec',     8.0),
        time_far_sec     = cfg_cluster.get('time_far_sec',        120.0),
        time_far_penalty = cfg_cluster.get('time_far_penalty',    0.10),
        min_cohesion     = cfg_cluster.get('min_cohesion',        0.0),
        split_step       = cfg_cluster.get('split_step',          0.03),
        max_split_rounds = cfg_cluster.get('max_split_rounds',    6),
        mode             = cluster_mode,
        series_gap_sec   = cfg_cluster.get('series_gap_sec',      14.0),
    )

    n_with_time = sum(1 for r in all_records if r.shot_time is not None)
    print(f'  Кадров с EXIF-временем: {n_with_time}/{len(all_records)}')
    if cluster_mode == 'series' and n_with_time < 0.9 * len(all_records):
        print('  [WARN] у большинства кадров нет EXIF-времени — разрез на серии по паузам '
              'работает частично; для режима «один кластер на человека» задайте '
              'clustering.mode: person')
    print(f'  Найдено персон: {len(clusters)}  '
          f'(за {time.time()-t1:.1f}s)')

    # ── Скоринг ─────────────────────────────────────────────────────────────
    print(f'\nШаг 3/3: Скоринг и выбор лучших...')

    weights    = cfg_scoring.get('weights', {})
    hard_gates = cfg_scoring.get('hard_gates', {})

    scored_by_cluster = [
        score_cluster(c, weights=weights or None, gates=hard_gates or None)
        for c in clusters
    ]

    # Нумерация персон — в порядке исходной папки: по номеру файла лучшего кадра,
    # чтобы порядок person0001, person0002... в best/ совпадал с порядком съёмки
    order = sorted(
        range(len(clusters)),
        key=lambda i: (pick_best(scored_by_cluster[i])['frame'].file_idx
                       if scored_by_cluster[i] else clusters[i].frames[0].file_idx),
    )
    clusters = [clusters[i] for i in order]
    scored_map = {}
    for new_id, i in enumerate(order, start=1):
        scored_map[new_id] = scored_by_cluster[i]
        clusters[new_id - 1].person_id = new_id

    print(f'  Распределение серий (топ 10 по числу кадров):')
    for c in sorted(clusters, key=lambda c: len(c.frames), reverse=True)[:10]:
        sim = f'{c.mean_sim:.2f}' if c.mean_sim is not None else '—'
        mark = '  [low_cohesion]' if c.low_cohesion else ''
        print(f'    Персона #{c.person_id}: {len(c.frames)} кадров, ср. cosine {sim}{mark}')
    if len(clusters) > 10:
        print(f'    ... и ещё {len(clusters)-10} персон')

    # ── Экспорт ─────────────────────────────────────────────────────────────
    print(f'\nЭкспорт результатов...')

    params = {
        'input_dir':         args.input_dir,
        'total_photos':      len(all_photos),
        'threshold_cluster': cfg_cluster.get('threshold', 0.50),
        'clustering':        cfg_cluster,
        'cluster_mode':      cluster_mode,
        'multi_face_policy': policy,
        'weights':           weights,
        'hard_gates':        hard_gates,
    }

    report = build_report(clusters, scored_map, rejected_records, params)
    save_report(report, report_file)
    print(f'  Отчёт: {report_file}')

    copy_best_photos(clusters, scored_map, best_dir, rejected_dir, rejected_records)
    print(f'  Лучшие фото: {best_dir}')
    print(f'  Отклонённые: {rejected_dir}')

    if contact_file:
        print(f'  Контактный лист (может занять время): {contact_file}')
        build_contact_sheet(clusters, scored_map, rejected_records, report, contact_file)

    # ── Итог ────────────────────────────────────────────────────────────────
    s = report['summary']
    print(f'\n{"="*50}')
    print(f'ГОТОВО')
    print(f'  Всего фото обработано:  {s["total_photos"]}')
    print(f'  Найдено персон:         {s["total_persons"]}')
    print(f'  Лучших фото сохранено:  {s["total_persons"]}')
    print(f'  Без лица / отклонено:   {s["total_rejected"]}')
    print(f'  Персон с низким кач-вом:{s["low_quality_persons"]}')
    if s['low_quality_persons'] > 0:
        print(f'  (рекомендуется ручная проверка помеченных серий)')
    if s.get('low_cohesion_persons'):
        print(f'  Серий с низкой связностью: {s["low_cohesion_persons"]} '
              f'(возможна склейка людей — проверьте)')
    if s.get('close_call_persons'):
        print(f'  Серий с почти равными кадрами: {s["close_call_persons"]} '
              f'(в контактном листе фильтр «только сомнительные»)')
    print(f'{"="*50}')
    print(f'  Результаты: {Path(best_dir).parent.resolve()}')


if __name__ == '__main__':
    main()
