"""
pipeline/cluster.py — группировка кадров в серии одного человека.

Алгоритм:
  1. Матрица cosine-сходства ArcFace (512-dim).
  2. Поправка сходства по порядку/времени съёмки:
       - кадры, снятые рядом (EXIF-время, а при его отсутствии — номер файла),
         получают бонус (threshold - threshold_time);
       - кадры, снятые далеко друг от друга по EXIF-времени, получают штраф.
  3. Агломеративная кластеризация с average-linkage (UPGMA) по порогу.
     В отличие от single-linkage (связные компоненты) одна «мостовая» пара
     похожих лиц не склеивает две разные серии.
  4. Кластеры с низкой внутренней связностью (средний cosine < min_cohesion)
     рекурсивно дробятся с более строгим порогом; если не удалось добиться
     нужной связности — помечаются флагом low_cohesion (возможна склейка людей).
     Размер кластера критерием не является: длинная серия одного человека — норма.
"""

import math
from typing import List, Optional

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


class FrameRecord:
    """Данные о кадре после детекции и embedding."""
    __slots__ = (
        'path', 'file_idx',          # путь и порядковый номер файла
        'shot_time',                 # EXIF-время съёмки (сек) или None
        'bbox', 'kps', 'det_conf',   # результаты детекции
        'emb_arc', 'emb_res',        # ArcFace 512-dim, FaceRes 1024-dim
        'gender', 'age',             # из Human FaceRes
        'metrics',                   # dict Q1-Q9 (Q8 = 0.0 пока)
        'yaw_deg', 'frontal_ratio',  # геометрия позы
        'pitch_deg', 'roll_deg',     # поза из MediaPipe (None, если недоступна)
        'rejected_reason',           # str или None
    )

    def __init__(self, path: str, file_idx: int):
        self.path           = path
        self.file_idx       = file_idx
        self.shot_time      = None
        self.bbox           = None
        self.kps            = None
        self.det_conf       = 0.0
        self.emb_arc        = None  # np.ndarray (512,)
        self.emb_res        = None  # np.ndarray (1024,)
        self.gender         = None
        self.age            = None
        self.metrics        = {}
        self.yaw_deg        = 0.0
        self.frontal_ratio  = 1.0
        self.pitch_deg      = None
        self.roll_deg       = None
        self.rejected_reason = None


def _robust_centroid(embs: List[np.ndarray], rounds: int = 3, z: float = 1.5) -> np.ndarray:
    """
    L2-нормированный центроид с итеративным отбрасыванием выбросов
    (отвёрнутые/смазанные кадры, чужие лица): центр → отбросить кадры с низким
    cosine к центру → пересчитать. Остаётся не менее половины кадров.
    """
    X = np.stack(embs)
    n = len(X)
    keep = np.ones(n, dtype=bool)
    min_keep = max(2, math.ceil(n / 2))

    for _ in range(rounds):
        if n < 4:
            break
        c = X[keep].mean(axis=0)
        c /= max(np.linalg.norm(c), 1e-9)
        sims = X @ c
        s = sims[keep]
        thr = s.mean() - max(z * s.std(), 0.03)
        new_keep = sims >= thr
        if new_keep.sum() < min_keep or np.array_equal(new_keep, keep):
            break
        keep = new_keep

    c = X[keep].mean(axis=0)
    norm = np.linalg.norm(c)
    return c / norm if norm > 1e-9 else c


class PersonCluster:
    """Кластер кадров одного человека."""
    __slots__ = ('person_id', 'frames', 'centroid_arc', 'centroid_res',
                 'mean_sim', 'low_cohesion')

    def __init__(self, person_id: int):
        self.person_id    = person_id
        self.frames: List[FrameRecord] = []
        self.centroid_arc = None  # np.ndarray (512,)
        self.centroid_res = None  # np.ndarray (1024,)
        self.mean_sim     = None  # средний ArcFace-cosine между кадрами внутри кластера
        self.low_cohesion = False # не удалось добиться min_cohesion — возможна склейка людей

    def compute_centroids(self):
        """Вычислить устойчивые L2-нормированные центроиды по валидным эмбеддингам."""
        arcs = [f.emb_arc for f in self.frames if f.emb_arc is not None]
        ress = [f.emb_res for f in self.frames if f.emb_res is not None]

        if arcs:
            self.centroid_arc = _robust_centroid(arcs)
        if ress:
            self.centroid_res = _robust_centroid(ress)

        if len(arcs) >= 2:
            X = np.stack(arcs)
            S = X @ X.T
            n = len(arcs)
            self.mean_sim = float((S.sum() - np.trace(S)) / (n * (n - 1)))


def _agglomerate(dist: np.ndarray, max_dist: float) -> np.ndarray:
    """Average-linkage по матрице расстояний. Возвращает метки 0..k-1."""
    n = dist.shape[0]
    if n == 1:
        return np.zeros(1, dtype=int)
    d = np.clip(dist, 0.0, None)
    np.fill_diagonal(d, 0.0)
    d = (d + d.T) / 2.0
    Z = linkage(squareform(d, checks=False), method='average')
    return fcluster(Z, t=max_dist, criterion='distance') - 1


def _proximity_masks(valid: List[FrameRecord], time_window: int, time_window_sec: float,
                     time_far_sec: float):
    """Матрицы «снято рядом» и «снято далеко» (по EXIF-времени, иначе по номеру файла)."""
    idx = np.array([f.file_idx for f in valid], dtype=np.int64)
    t = np.array([np.nan if f.shot_time is None else f.shot_time for f in valid],
                 dtype=np.float64)

    d_idx = np.abs(idx[:, None] - idx[None, :])
    d_t = np.abs(t[:, None] - t[None, :])
    has_t = ~np.isnan(d_t)

    near = np.where(has_t, d_t <= time_window_sec, d_idx <= time_window)
    far = has_t & (d_t > time_far_sec) if time_far_sec > 0 else np.zeros_like(has_t)
    np.fill_diagonal(near, False)
    np.fill_diagonal(far, False)
    return near, far


def _segment_by_time(valid: List[FrameRecord], gap_sec: float) -> List[List[int]]:
    """
    Разрезать кадры (в порядке файлов) на серии по паузам в съёмке: если между
    соседними кадрами с EXIF-временем пауза больше gap_sec — начинается новая серия.
    Кадры без EXIF-времени серию не разрывают.
    """
    order = sorted(range(len(valid)), key=lambda i: valid[i].file_idx)
    segments: List[List[int]] = [[order[0]]]
    for a, b in zip(order, order[1:]):
        ta, tb = valid[a].shot_time, valid[b].shot_time
        if ta is not None and tb is not None and abs(tb - ta) > gap_sec:
            segments.append([])
        segments[-1].append(b)
    return segments


def _cluster_by_identity(
    frames: List[FrameRecord],
    threshold: float,
    time_window: int,
    threshold_time: float,
    time_window_sec: float,
    time_far_sec: float,
    time_far_penalty: float,
    min_cohesion: float,
    split_step: float,
    max_split_rounds: int,
):
    """Кластеризация одного набора кадров по лицу. Возвращает список (кадры, low_cohesion)."""
    n = len(frames)
    if n == 1:
        return [([frames[0]], False)]

    embs = np.stack([f.emb_arc for f in frames])  # (n, 512)
    sim = embs @ embs.T

    near, far = _proximity_masks(frames, time_window, time_window_sec, time_far_sec)
    bonus = max(0.0, threshold - threshold_time)
    dist = 1.0 - (sim + bonus * near - time_far_penalty * far)

    groups: List[tuple] = []

    def cohesion(positions: List[int]) -> float:
        k = len(positions)
        if k < 2:
            return 1.0
        s = sim[np.ix_(positions, positions)]
        return float((s.sum() - np.trace(s)) / (k * (k - 1)))

    def split(positions: List[int], level: int):
        if min_cohesion <= 0 or cohesion(positions) >= min_cohesion:
            groups.append(([frames[p] for p in positions], False))
            return
        if level >= max_split_rounds:
            groups.append(([frames[p] for p in positions], True))
            return
        thr = threshold + split_step * (level + 1)
        sub = _agglomerate(dist[np.ix_(positions, positions)], 1.0 - thr)
        for lbl in sorted(set(sub.tolist())):
            split([positions[i] for i in np.where(sub == lbl)[0]], level + 1)

    labels = _agglomerate(dist, 1.0 - threshold)
    for lbl in sorted(set(labels.tolist())):
        split([int(i) for i in np.where(labels == lbl)[0]], 0)
    return groups


def cluster_frames(
    frames: List[FrameRecord],
    threshold:          float = 0.50,
    time_window:        int   = 8,
    threshold_time:     float = 0.40,
    time_window_sec:    float = 8.0,
    time_far_sec:       float = 120.0,
    time_far_penalty:   float = 0.10,
    min_cohesion:       float = 0.0,
    split_step:         float = 0.03,
    max_split_rounds:   int   = 6,
    mode:               str   = 'person',
    series_gap_sec:     float = 14.0,
    dense_limit:        int   = 6000,
) -> List[PersonCluster]:
    """
    Сгруппировать кадры в кластеры.

    frames            — список FrameRecord (только с валидными emb_arc).
    threshold         — средний cosine между кластерами для «один человек».
    time_window       — окно в номерах файлов (если нет EXIF-времени).
    threshold_time    — мягкий порог для кадров, снятых рядом (бонус = threshold - threshold_time).
    time_window_sec   — окно в секундах для «снято рядом» по EXIF.
    time_far_sec      — разрыв по EXIF-времени, после которого пара штрафуется (0 = выкл.).
    time_far_penalty  — штраф к cosine для далёких по времени пар.
    min_cohesion      — если > 0, кластеры со средним cosine внутри ниже этого значения дробятся строже.
    split_step        — шаг повышения порога при дроблении.
    max_split_rounds  — максимум шагов дробления.
    mode              — 'person': один кластер на человека во всей папке;
                        'series': сначала разрез на серии по паузам в съёмке
                        (series_gap_sec), затем кластеризация по лицу внутри серии.
    series_gap_sec    — пауза между соседними кадрами, начинающая новую серию.
    dense_limit       — в режиме 'person' при большем числе кадров работает двухэтапная
                        схема (матрица n×n иначе не помещается в память): кластеризация
                        внутри временных сегментов, затем слияние центроидов.

    Возвращает список PersonCluster, отсортированных по убыванию числа кадров.
    """
    valid = [f for f in frames if f.emb_arc is not None]
    if not valid:
        return []

    identity_args = (threshold, time_window, threshold_time, time_window_sec,
                     time_far_sec, time_far_penalty, min_cohesion, split_step,
                     max_split_rounds)

    two_stage = mode == 'person' and dense_limit > 0 and len(valid) > dense_limit
    if mode == 'series' or two_stage:
        segments = [[valid[i] for i in seg] for seg in _segment_by_time(valid, series_gap_sec)]
        if two_stage:
            # Слишком длинный сегмент (или файлы без EXIF-времени) режем подряд идущими кусками
            segments = [seg[k:k + dense_limit] for seg in segments
                        for k in range(0, len(seg), dense_limit)]
    else:
        segments = [valid]

    clusters = []
    for seg in segments:
        groups = _cluster_by_identity(seg, *identity_args)
        for group_frames, loose in groups:
            cluster = PersonCluster(0)
            cluster.frames = sorted(group_frames, key=lambda f: f.file_idx)
            cluster.low_cohesion = loose
            cluster.compute_centroids()
            clusters.append(cluster)

    if two_stage:
        clusters = _merge_clusters(clusters, threshold)

    clusters.sort(key=lambda c: len(c.frames), reverse=True)
    for new_id, cluster in enumerate(clusters, start=1):
        cluster.person_id = new_id

    return clusters


def _merge_clusters(clusters: List[PersonCluster], threshold: float) -> List[PersonCluster]:
    """Слить кластеры одного человека из разных сегментов (average-linkage по центроидам)."""
    if len(clusters) < 2:
        return clusters
    C = np.stack([c.centroid_arc for c in clusters])
    labels = _agglomerate(1.0 - C @ C.T, 1.0 - threshold)

    merged = []
    for lbl in sorted(set(labels.tolist())):
        members = [clusters[i] for i in np.where(labels == lbl)[0]]
        if len(members) == 1:
            merged.append(members[0])
            continue
        cluster = PersonCluster(0)
        cluster.frames = sorted((f for m in members for f in m.frames),
                                key=lambda f: f.file_idx)
        cluster.low_cohesion = any(m.low_cohesion for m in members)
        cluster.compute_centroids()
        merged.append(cluster)
    return merged


