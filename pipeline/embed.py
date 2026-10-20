"""
pipeline/embed.py — извлечение эмбеддингов лиц через ONNX Runtime.

Две модели:
  ArcFaceEmbedder   — w600k_mbf.onnx, 512-dim, используется для группировки серий.
  FaceResEmbedder   — human_faceres.onnx, 1024-dim, целевая метрика «узнаваемости»
                      (совпадает с моделью Face-ID Kiosk).

Форматы входных данных подтверждены через inspect:
  w600k_mbf:     [N, 3, 112, 112] float32, normalize (x-127.5)/127.5
  human_faceres: [N, 3, 224, 224] float32, raw RGB [0..255] (нормализация внутри графа)
"""

import os
import sys
import numpy as np
import cv2
from typing import Optional

# Предзагрузка CUDA/cuDNN DLL через ctypes ДО импорта onnxruntime.
# ORT C++ runtime использует LoadLibrary без LOAD_LIBRARY_SEARCH_USER_DIRS,
# поэтому os.add_dll_directory и изменение PATH не помогают после старта.
# Но если DLL уже загружена в процесс через ctypes, ORT находит её в кэше.
def _preload_cuda_dlls():
    import ctypes
    try:
        import site
        for sp in site.getsitepackages():
            nvidia_base = os.path.join(sp, 'nvidia')
            if not os.path.isdir(nvidia_base):
                continue
            for pkg in os.listdir(nvidia_base):
                bin_dir = os.path.join(nvidia_base, pkg, 'bin')
                if not os.path.isdir(bin_dir):
                    continue
                if hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(bin_dir)
                for dll in os.listdir(bin_dir):
                    if dll.lower().endswith('.dll'):
                        full = os.path.join(bin_dir, dll)
                        try:
                            ctypes.CDLL(full)
                        except OSError:
                            pass
    except Exception:
        pass

_preload_cuda_dlls()

import onnxruntime as ort


_PROVIDER_PRIORITY = ['CUDAExecutionProvider', 'DmlExecutionProvider',
                      'CoreMLExecutionProvider', 'CPUExecutionProvider']
_PROVIDER_ALIASES = {
    'cuda':   'CUDAExecutionProvider',
    'dml':    'DmlExecutionProvider',
    'coreml': 'CoreMLExecutionProvider',
    'cpu':    'CPUExecutionProvider',
}


def _best_providers() -> list:
    """
    Провайдеры по приоритету: CUDA (NVIDIA) → DirectML (Windows, AMD/Intel/любой DX12 GPU)
    → CoreML (macOS) → CPU. Принудительный выбор: BEST_FACE_PROVIDER=cpu|cuda|dml|coreml.
    """
    available = ort.get_available_providers()
    forced = _PROVIDER_ALIASES.get(os.environ.get('BEST_FACE_PROVIDER', '').strip().lower())
    if forced:
        return [forced, 'CPUExecutionProvider'] if forced != 'CPUExecutionProvider' \
            else ['CPUExecutionProvider']
    return [p for p in _PROVIDER_PRIORITY if p in available]


def _make_session(model_path: str) -> 'ort.InferenceSession':
    providers = _best_providers()
    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # без шума о недоступном CUDA при откате на другой провайдер
    if 'DmlExecutionProvider' in providers:
        # Требования DirectML EP
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(model_path, sess_options=opts, providers=providers)


def _l2_norm(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / n if n > 1e-9 else vec


class ArcFaceEmbedder:
    """
    InsightFace w600k_mbf: 512-dim ArcFace, вход 112×112.
    Нормализация: (RGB - 127.5) / 127.5 → NCHW.
    """
    INPUT_SIZE = 112
    DIM = 512

    def __init__(self, model_path: str):
        self._sess = _make_session(model_path)
        self._input_name  = self._sess.get_inputs()[0].name
        self._output_name = self._sess.get_outputs()[0].name
        used = self._sess.get_providers()[0]
        print(f'  ArcFaceEmbedder: {used}')

    def __call__(self, face_bgr_112: np.ndarray) -> Optional[np.ndarray]:
        """
        face_bgr_112 — BGR кроп (112, 112, 3) uint8.
        Возвращает L2-нормированный вектор (512,) float32 или None при ошибке.
        """
        try:
            rgb = cv2.cvtColor(face_bgr_112, cv2.COLOR_BGR2RGB).astype(np.float32)
            rgb = (rgb - 127.5) / 127.5               # [-1, 1]
            tensor = rgb.transpose(2, 0, 1)[np.newaxis]  # [1, 3, 112, 112]
            out = self._sess.run([self._output_name], {self._input_name: tensor})
            return _l2_norm(out[0][0])
        except Exception:
            return None


class FaceResEmbedder:
    """
    Human FaceRes (HSE_FaceRec_tf): 1024-dim, вход 224×224.
    Нормализация: raw RGB [0..255] — модель делает её внутри графа.
    Также возвращает gender/age (дополнительные выходы).
    """
    INPUT_SIZE = 224
    DIM = 1024

    def __init__(self, model_path: str):
        self._sess = _make_session(model_path)
        self._input_name   = self._sess.get_inputs()[0].name
        # Раскладка входа зависит от конвертации: NCHW [b,3,224,224] или NHWC [b,224,224,3]
        in_shape = self._sess.get_inputs()[0].shape
        self._nchw = len(in_shape) == 4 and in_shape[1] == 3
        # Именованные выходы: embedding, features256, gender, age
        self._out_names = [o.name for o in self._sess.get_outputs()]
        used = self._sess.get_providers()[0]
        print(f'  FaceResEmbedder:  {used} ({"NCHW" if self._nchw else "NHWC"})')
        self._warned = False

    def __call__(self, face_bgr_224: np.ndarray):
        """
        face_bgr_224 — BGR кроп (224, 224, 3) uint8.
        Возвращает (embedding_1024, gender_logit, age_probs) или (None, None, None).
        embedding — L2-нормированный (1024,) float32.
        gender_logit — float (>0 = female по модели Human; надо проверить знак).
        age_probs — (100,) float32 — распределение по 100 классам возраста (0-99 лет).
        """
        try:
            rgb = cv2.cvtColor(face_bgr_224, cv2.COLOR_BGR2RGB).astype(np.float32)
            # Без нормализации — модель делает (x/255 - mean) / std внутри
            tensor = (rgb.transpose(2, 0, 1) if self._nchw else rgb)[np.newaxis]
            outs = self._sess.run(self._out_names, {self._input_name: tensor})

            # Находим выходы по именам
            out_map = dict(zip(self._out_names, outs))
            emb = _l2_norm(out_map['embedding'][0])

            gender_logit = float(out_map.get('gender', [[0.0]])[0][0]) \
                           if 'gender' in out_map else None
            age_probs    = out_map.get('age', [None])[0] \
                           if 'age'    in out_map else None

            return emb, gender_logit, age_probs
        except Exception as e:
            if not self._warned:
                print(f'  [WARN] FaceResEmbedder: ошибка инференса: {e}')
                self._warned = True
            return None, None, None


def estimate_age_from_probs(age_probs: np.ndarray) -> float:
    """
    Оценить возраст как взвешенное среднее по распределению вероятностей.
    age_probs — (100,) вероятности для классов 0..99 лет.
    """
    ages = np.arange(100, dtype=np.float32)
    probs = np.asarray(age_probs, dtype=np.float32)
    probs = probs / (probs.sum() + 1e-9)
    return float(np.dot(probs, ages))


class NimaScorer:
    """
    NIMA (Neural Image Assessment) aesthetic quality scorer.

    Модель: MobileNetV2 backbone + Dropout + Linear(1280, 10) + Softmax.
    Обучена на датасете AVA (оценки красоты снимков 1..10).
    Применяется к уже выровненному кропу лица 224×224 — оценивает качество
    снимка лица: равномерность освещения, отсутствие пересветов, резкость.

    Вход:  BGR кроп 224×224 uint8.
    Выход: aesthetic score в [0, 1]  (= (mean_score − 1) / 9, где mean ∈ [1..10]).
           При недоступной модели возвращает 0.5 (нейтрально).

    Нормировка: ImageNet mean/std по каналам RGB.
    """
    INPUT_SIZE = 224
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path: Optional[str]):
        self._sess = None
        self._input_name  = None
        self._output_name = None

        if not model_path:
            print('  NimaScorer:       отключён (нет пути к модели)')
            return
        if not os.path.exists(model_path):
            print(f'  NimaScorer:       модель не найдена ({model_path}), Q9 отключён')
            print(f'  >> запустите: python export_nima.py')
            return

        self._sess = _make_session(model_path)
        self._input_name  = self._sess.get_inputs()[0].name
        self._output_name = self._sess.get_outputs()[0].name
        used = self._sess.get_providers()[0]
        print(f'  NimaScorer:       {used}')

    @property
    def available(self) -> bool:
        return self._sess is not None

    def __call__(self, face_bgr_224: np.ndarray) -> float:
        """
        Вернуть aesthetic score в [0, 1].
        При ошибке или отсутствии модели — 0.5 (нейтрально).
        """
        if self._sess is None:
            return 0.5
        try:
            rgb = cv2.cvtColor(face_bgr_224, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = (rgb - self._MEAN) / self._STD          # ImageNet normalization
            tensor = rgb[np.newaxis]                       # [1, 224, 224, 3] NHWC (Keras)
            probs = self._sess.run(
                [self._output_name], {self._input_name: tensor}
            )[0][0]                                       # (10,) softmax
            scores = np.arange(1, 11, dtype=np.float32)
            mean_score = float(np.dot(probs, scores))
            return (mean_score - 1.0) / 9.0               # → [0, 1]
        except Exception:
            return 0.5
