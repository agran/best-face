#!/usr/bin/env python3
"""
export_nima.py — скачать предобученные веса NIMA и экспортировать в ONNX.

Запускать один раз:
  python export_nima.py

Создаёт:
  models/nima_mobilenetv2.onnx  (~13 MB)

Источник весов:
  titu1994/neural-image-assessment  (Keras .h5, tag v0.3)
  https://github.com/titu1994/neural-image-assessment/releases

Архитектура:
  MobileNet (Keras Applications, 224x224, imagenet) +
  GlobalAveragePooling2D + Dropout(0.75) + Dense(10, softmax)
  Обучена на датасете AVA (оценки красоты 1..10).

Зависимости устанавливаются автоматически:
  tensorflow / keras  (для загрузки .h5)
  tf2onnx             (для конвертации в ONNX)
"""

import os
import sys
import urllib.request

# ─── Конфигурация ──────────────────────────────────────────────────────────────

OUT_ONNX  = 'models/nima_mobilenetv2.onnx'
H5_PATH   = 'models/nima_mobilenet_weights.h5'

# titu1994: v0.3 — обучена правильно (v0.1 некорректна).
# v0.2 и v0.3 дают схожие результаты; используем v0.3 как последнюю.
WEIGHT_URLS = [
    'https://github.com/titu1994/neural-image-assessment/releases/download/v0.3/mobilenet_weights.h5',
    'https://github.com/titu1994/neural-image-assessment/releases/download/v0.2/mobilenet_weights.h5',
]


# ─── Скачивание весов ──────────────────────────────────────────────────────────

def _progress_hook(block_num, block_size, total_size):
    done = block_num * block_size
    if total_size > 0:
        pct = min(100, done * 100 // total_size)
        mb  = done / 1024 / 1024
        tot = total_size / 1024 / 1024
        print(f'\r  {pct:3d}%  {mb:.1f} / {tot:.1f} MB', end='', flush=True)


def download_weights():
    if os.path.exists(H5_PATH):
        size = os.path.getsize(H5_PATH)
        if size > 5_000_000:
            print(f'Веса уже скачаны: {H5_PATH}  ({size//1024//1024} MB)')
            return True
        print(f'Файл {H5_PATH} повреждён (размер {size}), скачиваю заново...')
        os.remove(H5_PATH)

    print('Скачиваю веса NIMA MobileNet (titu1994/neural-image-assessment, v0.3)...')
    for url in WEIGHT_URLS:
        try:
            print(f'  {url}')
            tmp = H5_PATH + '.tmp'
            urllib.request.urlretrieve(url, tmp, _progress_hook)
            print()
            os.rename(tmp, H5_PATH)
            print(f'  OK: {H5_PATH}  ({os.path.getsize(H5_PATH)//1024//1024} MB)')
            return True
        except Exception as e:
            print(f'\n  Ошибка: {e}')
            if os.path.exists(H5_PATH + '.tmp'):
                os.remove(H5_PATH + '.tmp')

    print()
    print('─' * 60)
    print('Автоматическое скачивание не удалось.')
    print()
    print('Скачайте вручную:')
    print('  ' + WEIGHT_URLS[0])
    print(f'Сохраните как: {H5_PATH}')
    print('─' * 60)
    return False


# ─── Установка зависимостей ────────────────────────────────────────────────────

def _pip_install(*packages):
    import subprocess
    ret = subprocess.call([sys.executable, '-m', 'pip', 'install', '--quiet', *packages])
    return ret == 0


def ensure_deps():
    missing = []
    try:
        import tensorflow as tf   # noqa
    except ImportError:
        missing.append('tensorflow')

    try:
        import tf2onnx            # noqa
    except ImportError:
        missing.append('tf2onnx')

    if missing:
        print(f'Устанавливаю: {" ".join(missing)} ...')
        ok = _pip_install(*missing)
        if not ok:
            print(f'ERROR: не удалось установить {missing}')
            sys.exit(1)
        print('Установлено.')


# ─── Построение архитектуры NIMA (Keras) ──────────────────────────────────────

def build_nima_keras():
    """
    Воспроизвести архитектуру из titu1994/neural-image-assessment:
      MobileNet(alpha=1.0, input_shape=(224,224,3), include_top=False)
      GlobalAveragePooling2D -> Dropout(0.75) -> Dense(10, softmax)
    """
    from tensorflow import keras

    base = keras.applications.MobileNet(
        alpha=1.0,
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None,
    )
    x = base.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(rate=0.75)(x)
    x = keras.layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=base.input, outputs=x)
    return model


# ─── Конвертация ──────────────────────────────────────────────────────────────

def convert_to_onnx():
    import subprocess
    print(f'Конвертирую {H5_PATH} -> {OUT_ONNX} через tf2onnx...')

    # tf2onnx CLI умеет конвертировать Keras .h5 напрямую
    ret = subprocess.call([
        sys.executable, '-m', 'tf2onnx.convert',
        '--keras',  H5_PATH,
        '--output', OUT_ONNX,
        '--opset',  '13',
    ])
    if ret == 0 and os.path.exists(OUT_ONNX):
        return True

    # Fallback: загрузить модель вручную и экспортировать через Python API
    print('CLI не сработал, пробую через Python API...')
    return convert_via_python_api()


def convert_via_python_api():
    import tensorflow as tf
    import tf2onnx

    print('  Строю архитектуру NIMA...')
    model = build_nima_keras()

    print('  Загружаю веса...')
    try:
        model.load_weights(H5_PATH)
        print('  load_weights: OK')
    except Exception as e:
        print(f'  load_weights: {e}')
        print('  Пробую load_model...')
        from tensorflow import keras
        model = keras.models.load_model(H5_PATH, compile=False)
        print('  load_model: OK')

    model.trainable = False

    print('  Конвертирую в ONNX...')
    input_sig = [tf.TensorSpec([None, 224, 224, 3], tf.float32, name='input')]
    tf2onnx.convert.from_keras(
        model,
        input_signature=input_sig,
        opset=13,
        output_path=OUT_ONNX,
    )
    return os.path.exists(OUT_ONNX)


# ─── Проверка готовой ONNX-модели ─────────────────────────────────────────────

def verify_onnx():
    import numpy as np
    import onnxruntime as ort

    print(f'Проверяю {OUT_ONNX}...')
    sess = ort.InferenceSession(OUT_ONNX, providers=['CPUExecutionProvider'])
    inp  = sess.get_inputs()[0]
    out_name = sess.get_outputs()[0].name

    dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
    probs = sess.run([out_name], {inp.name: dummy})[0][0]
    mean_score = float(sum((i + 1) * probs[i] for i in range(10)))
    print(f'  OK: тестовый NIMA score = {mean_score:.2f}/10')
    print(f'  Размер файла: {os.path.getsize(OUT_ONNX) // 1024 // 1024} MB')
    print(f'  Форма входа: {inp.shape}  (NHWC)')
    return inp.shape


# ─── Патч NimaScorer в embed.py под NHWC (Keras-формат) ───────────────────────

def patch_embed_for_nhwc():
    """
    Keras-модель ожидает NHWC [N, 224, 224, 3].
    Исходный NimaScorer написан под NCHW [N, 3, 224, 224] (PyTorch-стиль).
    Патчим embed.py: убираем transpose, формат остаётся HWC.
    """
    embed_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline', 'embed.py')
    with open(embed_path, 'r', encoding='utf-8') as f:
        src = f.read()

    nchw_line = "rgb.transpose(2, 0, 1)[np.newaxis]  # [1, 3, 224, 224]"
    nhwc_line = "rgb[np.newaxis]                       # [1, 224, 224, 3] NHWC (Keras)"

    if nchw_line in src:
        src = src.replace(nchw_line, nhwc_line)
        with open(embed_path, 'w', encoding='utf-8') as f:
            f.write(src)
        print('  embed.py обновлён: NimaScorer теперь использует NHWC.')
    elif nhwc_line in src:
        print('  embed.py уже в NHWC-режиме.')
    else:
        print('  ВНИМАНИЕ: не удалось автоматически патчить embed.py.')
        print('  Проверьте NimaScorer.__call__ вручную — нужно убрать transpose.')


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs('models', exist_ok=True)

    if os.path.exists(OUT_ONNX):
        size = os.path.getsize(OUT_ONNX)
        if size > 1_000_000:
            print(f'Модель уже существует: {OUT_ONNX}  ({size//1024//1024} MB)')
            print('Удалите файл если хотите пересоздать.')
            return
        print(f'Файл {OUT_ONNX} повреждён (размер {size}), удаляю...')
        os.remove(OUT_ONNX)

    if not download_weights():
        sys.exit(1)

    ensure_deps()

    ok = convert_to_onnx()
    if not ok or not os.path.exists(OUT_ONNX):
        print('ERROR: конвертация не удалась.')
        sys.exit(1)

    verify_onnx()
    patch_embed_for_nhwc()

    print()
    print('=' * 50)
    print('ГОТОВО!  Модель:', OUT_ONNX)
    print()
    print('Запуск:')
    print('  python best_face.py "photos/..." --output output --no-contact-sheet')
    print('=' * 50)


if __name__ == '__main__':
    main()
