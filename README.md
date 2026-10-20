# Best-Face

Автоматический отбор лучшего фото из серий. На вход — папка с фотографиями
(несколько кадров каждого человека вперемешку), на выход — по одному лучшему кадру
на человека: самому резкому, фронтальному, с открытыми глазами и наиболее
«узнаваемому» для систем распознавания лиц.

## Как это работает

1. Детекция лица (YuNet, OpenCV) и выравнивание по 5 точкам.
2. Эмбеддинги: ArcFace `w600k_mbf` (512-dim) для группировки; опционально Human FaceRes (1024-dim) для метрики узнаваемости.
3. Группировка кадров: среднее сцепление (average-linkage) по cosine-сходству с поправкой на время съёмки (EXIF). Режим `person` (по умолчанию) собирает все кадры человека в один кластер и выбирает один лучший кадр на человека, даже если его снимали несколько раз; режим `series` сначала режет папку на серии по паузам в съёмке (EXIF), затем разделяет людей внутри серии и выбирает лучший кадр на каждую серию.
4. Метрики качества кадра: резкость, фронтальность и поза головы (yaw/pitch по MediaPipe), открытость глаз (EAR + blendshapes), выражение (blendshapes), размер лица, экспозиция, уверенность детектора, близость к центру серии, эстетика лица (NIMA).
5. Жёсткие фильтры + взвешенная сумма → лучший кадр. Веса и пороги — в `config.yaml`.
6. Результат: `best/` (лучшие кадры), `rejected/`, `report.json` и HTML-контактный лист `contact_sheet.html`.

Персоны (серии) нумеруются в порядке исходной папки — по лучшему кадру.

### Кэш и скорость

Результаты первого этапа (детекция, эмбеддинги, метрики) кэшируются в `<папка результатов>/frame_cache.pkl`.
Повторный запуск с другими порогами или весами занимает секунды. Кэш сбрасывается при смене моделей или параметров детекции
и пропускается ключом `--no-cache`. Чтение JPEG идёт в потоках (`performance.workers` в конфиге).

### Контроль и подбор весов

В `contact_sheet.html` есть фильтр «только сомнительные» (близкие по score кадры, подозрение на склейку разных людей,
низкое качество). Кликом по превью можно выбрать лучший кадр вручную; выбор скачивается как `manual_choices.json`.
По таким выборам `tune_weights.py` обучает веса скоринга:

```bash
python tune_weights.py "<результаты>/report.json" manual_choices.json
```

## Установка

Нужен Python 3.10+.

```bash
pip install -r requirements.txt
```

На Windows ставится `onnxruntime-directml` (GPU AMD/Intel/NVIDIA), на macOS и Linux — `onnxruntime`.
Для CUDA на NVIDIA замените пакет на `onnxruntime-gpu` (в окружении должен быть **только один** пакет `onnxruntime*`).
Принудительный выбор провайдера: переменная `BEST_FACE_PROVIDER=cpu|cuda|dml|coreml`.

### Модели

Положите файлы в папку `models/` (пути настраиваются в `config.yaml`):

| Файл | Назначение | Откуда |
| --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | детектор лиц | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) |
| `w600k_mbf.onnx` | ArcFace 512-dim | архив [buffalo_s.zip](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip) (InsightFace) |
| `face_landmarker.task` | открытость глаз (EAR) | [MediaPipe](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) |
| `nima_mobilenetv2.onnx` | эстетика лица (Q9) | `python export_nima.py` (нужны tensorflow и tf2onnx) |
| `human_faceres.onnx` | узнаваемость, пол/возраст (опционально) | конвертация [HSE_FaceRec_tf](https://github.com/av-savchenko/HSE_FaceRec_tf) в ONNX |

Необязательные модели (`nima`, `ear`, `faceres`) можно отключить, указав `null` в `config.yaml`.
Без FaceRes метрика узнаваемости считается по ArcFace.

## Запуск

```bash
python best_face.py "путь/к/папке/с/фото" [--output папка] [--limit N] [--no-contact-sheet] [--no-cache]
```

По умолчанию результаты сохраняются в `<папка_с_фото>/best_face_output/`.

Для удобства:

- **Windows:** перетащите папку с фото на `run.bat`.
- **macOS:** двойной клик по `run.command` (откроется выбор папки).
- **Linux:** `./run.sh "путь/к/папке"`.

На macOS и Linux один раз выполните `chmod +x run.sh run.command`;
при первом запуске скрипт сам создаст `.venv` и поставит зависимости.

## Тесты

```bash
pip install pytest
python -m pytest tests
```

## Подробнее

Исходное проектное описание — в [ОПИСАНИЕ_ПРОЕКТА.md](ОПИСАНИЕ_ПРОЕКТА.md).
