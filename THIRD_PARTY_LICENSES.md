# Сторонние модели и лицензии

Код Best-Face использует готовые обученные модели. Ниже: откуда каждая, на каких условиях
распространяется и что с ней сделано. Тексты лицензий лежат в папке [licenses/](licenses/).

Файлы моделей в репозитории хранятся **без изменений** (кроме конвертации формата, где это указано),
и поставляются «как есть», без гарантий. Если вы используете проект в коммерческих целях,
самостоятельно проверьте условия, указанные в колонке «Оговорки».

## Модели, лежащие в репозитории (`models/`)

| Файл | Размер | Назначение | Источник | Лицензия | Оговорки |
| --- | --- | --- | --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | 0.2 МБ | детектор лиц YuNet | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), автор Shiqi Yu | MIT, [текст](licenses/yunet-opencv_zoo-LICENSE.txt) | уведомление об авторских правах сохранено в тексте лицензии |
| `face_landmarker.task` | 3.6 МБ | MediaPipe Face Landmarker: ландмарки, blendshapes (глаза, мимика, взгляд), поза головы | [Google MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) | Apache-2.0 (лицензия репозитория MediaPipe), [текст](licenses/mediapipe-LICENSE.txt) | файл `face_landmarker.task` взят без изменений с `storage.googleapis.com/mediapipe-models`; условия модели см. в её model card на странице выше |
| `human_faceres.onnx` | 13.3 МБ | «узнаваемость» (Q8), оценка пола и возраста | модель FaceRes из библиотеки [vladmandic/human](https://github.com/vladmandic/human) ([human-models](https://github.com/vladmandic/human-models), MIT), основанная на работах HSE ([av-savchenko/HSE_FaceRec_tf](https://github.com/av-savchenko/HSE_FaceRec_tf), Apache-2.0); в проекте Face-ID Kiosk сконвертирована из TF.js в ONNX | MIT (human-models) и Apache-2.0 (HSE_FaceRec_tf), [текст Apache-2.0](licenses/faceres-HSE_FaceRec_tf-LICENSE.txt) | происхождение установлено по документации проекта Face-ID Kiosk; побайтное соответствие опубликованным весам не сверялось. Веса обучены на открытых датасетах (для моделей HSE_FaceRec_tf — VGGFace2); их условия использования могут ограничивать коммерческое применение, **проверяйте самостоятельно** |
| `nima_mobilenetv2.onnx` | 12.3 МБ | эстетика лица (Q9), NIMA | веса [titu1994/neural-image-assessment](https://github.com/titu1994/neural-image-assessment) (MobileNet), собраны в ONNX скриптом `export_nima.py` | MIT, [текст](licenses/nima-titu1994-LICENSE.txt) | сеть обучена на датасете AVA (оценки эстетики снимков) с предобучением на ImageNet; условия этих датасетов могут ограничивать коммерческое применение, **проверяйте самостоятельно** |

## Модель, которая НЕ распространяется в репозитории

| Файл | Размер | Назначение | Источник | Лицензия |
| --- | --- | --- | --- | --- |
| `w600k_mbf.onnx` | 13.0 МБ | ArcFace: группировка кадров по людям | [InsightFace, пакет `buffalo_s`](https://github.com/deepinsight/insightface/releases/tag/v0.7) | **только некоммерческие исследовательские цели** |

Эта модель скачивается на ваш компьютер при первом запуске (`download_models.py`) прямо из официального
релиза InsightFace; в данном репозитории её копии нет. По условиям InsightFace
(см. [model_zoo/README.md](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md):
*«ALL models are available for non-commercial research purposes only»*) она разрешена
**только для некоммерческого исследовательского использования**. Пользуясь программой, вы принимаете
эти условия самостоятельно. Для коммерческого применения замените эту модель на подходящую по лицензии
(потребуется заново подобрать пороги кластеризации в `config.yaml`).

## Библиотеки

Программа использует открытые Python-пакеты (OpenCV, NumPy, SciPy, scikit-learn, ONNX Runtime, MediaPipe,
Pillow, PyYAML, tqdm), ставящиеся через `requirements.txt` со своими лицензиями. Они в репозитории не лежат.
