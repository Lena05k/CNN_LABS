# Лабораторная работа №5 — Автоэнкодер

## Задание
**Часть 1:** Получить эмбеддинг изображений CIFAR-100 с помощью автоэнкодера и визуализировать его.
**Часть 2:** Загрузить собственную аудиозапись и использовать автоэнкодер для удаления шума.

## Запуск
```bash
# Часть 1 — Image AE (одиночный прогон, hidden=512)
python lab5/scripts/run_lab5_image.py

# Сравнение 3 размерностей скрытого пространства (hidden: 128, 256, 512)
python lab5/scripts/run_lab5_image.py --all

# Часть 2 — Audio Denoising AE
# Сначала подготовьте файлы:
# ffmpeg -y -i <ваша_музыка>    -ac 1 -ar 16000 lab5/data/audio.wav
# ffmpeg -y -i <ваш_шум.mp3>   -ac 1 -ar 16000 lab5/data/noise.wav
python lab5/scripts/run_lab5_audio.py

# Свои файлы:
python lab5/scripts/run_lab5_audio.py --audio my.wav --noise pink.wav
```

## Архитектура Image AE (hidden=512, bottleneck=64)
```
Normalize (NHWC → flat 3072)
Encoder: Linear(3072→512) → ELU → Linear(512→256) → ELU → Linear(256→64) → Tanh
Decoder: Linear(64→256)  → ELU → Linear(256→512) → ELU → Linear(512→3072)
```

## Архитектура Audio AE (Conv1d, input: (batch, 2, F))
```
Encoder: Conv1d(2→256, stride=2) → Conv1d(256→512, stride=2) → Conv1d(512→1024, stride=2)
Decoder: ConvTranspose1d(1024→512) → ConvTranspose1d(512→256) → ConvTranspose1d(256→2)
```

## Структура выходов
```
lab5/outputs/
├── image_ae/<hidden>/checkpoints, figures/, tables/, metrics/
├── image_ae/hidden_comparison.csv
└── audio_ae/checkpoints/denoising_ae.pth, figures/, metrics/, denoised.wav
```
