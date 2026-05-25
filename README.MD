# CNN_LABZ

Репозиторий лабораторных работ по курсу **«Нейронные сети и компьютерное зрение»**.  
Содержит 8 лабораторных работ: от базовых многослойных персептронов до LSTM и ансамблей сверточных сетей.

---

## Стек технологий

| Назначение | Инструменты |
|---|---|
| Глубокое обучение (Labs 1–5, 8) | PyTorch ≥ 2.0, torchvision |
| Глубокое обучение (Labs 6–7) | TensorFlow ≥ 2.10 / Keras |
| Данные | CIFAR-100, спутниковые снимки Sentinel-2, облака точек (HDF5), аудио (WAV) |
| Метрики и препроцессинг | scikit-learn, scipy, scikit-image, opencv-python |
| Визуализация | matplotlib, TensorBoard |
| Экспорт моделей | ONNX |

---

## Структура репозитория

```
CNN_LABZ/
├── README.MD
├── requirements.txt
├── Dockerfile / docker-compose.yml
│
├── data/                        # Общие датасеты (не коммитятся)
│
├── src/                         # Общие модули — используются всеми лабами
│   ├── models.py                # Классы моделей (MLP, CNN, AE, LSTM, …)
│   ├── train.py                 # Циклы обучения с метриками
│   ├── data.py                  # Загрузка данных и DataLoader
│   ├── evaluate.py              # Оценка, отчёты классификации
│   └── utils.py                 # Построение графиков, сохранение артефактов
│
├── lab1/ … lab8/                # Лабораторные работы (см. ниже)
│
└── labN/                        # Шаблон структуры каждой работы
    ├── README.MD                # Описание, запуск, результаты
    ├── task.md                  # Условие задания
    ├── notebook.ipynb           # Jupyter-ноутбук
    ├── configs/config.yaml      # Гиперпараметры
    ├── scripts/run_labN.py      # Скрипт запуска
    └── outputs/
        ├── checkpoints/         # Сохранённые веса моделей
        ├── figures/             # Графики обучения и результатов
        ├── tables/              # CSV с историей обучения
        └── metrics/             # Итоговые метрики
```

---

## Установка и запуск

### Локально

```bash
# 1. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Скачать CIFAR-100 (нужен для Labs 1–5)
mkdir -p data/raw && cd data/raw
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xvzf cifar-100-python.tar.gz
cd ../..

# 4. Указать классы по варианту в lab1/configs/config.yaml
#    classes: [класс1, класс2, класс3]

# 5. Запустить нужную лабораторную
python lab1/scripts/run_lab1.py
```

### Docker

```bash
docker compose build
docker compose run cnn_labz          # Запуск обучения
docker compose up jupyter            # Jupyter → http://localhost:8888
```

---

## Лабораторные работы

### Lab 1 — Многослойный персептрон (MLP)

| | |
|---|---|
| **Тема** | Ручное обратное распространение · Autograd PyTorch · FC MLP · CIFAR-100 |
| **Модель** | 2-слойный MLP: Linear(3072→10) → ReLU → Linear(10→3) |
| **Оптимизатор** | SGD |
| **Результат** | train ≈ 99.5 % / test ≈ 74.7 % (переобучение — ключевой объект изучения) |

```bash
python lab1/scripts/run_lab1.py
```

---

### Lab 2 — Свёрточная нейронная сеть (CNN)

| | |
|---|---|
| **Тема** | CNN · три варианта пулинга · сравнение архитектур |
| **Модель** | 2-блочная CNN: Conv→ReLU→Pool×2 → Linear |
| **Эксперименты** | `avg` / `max` / `stride` пулинг |

```bash
python lab2/scripts/run_lab2.py
python lab2/scripts/run_lab2.py --all    # все варианты пулинга
```

---

### Lab 3 — Регуляризация и аугментация

| | |
|---|---|
| **Тема** | Dropout2d · Weight Decay · онлайн-аугментация · Label Smoothing |
| **Эксперименты** | dropout (0.0/0.2/0.5) · weight_decay (0/1e-5/1e-3) · augmentation (none/light/heavy) |

```bash
python lab3/scripts/run_lab3.py --all
```

---

### Lab 4 — Transfer Learning

| | |
|---|---|
| **Тема** | Предобученные модели · заморозка весов · fine-tuning · сравнение оптимизаторов |
| **Модели** | `resnet20` (чётный вариант) / `mobilenetv2_x0_5` (нечётный) |
| **Эксперименты** | frozen vs full fine-tune · SGD / Adam / AdamW |

```bash
python lab4/scripts/run_lab4.py --all
python lab4/scripts/run_lab4.py --model mobilenetv2 --all   # нечётный вариант
```

---

### Lab 5 — Автоэнкодер

| | |
|---|---|
| **Тема** | Автоэнкодер · латентное пространство · шумоподавление аудио |
| **Часть 1** | FC-автоэнкодер изображений CIFAR-100 (3 размера бутылочного горлышка) |
| **Часть 2** | Conv1D автоэнкодер шумоподавления аудио (STFT) |

```bash
python lab5/scripts/run_lab5_image.py
python lab5/scripts/run_lab5_audio.py   # требует lab5/data/audio.wav + noise.wav
```

---

### Lab 6 — PointNet (классификация облаков точек)

| | |
|---|---|
| **Тема** | 3D облака точек · PointNet · T-Net · классификация пород деревьев |
| **Стек** | TensorFlow / Keras |
| **Данные** | `lab6/data/v1.h5` (нечётный) / `v2.h5` (чётный) |

```bash
pip install tensorflow>=2.10.0 h5py>=3.7.0
python lab6/scripts/run_lab6.py
```

---

### Lab 7 — U-Net + Ансамблирование

| | |
|---|---|
| **Тема** | Семантическая сегментация · U-Net · функции потерь · ансамблевые методы |
| **Стек** | TensorFlow / Keras |
| **Данные** | `lab7/data/forest.tif` — мультиспектральный снимок Sentinel-2 (256×256×27) |
| **Эксперимент A** | 3 U-Net: потери `dice` / `bce+dice` / `tversky` |
| **Эксперимент B** | 3 ансамбля: Average / WeightedAverage / Stacking (Conv2D мета-сеть) |

```bash
pip install tensorflow>=2.10.0 scikit-image>=0.20.0 opencv-python>=4.7.0
python lab7/scripts/run_lab7.py --exp unet      # обучить 3 U-Net
python lab7/scripts/run_lab7.py --exp ensemble  # построить ансамбли
python lab7/scripts/run_lab7.py                 # всё сразу
```

---

### Lab 8 — LSTM для прогнозирования временных рядов

| | |
|---|---|
| **Тема** | Авторегрессия (AR) · двунаправленный LSTM · прогнозирование |
| **Стек** | PyTorch |
| **Часть 1** | AR-baseline (LinearRegression) + LSTM на синтетическом ряду |
| **Часть 2** | LSTM на реальных метеорологических данных (CSV: год/месяц/день/температура) |

```bash
# Часть 1 — данные генерируются автоматически
python lab8/scripts/run_lab8_synthetic.py

# Часть 2 — положите CSV в lab8/data/<город>.csv
python lab8/scripts/run_lab8_weather.py
```

---

## Соглашения об именовании артефактов

| Артефакт | Путь |
|---|---|
| Условие задания | `labN/task.md` |
| Описание и результаты | `labN/README.MD` |
| Jupyter-ноутбук | `labN/notebook.ipynb` |
| Скрипт запуска | `labN/scripts/run_labN.py` |
| Конфигурация | `labN/configs/config.yaml` |
| Лучшие веса | `labN/outputs/checkpoints/best_model.pth` |
| График обучения | `labN/outputs/figures/training_curves.png` |
| История по эпохам | `labN/outputs/tables/training_history.csv` |
| Итоговые метрики | `labN/outputs/metrics/metrics.csv` |
