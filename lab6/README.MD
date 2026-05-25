# Лабораторная работа №6 — PointNet: классификация пород деревьев

## Задание
Обучить PointNet для классификации пород деревьев по файлам облаков точек (3D).

**Стек:** TensorFlow / Keras

## Данные
Скачайте h5-файл из репозитория курса и поместите в `lab6/data/`:

| Вариант    | Файл   |
|------------|--------|
| Нечётный   | v1.h5  |
| Чётный     | v2.h5  |

## Установка зависимостей
```bash
pip install tensorflow>=2.10.0 h5py>=3.7.0
```

## Запуск
```bash
python lab6/scripts/run_lab6.py
python lab6/scripts/run_lab6.py --data lab6/data/v2.h5   # чётный вариант
```

## Архитектура PointNet
```
Input (N, 4096, 3)
  → T-Net(3)           # пространственная трансформация
  → Conv1D(32) × 2
  → T-Net(32)          # признаковая трансформация
  → Conv1D(32,64,512)
  → GlobalMaxPool1D
  → Dense(256) → Dropout(0.3)
  → Dense(128) → Dropout(0.3)
  → Dense(num_classes, softmax)
```

## Выходы
```
lab6/outputs/
├── figures/training_history.png
├── figures/confusion_matrix.png
├── figures/point_cloud_samples.png
├── checkpoints/pointnet_best.keras
├── metrics/metrics.csv
└── tables/training_history.csv
```
