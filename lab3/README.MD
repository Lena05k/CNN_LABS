# Лабораторная работа №3 — CNN с дропаутом и аугментацией данных

## Задание

Обучить свёрточную нейронную сеть на CIFAR-100 (классы по варианту из ЛР1)  
с применением техник **дропаута** и **аугментации данных**.  
Сравнить три варианта дропаута, три варианта L2-регуляризации и три варианта аугментации.

## Структура

```
lab3/
├── task.md              ← задание и теория (переименовано из Lab3.md)
├── notebook.ipynb       ← оригинальный ноутбук (переименовано из Lab3.ipynb)
├── configs/
│   └── config.yaml      ← гиперпараметры и варианты экспериментов
├── scripts/
│   └── run_lab3.py      ← точка входа
└── outputs/
    ├── dropout/         ← Experiment A: 3 варианта дропаута
    ├── weight_decay/    ← Experiment B: 3 варианта L2-штрафа
    ├── augmentation/    ← Experiment C: 3 варианта аугментации
    ├── tensorboard/     ← логи для TensorBoard
    ├── onnx/            ← лучшая модель в формате ONNX
    ├── dropout_comparison.csv
    ├── weight_decay_comparison.csv
    └── augmentation_comparison.csv
```

## Запуск

```bash
# Одиночный запуск (параметры из config.yaml)
python lab3/scripts/run_lab3.py

# Только эксперимент с дропаутом
python lab3/scripts/run_lab3.py --exp dropout

# Только эксперимент с аугментацией
python lab3/scripts/run_lab3.py --exp augmentation

# Все 3 группы экспериментов (9 обучений)
python lab3/scripts/run_lab3.py --all
```

## TensorBoard

```bash
tensorboard --logdir lab3/outputs/tensorboard
```

## Архитектура модели (`Cifar100CNNReg`)

```
NormalizeCNN (NHWC → NCHW, [0,255] → нормировано)
Conv2d(3, 64, kernel=3, stride=4)   # 32×32 → 8×8
ReLU
Dropout2d(p)
Conv2d(64, 128, kernel=3, padding=1) # 8×8 → 8×8
ReLU
AvgPool2d(4)                          # 8×8 → 2×2
Dropout2d(min(p+0.1, 0.9))
Flatten                               # 128×2×2 = 512
Linear(512, 3)
```

Параметров: ~77 тыс.

## Эксперименты

### A. Дропаут (aug=light, weight_decay=1e-5)
| Вариант | dropout_p (Conv1/Conv2) |
|---------|------------------------|
| 1       | 0.0 / 0.1              |
| 2       | 0.2 / 0.3              |
| 3       | 0.5 / 0.6              |

### B. L2-регуляризация (aug=light, dropout_p=0.2)
| Вариант | weight_decay |
|---------|-------------|
| 1       | 0.0         |
| 2       | 1e-5        |
| 3       | 1e-3        |

### C. Аугментация данных (dropout_p=0.2, weight_decay=1e-5)
| Вариант | Описание                                          |
|---------|--------------------------------------------------|
| none    | без аугментации                                   |
| light   | RandomHorizontalFlip + ColorJitter (слабая)       |
| heavy   | HorizontalFlip + ColorJitter + RandomAffine (сильная) |

## Классы (по варианту)

Те же 3 класса CIFAR-100, что в ЛР1 и ЛР2: `[0, 55, 58]`  
Указать свой вариант в `configs/config.yaml` → поле `classes`.
