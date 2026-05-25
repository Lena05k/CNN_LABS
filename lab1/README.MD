# Lab 1 — Introduction to PyTorch: Regression and Classification

## Overview

Three progressively complex ML tasks implemented in PyTorch from scratch:

| Part | Task | Technique |
|------|------|-----------|
| 1 | Sigmoid approximation (regression) | Manual backpropagation, full-batch GD |
| 2 | Binary classification (XOR / circles) | PyTorch `autograd`, manual weight update |
| **3** | **CIFAR-100 3-class classification** | `nn.Module`, `DataLoader`, `SGD`, `CrossEntropyLoss` |

**Main deliverable is Part 3.** Parts 1–2 demonstrate foundational concepts interactively in `notebook.ipynb`.

---

## Dataset

**CIFAR-100** — 3 fine-label classes selected by student variant.

| Split | Samples/class | Total | Image size |
|-------|---------------|-------|------------|
| Train | 500 | 1 500 | 32×32 RGB |
| Test | 100 | 300 | 32×32 RGB |

Input to model: NHWC image (uint8) → normalized → flattened to 3 072 features.

**Download:**
```bash
cd data/raw
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xvzf cifar-100-python.tar.gz
```

---

## Model Architecture

```
Input (N, 32, 32, 3)  uint8
       ↓
   Normalize            /255 → subtract channel mean → divide channel std → flatten
       ↓
(N, 3072)
       ↓
  Linear(3072, H)       H = hidden_size  [default: 10]
       ↓
    ReLU
       ↓
  Linear(H, 3)
       ↓
(N, 3)  logits
```

| Layer | Params (hidden=10) |
|-------|-------------------|
| Normalize | 0 (fixed CIFAR-100 stats) |
| Linear(3072 → 10) | 30 730 |
| Linear(10 → 3) | 33 |
| **Total** | **30 763** |

---

## Configuration

`configs/config.yaml` — change before running:

```yaml
classes: [0, 55, 58]   # ← YOUR VARIANT CLASSES
hidden_size: 10
lr: 0.005
batch_size: 128
epochs: 250
```

**Variant rule:**

| Class slot | Rule |
|---|---|
| 1 | Group number + 7 |
| 2 | Variant number + 30 |
| 3 | ИУ5: variant+50 / ГУИМЦ: 90 / International: 93 |

---

## Quickstart

### Local

```bash
# From project root:
python lab1/scripts/run_lab1.py

# Custom config:
python lab1/scripts/run_lab1.py --config lab1/configs/config.yaml
```

### Docker

```bash
docker compose build
docker compose run cnn_labz        # runs run_lab1.py

docker compose up jupyter          # → http://localhost:8888
```

---

## Outputs

```
lab1/outputs/
├── checkpoints/
│   ├── best_model.pth         # checkpoint with highest val accuracy
│   └── last_model.pth         # final epoch checkpoint
├── figures/
│   └── training_curves.png    # loss and accuracy vs. epoch
├── tables/
│   └── training_history.csv   # per-epoch: train/val loss, train/val acc
└── metrics/
    ├── config.json            # experiment config snapshot
    └── metrics.csv            # final accuracy and macro F1 for train/test
```

---

## Results

### Baseline — FC(10), FC(3) | lr=0.005 | batch=128 | epochs=250

#### Final Metrics

| Split | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|-------|----------|-----------------|--------------|----------|
| Train | **99.53%** | 0.9954 | 0.9953 | 0.9953 |
| Test  | **74.67%** | 0.7407 | 0.7467 | 0.7415 |

Разрыв ~25 п.п. между обучающей и тестовой выборками — явный признак переобучения.

#### Per-class Report — Train

| Класс | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0     | 1.0000    | 0.9900 | 0.9950 | 500 |
| 55    | 0.9881    | 1.0000 | 0.9940 | 500 |
| 58    | 0.9980    | 0.9960 | 0.9970 | 500 |
| **macro avg** | **0.9954** | **0.9953** | **0.9953** | 1500 |

#### Per-class Report — Test

| Класс | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0     | 0.8174    | 0.9400 | 0.8744 | 100 |
| 55    | 0.7158    | 0.6800 | 0.6974 | 100 |
| 58    | 0.6889    | 0.6200 | 0.6526 | 100 |
| **macro avg** | **0.7407** | **0.7467** | **0.7415** | 300 |

#### Анализ по классам

- **Класс 0** — лучший результат (F1=0.87 на тесте). Наибольший recall (0.94): модель почти не пропускает примеры этого класса.
- **Класс 55** — средний результат (F1=0.70). Низкий recall (0.68): треть примеров класса классифицируется неверно.
- **Класс 58** — наихудший результат (F1=0.65). И precision, и recall ниже 0.70.

#### Переобучение

На обучающей выборке модель достигает ~99.5% уже к ~80–100 эпохе, тогда как тестовая точность перестаёт расти. Переобучение устраняется без регуляризации через:
- уменьшение числа эпох (ранняя остановка по val loss);
- уменьшение размера скрытого слоя.

---

### Эксперименты по самостоятельной работе

| Конфигурация | lr | batch | epochs | Train Acc | Test Acc | Комментарий |
|---|---|---|---|---|---|---|
| FC(10), FC(3) | 0.005 | 128 | 250 | 99.53% | 74.67% | Базовый вариант |
| FC(10), FC(3) | 0.005 | 256 | 500 | — | — | batch×2, epochs×2 — равное число шагов |
| FC(10), FC(3) | 0.0017 | 128 | 750 | — | — | lr÷3, epochs×3 |
| FC(?), FC(3)  | — | — | — | — | — | Изменённая архитектура |

> Строки без результатов заполняются после запуска экспериментов.

---

## Artifact Formats

| Artifact | Format | Location |
|---|---|---|
| Model weights | `.pth` (PyTorch state dict) | `outputs/checkpoints/` |
| Training curves | `.png` (150 dpi) | `outputs/figures/` |
| Per-epoch metrics | `.csv` | `outputs/tables/` |
| Final metrics | `.csv` | `outputs/metrics/` |
| Config snapshot | `.json` | `outputs/metrics/` |
