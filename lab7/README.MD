# Лабораторная работа №7 — U-Net + Ансамблирование (Сегментация спутниковых снимков)

## Задание
Обучить несколько U-Net с разными функциями потерь для бинарной сегментации
повреждённых лесных массивов по мультиспектральным снимкам Sentinel-2.
Затем объединить их в ансамбль тремя методами.

## Данные
Мультиспектральный tif (256×256×27):
- Канал 0 — бинарная маска (повреждения леса)
- Каналы 1–13 — новые снимки Sentinel-2
- Каналы 14–26 — старые снимки Sentinel-2

Поместите файл в `lab7/data/forest.tif`.

## Запуск
```bash
pip install tensorflow>=2.10.0 scikit-image>=0.20.0 opencv-python>=4.7.0
python lab7/scripts/run_lab7.py                    # все эксперименты
python lab7/scripts/run_lab7.py --exp unet         # только U-Net (3 функции потерь)
python lab7/scripts/run_lab7.py --exp ensemble     # только ансамбли (по сохранённым чекпоинтам)
python lab7/scripts/run_lab7.py --data path/to.tif # другой файл данных
```

## Эксперименты
| Эксперимент | Описание |
|-------------|----------|
| **A** | 3 U-Net с разными функциями потерь: `dice`, `bce+dice`, `tversky` |
| **B** | 3 ансамблевых метода: Average / WeightedAverage / Stacking (Conv2D мета-сеть) |

## Архитектура U-Net
```
Input (256×256×16)
→ EncoderBlock(32) → EncoderBlock(64) → EncoderBlock(128) → EncoderBlock(256)
→ Bridge(512)
→ DecoderBlock(256) → DecoderBlock(128) → DecoderBlock(64) → DecoderBlock(32)
→ Conv2D(1, sigmoid)
```

## Метрики
- **Dice coefficient** (основная): 2·|A∩B| / (|A|+|B|)
- **Tversky** (penalty для FN): (TP) / (TP + α·FN + β·FP)

## Выходные файлы
```
lab7/outputs/
  figures/
    unet_training_history.png
    predictions.png
  checkpoints/
    unet_dice.keras
    unet_bce.keras
    unet_tversky.keras
    meta_learner.keras
  metrics/
    unet_metrics.csv
    ensemble_metrics.csv
    metrics.csv
  tables/
    training_history.csv
```
