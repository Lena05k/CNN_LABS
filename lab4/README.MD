# Лабораторная работа №4 — Transfer Learning

## Задание
Загрузить предобученную модель по варианту, заморозить веса и провести дообучение на классах CIFAR-100 из ЛР1.

| Вариант    | Модель             |
|------------|--------------------|
| Чётный     | `resnet20`         |
| Нечётный   | `mobilenetv2_x0_5` |

## Запуск
```bash
python lab4/scripts/run_lab4.py            # одиночный прогон (frozen + SGD)
python lab4/scripts/run_lab4.py --exp mode       # frozen vs full fine-tune
python lab4/scripts/run_lab4.py --exp optimizer   # SGD vs Adam vs AdamW
python lab4/scripts/run_lab4.py --all             # все эксперименты (5 обучений)
python lab4/scripts/run_lab4.py --model mobilenetv2 --all   # нечётный вариант
tensorboard --logdir lab4/outputs/tensorboard
```

## Эксперименты
- **A**: `frozen` (только голова ~195 параметров) vs `full` (все ~272 тыс.) — оптимизатор SGD
- **B**: Сравнение оптимизаторов при замороженных весах: SGD / Adam / AdamW

## Архитектура (resnet20)
```
NormalizeCNN → CifarResNet20 (backbone) → Linear(64 → 3)
```
