# Лабораторная работа №8 — LSTM для прогнозирования временных рядов

## Задание
Реализовать авторегрессионную модель (AR) и двунаправленный LSTM для прогнозирования временных рядов.
Сравнить модели на синтетических данных и применить LSTM к реальным метеорологическим данным.

## Запуск

### Часть 1 — синтетический ряд (AR vs LSTM)
```bash
python lab8/scripts/run_lab8_synthetic.py
python lab8/scripts/run_lab8_synthetic.py --config lab8/configs/config.yaml
```

### Часть 2 — реальные погодные данные
```bash
# Поместите CSV файлы в lab8/data/<city>.csv
# Формат: year,month,day,Temperature
python lab8/scripts/run_lab8_weather.py
python lab8/scripts/run_lab8_weather.py --city moscow spb    # явные файлы
```

## Данные

### Синтетический ряд
Генерируется автоматически по формуле:
```
y[i] = sin(i/50) - sin(i/200) + (2i/N)² + noise
```

### Погодные данные
CSV файлы с колонками: `year`, `month`, `day`, `Temperature`
Положите в `lab8/data/`:
- `city1.csv` — первый город
- `city2.csv` — второй город

## Архитектура LSTM
```
Input (batch, seq_len)
→ Unsqueeze → (batch, seq_len, 1)
→ BiLSTM(input=1, hidden=64, layers=2, bidirectional=True)
→ Last timestep (batch, 128)
→ Dropout(0.4)
→ Linear(128 → num_features)
```

## Эксперименты
| Эксперимент | Описание |
|-------------|----------|
| **A** | AR (LinearRegression, лаги=seq_len): одношаговый + рекурсивный прогноз |
| **B** | LSTM: скользящее окно, Adam, StepLR, gradient clipping |

## Метрики
- **R²** (coefficient of determination): чем ближе к 1 — тем лучше
- **RMSE**: среднеквадратичная ошибка

## Выходные файлы
```
lab8/outputs/
  figures/
    synthetic_series.png
    ar_prediction.png
    lstm_training.png
    lstm_prediction.png
    weather_<city>_series.png
    weather_<city>_prediction.png
  checkpoints/
    lstm_synthetic.pth
    lstm_weather_<city>.pth
  metrics/
    synthetic_metrics.csv
    weather_metrics.csv
  tables/
    synthetic_history.csv
    weather_history.csv
```
