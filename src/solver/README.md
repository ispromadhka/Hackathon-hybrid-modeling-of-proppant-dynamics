## Назначение

`src/solver` содержит численный CPU‑решатель (полная физика) и утилиты генерации/визуализации данных.

## Быстрый запуск

Из корня репозитория:

```bash
python app.py --generate --samples 500 --workers -1 --config configs/default.json
```

Это построит датасет `data/processed` через общий пайплайн (симуляции → тензоры → sample_*.npz).

## Генерация симуляций

Скрипт: `src/solver/generation.py`

Он:

- создаёт `simulation_data/` и `simulation_timeseries/`
- ведёт `simulation_results.csv` (метаданные, пути, базовые метрики)
- не запускает пересчёт, если симуляция уже есть и актуальна
- пересчитывает симуляцию, если она устарела:
  - `max_time < Tmax` из конфига
  - изменился `gen_hash` (сигнатура `grid/numerics/physics/boundary`)

## Параметры и соглашения

Параметры берутся из `configs/default.json` → `solver_generation`.

Основные параметры, которые входят в имя файла и `param_hash`:

- `c_in`: входная концентрация (в `case1.ipynb` пример: `0.45`)
- `w0`: ширина трещины (`w` в решателе), м
- `mu0`: базовая вязкость, Па·с
- `Q`: расход, м²/с (в проекте используется знак как в `case1.ipynb`: обычно `Q < 0`)
- `chi`: ширина зоны инжекции по `y`
- `c_in_times`: времена переключения входной концентрации (первое значение)
- `dT`: шаг сохранения по времени (выходные кадры)

Важно: поле, которое сохраняется в time‑series файле, называется `Q`, но это не расход. Это массив состояния решателя размера `(Nt, Ny, Nx)`. Для визуализации/обучения он приводится к концентрации как \(c = Q / w\).

## Структура выходных данных

В корне проекта:

```text
simulation_data/           финальные поля (последний кадр), .npy
simulation_timeseries/     временные ряды, *_series.npz
simulation_results.csv     индекс всех симуляций + метрики
```

Формат `simulation_timeseries/*_series.npz`:

- `Q`: `(n_frames, Ny, Nx)`
- `times`: `(n_frames,)`

## Использование результатов

```python
import pandas as pd
import numpy as np

df = pd.read_csv("simulation_results.csv")
r = df.iloc[0]
final_frame = np.load(r["matrix_path"])
ts = np.load(r["timeseries_path"])
Q_series = ts["Q"]
times = ts["times"]
```

## Визуализация

Streamlit‑визуализатор: `src/solver/visualizer.py`

Из корня:

```bash
streamlit run src/solver/visualizer.py
```

Отображение строится по концентрации \(c = Q/w\) с шкалой цвета `0..c_in` (как в `case1.ipynb`).
