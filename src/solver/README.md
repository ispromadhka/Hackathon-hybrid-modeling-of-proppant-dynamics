## Запуск симуляций

'''
import sys
import os
module_path = os.path.abspath('../')
if module_path not in sys.path:
sys.path.append(module_path)

# Запустите скрипт напрямую - он сам:

# - Создаст все папки

# - Проверит предыдущие результаты

# - Запустит новые симуляции

# - Сохранит данные

'''

## Структура выходных данных

'''
├── simulation_data/ # Финальные поля концентраций (.npy)
├── simulation_timeseries/ # Временные ряды (.npz)
└── simulation_results.csv # Метаданные и метрики всех симуляций
'''

## Параметры симуляций

Основные параметры в коде:

c_in: входная концентрация (0.05)

w0: ширина пласта (0.01)

mu0: вязкость (0.001)

Q: скорость потока (-0.05)

chi: ширина зоны инжекции (H/6)

c_in_times: время подачи концентрации (50)

dT: шаг по времени (2)

Измените значения в соответствующих списках для новых экспериментов

# Использование результатов

'''
import pandas as pd
import numpy as np

# Метаданные всех симуляций

df = pd.read_csv('simulation_results.csv')

# Загрузка конкретной симуляции

param_hash = df.iloc[0]['param_hash']
matrix = np.load(df.iloc[0]['matrix_path']) # Финальный кадр
ts_data = np.load(df.iloc[0]['timeseries_path']) # Временной ряд
'''

## Структура данных

Для временных рядов:

'''
Q_series = ts_data['Q'] # Форма: (кадры, 100, 100)
times = ts_data['times'] # Временные метки
'''
