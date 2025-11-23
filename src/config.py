# Настройки проекта — где лежат данные, куда сохранять результаты

from pathlib import Path

# Путь к корню проекта (откуда запускается main.py)
ROOT_DIR = Path(__file__).parent.parent

# Пути к файлам и папкам
DATA_PATH = ROOT_DIR / "data" / "financials.csv"           # исходные данные
MODELS_DIR = ROOT_DIR / "models"                          # сюда сохраняем модели
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"                         # графики
# PREDICTIONS_DIR = RESULTS_DIR / "predictions"             # таблицы с прогнозами

# Создаём все нужные папки, если их нет
for directory in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Общие параметры
RANDOM_STATE = 42          # для воспроизводимости
TEST_SIZE = 0.2            # 20% данных — на тест
TARGET = "Market Cap"      # целевая переменная — рыночная капитализация