import os

# Путь
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "financials.csv")

# Папки для результатов
MODELS_PATH = os.path.join(BASE_DIR, "models")
PLOTS_PATH = os.path.join(BASE_DIR, "results", "plots")

# Создаём папки, если их нет
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)

# Признаки
FEATURES = [
    'Price', 'Price/Earnings', 'Dividend Yield', 'Earnings/Share',
    '52 Week Low', '52 Week High', 'EBITDA', 'Price/Sales', 'Price/Book'
]

# Целевая переменная
TARGET = 'Market Cap'