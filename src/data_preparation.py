# Загрузка данных, создание признаков и разделение на train/test

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .config import DATA_PATH, RANDOM_STATE, TEST_SIZE, TARGET


def load_and_prepare_data():
    """
    Главная функция подготовки данных.
    Делает всё за один вызов:
    - грузит csv
    - чистит пропуски в целевой переменной
    - создаёт логарифмы и новые признаки
    - делает target encoding по секторам
    - стратифицированно делит на train/test
    - возвращает готовый словарь для обучения
    """
    print("Загрузка и подготовка данных...")

    # 1. Читаем исходный файл
    df = pd.read_csv(DATA_PATH)

    # 2. Убираем строки, где нет капитализации (иначе модель не сможет учиться)
    df = df.dropna(subset=[TARGET])                # удаляем пропуски в Market Cap
    df = df[df[TARGET] > 0].copy()                 # оставляем только положительные значения

    # 3. Логарифм целевой переменной — сильно улучшает качество
    df["log_target"] = np.log1p(df[TARGET])

    # 4. Логарифмируем основные числовые признаки (чтобы убрать перекос)
    for col in ["Price", "EBITDA", "Price/Earnings", "Price/Sales", "Price/Book"]:
        if col in df.columns:
            # clip(lower=...) защищает от логарифма нуля или отрицательных
            new_name = f"log_{col.lower().replace('/', '_').replace(' ', '_')}"
            df[new_name] = np.log1p(df[col].clip(lower=0.01))

    # 5. Полезные финансовые коэффициенты
    df["ebitda_yield"] = df["EBITDA"] / df[TARGET].replace(0, np.nan)   # EBITDA / Market Cap
    df["pe_inverse"] = 1.0 / df["Price/Earnings"].clip(lower=0.1)       # 1/(P/E) — чем выше прибыльность, тем лучше

    # 6. Target encoding по секторам (очень мощный признак)
    # Считаем среднюю log-капитализацию по сектору только на всей выборке до сплита — это честно
    sector_mean = df.groupby("Sector")["log_target"].mean()
    df["Sector_te"] = df["Sector"].map(sector_mean)

    # 7. Бины по капитализации — нужны только для стратифицированного сплита
    df["cap_bin"] = pd.qcut(df["log_target"], q=10, duplicates="drop", labels=False)

    # 8. Список всех признаков, которые пойдут в модель
    feature_columns = [
        "log_price", "log_ebitda", "log_price_earnings", "log_price_sales", "log_price_book",
        "ebitda_yield", "pe_inverse", "Dividend Yield", "Sector_te"
    ]

    # 9. Матрица признаков X и таргеты
    X = df[feature_columns].fillna(df[feature_columns].median())   # заполняем редкие пропуски медианой
    y_log = df["log_target"]                                        # логарифмированная капитализация
    y_real = df[TARGET]                                             # настоящая капитализация в долларах
    names = df["Name"]                                              # названия компаний (для красивых таблиц)

    # 10. ОДИН ЕДИНСТВЕННЫЙ сплит — передаём сразу все нужные колонки
    #     train_test_split вернёт ровно 8 объектов (по 2 на каждый входной массив)
    X_train, X_test, \
    y_train_log, y_test_log, \
    y_real_train, y_real_test, \
    names_train, names_test = train_test_split(
        X, y_log, y_real, names,           # 4 входных массива → 8 выходных
        test_size=TEST_SIZE,               # 20% на тест
        stratify=df["cap_bin"],            # сохраняем распределение по размеру компаний
        random_state=RANDOM_STATE          # для воспроизводимости
    )

    # Мы используем только тестовые y_real_test и names_test — тренировочные части не нужны дальше
    print(f"Готово: {len(X_train)} train | {len(X_test)} test | {len(feature_columns)} признаков")

    # 11. Возвращаем всё в одном словаре — удобно передавать дальше по пайплайну
    return {
        "X_train": X_train.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "y_train_log": y_train_log.reset_index(drop=True),
        "y_test_log": y_test_log.reset_index(drop=True),
        "y_test_real": y_real_test.reset_index(drop=True),   # настоящая капитализация на тесте
        "names_test": names_test.reset_index(drop=True),     # названия компаний на тесте
        "feature_names": feature_columns                     # список имён признаков (для графиков важности)
    }


# Если вдруг запустишь этот файл напрямую — просто покажет помощь
if __name__ == "__main__":
    data = load_and_prepare_data()
    print("Ключи в возвращаемом словаре:", data.keys())
    print("Пример названий компаний в тесте:", data["names_test"].head(10).tolist())