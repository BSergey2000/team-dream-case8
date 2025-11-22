import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from src.config import DATA_PATH, FEATURES, TARGET


def load_and_prepare_data():
    print("Загрузка данных...")
    df = pd.read_csv(DATA_PATH)

    print(f"Размер данных: {df.shape}")

    # НЕ УДАЛЯЕМ Name — она нужна для красивой таблички в конце!
    # Удаляем только технические колонки
    cols_to_drop = ['Symbol', 'Sector', 'SEC Filings']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    # Обрабатываем пропуски
    df['Price/Earnings'] = df['Price/Earnings'].fillna(df['Price/Earnings'].median())
    df['Price/Book'] = df['Price/Book'].fillna(df['Price/Book'].median())

    # Логарифмируем целевую переменную
    df['Market Cap_log'] = np.log1p(df[TARGET])

    print("Пропуски обработаны, логарифмирование выполнено")

    X = df[FEATURES]
    y = df['Market Cap_log']

    # Масштабирование
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES, index=df.index)  # сохраняем индекс!

    return df, X_scaled, y, scaler