"""
Модуль для подготовки данных.
Загружаем данные, чистим, добавляем новые признаки, делим на части
Если ошибка, программа запишет в лог и попробует не остановиться
"""

# Настройки:
from configs.config import DATA_PATH, TARGET, RANDOM_STATE, TEST_SIZE, logger
# Объединяет шаги для чисел и слов
from sklearn.compose import ColumnTransformer
# Заменен StandardScaler на RobustScaler для устойчивости к выбросам
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer
# Заполняет пустые места
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline  # Конвейер шагов
# Для деления данных
from sklearn.model_selection import train_test_split
import numpy as np  # Для чисел
import pandas as pd  # Для таблиц
import sys  # Для поиска файлов
from pathlib import Path  # Для путей к файлам

ROOT_DIR = Path(__file__).parent.parent  # Главная папка проекта
sys.path.insert(0, str(ROOT_DIR))  # Добавляем путь
sys.path.insert(0, str(ROOT_DIR / 'src'))  # К другим частям


# Вложенная функция для добавления производных признаков (вынесена на уровень модуля для pickling)
# Эта функция должна быть на уровне модуля, чтобы FunctionTransformer мог сохранить Pipeline с joblib
def _create_derived_features(X):
    """Добавляет вычисляемые признаки (коэффициенты, логарифмы)
    Всегда создаём копию, чтобы избежать SettingWithCopyWarning"""
    X = X.copy()
    # Добавляем EBITDA_Price_Ratio если есть необходимые столбцы
    if 'EBITDA' in X.columns and 'Price' in X.columns:
        X['EBITDA_Price_Ratio'] = X['EBITDA'] / X['Price'].clip(lower=1e-6)
    # Добавляем Earnings_Price_Ratio
    if 'Earnings/Share' in X.columns and 'Price' in X.columns:
        X['Earnings_Price_Ratio'] = X['Earnings/Share'] / \
            X['Price'].clip(lower=1e-6)
    # Логарифмиремые версии для стабилизации распределения
    if 'Price' in X.columns:
        X['Price_log'] = np.log1p(X['Price'].clip(lower=0))
    if 'EBITDA' in X.columns:
        X['EBITDA_log'] = np.log1p(X['EBITDA'].clip(lower=0))
    if 'Price/Sales' in X.columns:
        X['Price/Sales_log'] = np.log1p(X['Price/Sales'].clip(lower=0))
    return X


def load_data() -> pd.DataFrame:
    """
    Что делает: Загружает данные из файла CSV
    Возвращает: Таблицу данных или ошибку
    Ошибки: Если файл не найден, подсказка где искать
    """
    try:
        df = pd.read_csv(DATA_PATH)  # Читаем файл
        # Сколько всего
        logger.info(
            f"Данные загружены: {df.shape[0]} строк, {df.shape[1]} столбцов.")
        return df
    except FileNotFoundError:
        # Подсказка.
        logger.error(
            f"Файл не найден: {DATA_PATH}. Убедитесь, что financials.csv в data/.")
        raise
    except Exception as e:
        # Любая проблема.
        logger.error(f"Ошибка загрузки данных: {e}. Проверьте файл.")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Что делает: Чистит данные
    Удаляет копии, убирает отрицательные, фильтрует цену >0
    Параметры: df — таблица
    Возвращает: Чистую таблицу
    Отрицательные — как убытки, мы их обрезаем к 0, чтобы модель видела только положительное. Дубликаты — одинаковые строки, убираем, чтобы не повторять
    """
    try:
        # Запоминаем размер до чистки
        initial_shape = df.shape
        df = df.drop_duplicates()  # Убираем копии
        # Только положительные цены, как реальные вещи
        df = df[df['Price'] > 0]
        # Обрезаем отрицательные в столбцах
        for col in ['Earnings/Share', 'EBITDA', 'Price/Earnings']:
            if col in df:
                # Clip — обрезаем ниже 0, как убираем минусы
                df[col] = df[col].clip(lower=0)
                logger.info(f"Обрезали отрицательные в {col} к 0.")
        logger.info(
            f"Чистка завершена. Размер изменился: {initial_shape} -> {df.shape}")
        return df
    except KeyError as e:
        logger.error(
            f"Ошибка чистки: столбец не найден {e}. Проверьте данные.")
        raise
    except Exception as e:
        logger.error(f"Ошибка чистки: {e}. Возможно, типы данных.")
        raise


def feature_engineering_func(df: pd.DataFrame) -> pd.DataFrame:
    """
    Что делает: Добавляет новые признаки
    Параметры: df — таблица.
    Возвращает: Таблицу с новыми столбцами.
    Новые признаки — это расчёты из старых. Log — делает числа меньше, чтобы модель лучше работала с 'кривым' распределением.
    """
    try:
        # Названия новых признаков
        new_ratio_cols = ['EBITDA_Price_Ratio', 'Earnings_Price_Ratio']
        new_log_cols = [f'{col}_log' for col in [
            'Price', 'EBITDA', 'Price/Sales']]
        new_cols = new_ratio_cols + new_log_cols

        # Если все новые колонки уже присутствуют — ничего не делаем (уменьшаем шум лога)
        if all(col in df.columns for col in new_cols):
            return df

        # Вычисляем только отсутствующие признаки
        if 'EBITDA_Price_Ratio' not in df.columns:
            df['EBITDA_Price_Ratio'] = df['EBITDA'] / \
                df['Price'].clip(lower=1e-6)
        if 'Earnings_Price_Ratio' not in df.columns:
            df['Earnings_Price_Ratio'] = df['Earnings/Share'] / \
                df['Price'].clip(lower=1e-6)

        for col in ['Price', 'EBITDA', 'Price/Sales']:
            target_col = f'{col}_log'
            if target_col not in df.columns:
                df[target_col] = np.log1p(df[col].clip(lower=0))

        logger.info(
            "Новые признаки добавлены: ratios и logs для лучшей модели.")
        return df
    except KeyError as e:
        logger.error(f"Ошибка новых признаков: столбец не найден {e}.")
        raise
    except Exception as e:
        logger.error(f"Ошибка новых признаков: {e}. Возможно, деление на 0.")
        raise


def build_preprocess_pipeline():
    """
    Создаёт и возвращает надёжный sklearn Pipeline для подготовки признаков
    Структура:
    - FunctionTransformer: вычисление пользовательских признаков (idempotent)
    - ColumnTransformer: числовая обработка (импутация + RobustScaler) и категориальная (импутация + OneHot)
    Возвращает: sklearn.Pipeline
    """
    try:
        # Числовые признаки из реальных данных (EBITDA_Price_Ratio и другие вычисляются в feature_engineering_func)
        # Исходные столбцы: Price, Price/Earnings, Dividend Yield, Earnings/Share, 52 Week Low/High, EBITDA, Price/Sales, Price/Book
        numeric_features = [
            'Price', 'Price/Earnings', 'Dividend Yield', 'Earnings/Share',
            '52 Week Low', '52 Week High', 'EBITDA', 'Price/Sales', 'Price/Book'
        ]

        # Трансформер для числовых: сначала impute медианой, затем RobustScaler для устойчивости к выбросам
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])

        # Категориальные признаки и их трансформер (impute most frequent + one-hot encoding)
        # В данных: 'Sector'. 'Country' может отсутствовать — ColumnTransformer умеет с этим работать
        categorical_features = ['Sector']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Объединяем в ColumnTransformer; остальные столбцы отбрасываем (remainder='drop')
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )

        # FunctionTransformer для безопасного добавления новых признаков
        # Используем _create_derived_features, вынесенную на уровень модуля для поддержки pickling (joblib)
        feature_engineer = FunctionTransformer(
            _create_derived_features, validate=False)

        # Полный пайплайн: сначала добавляем новые признаки, затем применяем ColumnTransformer
        # Теперь ColumnTransformer будет работать с исходными + вычисляемыми признаками
        full_pipeline = Pipeline(steps=[
            ('feature_engineer', feature_engineer),
            ('preprocessor', preprocessor)
        ])

        logger.info(
            'Конвейер подготовки собран: feature_engineer + preprocessor.')
        return full_pipeline
    except Exception as e:
        logger.error(f'Ошибка сборки конвейера: {e}. Проверьте столбцы.')
        raise


def preprocess_main() -> tuple:
    """
    Что делает: Главная функция подготовки
    Загружает, чистит, делит, возвращает части
    Возвращает: X_train (признаки для обучения), X_test (для проверки), y_train (цель для обучения), y_test (для проверки), конвейер
    Делим данные на train (80%, для учебы) и test (20%, для экзамена), чтобы модель не 'подглядывала'
    """
    try:
        df = load_data()  # Загружаем
        df = clean_data(df)  # Чистим

        # Делим на признаки (X — вопросы) и цель (y — ответы)
        # Убираем ненужное
        X = df.drop(columns=[TARGET, 'Symbol', 'Name', 'SEC Filings'])
        y = df[TARGET]  # Цель — Market Cap.

        # Делим на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            # Test_size — доля, random_state — фиксированный разрез
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        logger.info(
            f"Данные разделены: train {X_train.shape[0]}, test {X_test.shape[0]}")

        preprocessor = build_preprocess_pipeline()  # Собираем конвейер
        return X_train, X_test, y_train, y_test, preprocessor
    except Exception as e:
        logger.error(f"Ошибка в главной подготовке: {e}. Проверьте начало.")
        raise


# Если запуск отдельно (для теста)
if __name__ == '__main__':
    preprocess_main()  # Запускаем главную


def get_feature_names(preprocessor, X_sample):
    """
    Возвращает имена признаков после преобразования preprocessor (Pipeline)
    Пытается получить имена из ColumnTransformer через get_feature_names_out,
    при неудаче генерирует простые имена feature_0..n.
    """
    try:
        # Если preprocessor — Pipeline, ищем шаг 'preprocessor' (ColumnTransformer внутри)
        if hasattr(preprocessor, 'named_steps'):
            steps = preprocessor.named_steps
            # В нашем Pipeline: feature_engineer -> preprocessor (ColumnTransformer)
            if 'preprocessor' in steps:
                ct = steps['preprocessor']
                # Попробуем получить имена из ColumnTransformer
                try:
                    if hasattr(ct, 'get_feature_names_out'):
                        return ct.get_feature_names_out(X_sample.columns) if hasattr(X_sample, 'columns') else ct.get_feature_names_out()
                except Exception:
                    pass

        # Фоллбек: трансформируем одну строку и генерируем имена
        transformed = preprocessor.transform(X_sample.head(1))
        return [f'feature_{i}' for i in range(transformed.shape[1])]
    except Exception as e:
        # Последний resort: просто генерируем имена на основе числа признаков
        logger.warning(
            f"Не удалось получить имена признаков: {e}. Используем fallback.")
        return [f'feature_{i}' for i in range(X_sample.shape[1])]
