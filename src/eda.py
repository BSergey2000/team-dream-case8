"""
Модуль для анализа данных (EDA — Exploratory Data Analysis).
EDA — первый шаг в проекте, где мы изучаем данные, как смотрим фото перед поездкой. Мы проверяем, сколько строк, что в столбцах, есть ли пустые места (пропуски), как числа распределены (равномерно или кучками).
Здесь мы рисуем графики, считаем числа (среднее, skew — когда данные 'скучены' в одну сторону), проверяем на нормальность и ищем странные значения (выбросы).
Все графики сохраняем в папку, чтобы потом посмотреть.
Если ошибка, программа запишет в лог и не сломается полностью.
"""

import sys  # Это помощник для путей к файлам
from pathlib import Path  # Помогает с адресами файлов

ROOT_DIR = Path(__file__).parent.parent  # Находим главную папку проекта
sys.path.insert(0, str(ROOT_DIR))  # Добавляем путь, чтобы найти другие файлы
sys.path.insert(0, str(ROOT_DIR / 'src'))  # Добавляем путь к модулям

import pandas as pd  # Библиотека для таблиц
import numpy as np  # Для чисел и математики
import matplotlib.pyplot as plt  # Для рисования графиков
import seaborn as sns  # Красивые графики
from scipy import stats  # Для тестов, проверка на 'нормальность'
from configs.config import DATA_PATH, PLOTS_DIR, logger, TARGET, RANDOM_STATE  # Настройки

# Делаем графики красивыми: белый фон с сеткой, пастельные цвета
sns.set_style("whitegrid")  # Белый фон с линиями
sns.set_palette("husl")  # Мягкие цвета
plt.rcParams['figure.figsize'] = (12, 8)  # Размер графика
plt.rcParams['font.size'] = 12  # Крупный рифт
np.random.seed(RANDOM_STATE)  # Фиксируем случайность, чтобы результаты были одинаковыми каждый раз

def load_data():
    """
    Что делает: Загружает данные из файла CSV
    Возвращает: Таблицу с данными или ошибку, если файл не найден.
    Ошибки: Если файл не найден, программа скажет, где искать, и остановится.
    """
    try:
        df = pd.read_csv(DATA_PATH)  # Читаем файл
        logger.info(f"Данные загружены успешно. Размер датасета: {df.shape}")  # Записываем, сколько строк и столбцов
        return df  # Возвращаем таблицу.
    except FileNotFoundError:
        logger.error(f"Файл {DATA_PATH} не найден. Убедитесь, что financials.csv в директории data/.")  # Если файл потерялся, подсказка где искать
        raise  # Поднимаем ошибку, чтобы остановиться.
    except pd.errors.ParserError:
        logger.error("Ошибка чтения CSV. Проверьте, правильный ли формат файла, как текст без лишних символов.")  # Если файл сломан
        raise
    except Exception as e:
        logger.error(f"Неизвестная ошибка при загрузке данных: {e}")  # Любая другая проблема
        raise

def basic_info(df):
    """
    Что делает: Показывает основную информацию о данных: сколько строк, что в столбцах, статистика чисел, где пропуски
    Рисует карту пропусков
    """
    try:
        logger.info("Базовая информация о датасете:")  # Записываем начало
        df_info = df.info()  # Показывает типы и количество
        logger.info(df_info)

        logger.info("Статистическое описание числовых признаков:")
        describe_num = df.describe()  # Считает числа
        logger.info(describe_num)

        # Анализ пропусков
        missing = df.isnull().sum()  # Сумма пустых в каждом столбце
        missing_pct = 100 * missing / len(df)  # Процент пустых
        missing_df = pd.DataFrame({'count': missing, 'percent': missing_pct}).sort_values(by='count', ascending=False)  # Таблица, отсортированная от худшего
        logger.info("Пропуски в данных (топ):")  # Показываем только где есть пропуски
        logger.info(missing_df[missing_df['count'] > 0])

        # Карта пропусков
        plt.figure()
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)  # Тепловая карта, viridis — цвета от синего (полный) к желтому (пустой)
        plt.title('Карта пропусков в датасете', fontsize=16, fontweight='bold')  # Заголовок большой и жирный
        plt.xlabel('Признаки', fontsize=14)  # Подписи осей
        plt.ylabel('Записи', fontsize=14)
        plt.tight_layout()  # Чтобы всё поместилось.
        save_path = PLOTS_DIR / 'missing_values_heatmap.png'  # Путь для сохранения
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Сохраняем
        plt.close()  # Закрываем рисунок, чтобы не висел в памяти
        logger.info(f"Карта пропусков сохранена: {save_path}")
    except Exception as e:
        logger.error(f"Ошибка в базовой информации: {e}. Проверьте данные на правильность.")

def analyze_categorical(df):
    """
    Что делает: Смотрит на категории, считает, сколько в каждой, уникальные.
    Рисует столбцы для секторов
    """
    categorical_cols = ['Symbol', 'Name', 'Sector']  # Список столбцов с категориями
    for col in categorical_cols:  # Для каждого столбца
        if col in df.columns:  # Проверяем, есть ли столбец
            try:
                logger.info(f"Анализ категории '{col}':")  # Записываем начало
                value_counts = df[col].value_counts()  # Считаем, сколько раз каждое значение
                logger.info(value_counts)
                logger.info(f"Уникальных значений: {df[col].nunique()}")  # Сколько разных

                # График для сектора
                if col == 'Sector':
                    plt.figure()
                    sns.countplot(y=col, data=df, order=value_counts.index, edgecolor='black', linewidth=1.5)  # Горизонтальные столбики, order — от большего
                    plt.title('Распределение компаний по секторам', fontsize=16, fontweight='bold')  # Заголовок
                    plt.xlabel('Количество компаний', fontsize=14)  # Подписи
                    plt.ylabel('Сектор', fontsize=14)
                    plt.tight_layout()
                    save_path = PLOTS_DIR / 'sector_distribution.png'
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"График по секторам сохранён: {save_path}")
            except Exception as e:
                logger.error(f"Ошибка в анализе категории {col}: {e}. Проверьте столбец.")

def analyze_numerical(df):
    """
    Что делает: Смотрит на числа: как они распределены, skew (скучены ли), kurtosis (острые ли пики), тест на нормальность
    Рисует графики, log-графики если скучены, пары графиков
    """
    numerical_cols = [
        'Price', 'Price/Earnings', 'Dividend Yield', 'Earnings/Share',
        '52 Week Low', '52 Week High', 'Market Cap', 'EBITDA',
        'Price/Sales', 'Price/Book'
    ]  # Список числовых столбцов
    for col in numerical_cols:  # Для каждого
        if col in df.columns:
            try:
                logger.info(f"Анализ числа '{col}':")
                stats_col = df[col].describe()  # Среднее, мин/макс и т.д
                skew = df[col].skew()  # Skew — скученность
                kurt = df[col].kurtosis()  # Kurtosis — острота пика
                logger.info(stats_col)
                logger.info(f"Skew: {skew:.2f}, Kurtosis: {kurt:.2f}")

                # Тест на нормальность (Shapiro, для малого количества, p>0.05 — нормально)
                if len(df[col].dropna()) < 5000:  # Для больших данных тест не точный
                    shapiro_stat, shapiro_p = stats.shapiro(df[col].dropna())  # Stats — как проверка формы
                    logger.info(f"Shapiro test: stat={shapiro_stat:.4f}, p-value={shapiro_p:.4f} (нормальность: p > 0.05)")

                # График распределения (гистограмма с кривой)
                plt.figure()
                sns.histplot(df[col], kde=True)  # Histplot — столбики, kde — гладкая кривая
                plt.title(f'Распределение {col}', fontsize=16, fontweight='bold')
                save_path = PLOTS_DIR / f'distribution_{col.lower().replace("/", "_")}.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"График распределения для {col} сохранен: {save_path}")

                # Log-график, если skew >1 (делаем ровнее)
                if skew > 1:
                    df_log = np.log1p(df[col].clip(lower=0))  # Clip — обрезаем отрицательные, log1p — логарифм +1 для 0
                    plt.figure()
                    sns.histplot(df_log, kde=True)
                    plt.title(f'Log-распределение {col}', fontsize=16, fontweight='bold')
                    save_path_log = PLOTS_DIR / f'log_distribution_{col.lower().replace("/", "_")}.png'
                    plt.savefig(save_path_log, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Log-график для {col} сохранен: {save_path_log}")
            except Exception as e:
                logger.error(f"Ошибка в анализе числа {col}: {e}. Проверьте данные.")

    # Пары графиков для ключевых чисел (чтобы увидеть связи)
    try:
        key_num_cols = ['Price', 'EBITDA', 'Price/Earnings', TARGET]  # Ключевые, из корреляций
        sns.pairplot(df[key_num_cols])  # Pairplot — сетка графиков
        save_path = PLOTS_DIR / 'numerical_pairplot.png'
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Пары графиков для ключевых чисел сохранены: {save_path}")
    except Exception as e:
        logger.error(f"Ошибка в парах графиков: {e}")

def correlation_analysis(df):
    """
    Что делает: Считает связи между числами (Pearson — для прямых, Spearman — для кривых).
    Рисует карты связей, показывает топ с целью
    """
    try:
        numerical_df = df.select_dtypes(include=np.number)  # Только числа
        corr_pearson = numerical_df.corr(method='pearson')  # Pearson — для нормальных распределений
        corr_spearman = numerical_df.corr(method='spearman')  # Spearman — для кривых, как ранги

        # Карта Pearson (abs, цвета от холодного к теплому)
        plt.figure()
        sns.heatmap(np.abs(corr_pearson), annot=True, cmap='coolwarm', vmin=0, vmax=1)  # Heatmap — тепловая карта
        plt.title('Карта связей Пирсона (abs)', fontsize=16, fontweight='bold')
        save_path = PLOTS_DIR / 'correlation_pearson_abs.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Карта связей Пирсона (abs) сохранена: {save_path}")

        # Карта Spearman
        plt.figure()
        sns.heatmap(np.abs(corr_spearman), annot=True, cmap='coolwarm', vmin=0, vmax=1)
        plt.title('Карта связей Спирмена (abs)', fontsize=16, fontweight='bold')
        save_path = PLOTS_DIR / 'correlation_spearman_abs.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Карта связей Спирмена (abs) сохранена: {save_path}")

        # Топ связей с целью (для понимания, что важно)
        logger.info("Топ связей Пирсона с Market Cap:")
        logger.info(corr_pearson[TARGET].sort_values(ascending=False))
        logger.info("Топ связей Спирмена с Market Cap:")
        logger.info(corr_spearman[TARGET].sort_values(ascending=False))
    except Exception as e:
        logger.error(f"Ошибка в анализе связей: {e}. Проверьте числа в данных.")

def sector_analysis(df):
    """
    Что делает: Смотрит данные по группам (секторам): среднее, медиана, разброс, количество.
    Рисует коробки для цели и ключевого (P/E).
    Сектора — как отделы в компании. Мы считаем среднюю цену, рисуем коробки (boxplot — как ящик с усами, показывает среднее и разброс)
    """
    try:
        logger.info("Анализ по секторам:")
        sector_stats = df.groupby('Sector').agg({  # Группируем по секторам, как сортировка по папкам
            TARGET: ['mean', 'median', 'std', 'count'],  # Среднее — общая середина, медиана — середина без крайностей
            'EBITDA': ['mean', 'median']
        }).sort_values(by=(TARGET, 'mean'), ascending=False)  # Сортируем от дорогих
        logger.info("Статистика по секторам (средние, медиана, std, count):")
        logger.info(sector_stats)

        # Коробка для Market Cap по секторам.
        plt.figure()
        sns.boxplot(x='Sector', y=TARGET, data=df, notch=True, showfliers=False, linewidth=1.5)  # Notch — вмятина для медианы
        plt.title(f'Распределение {TARGET} по секторам', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=12)  # Поворачиваем метки, чтобы не налезали
        plt.xlabel('Сектор', fontsize=14)
        plt.ylabel(TARGET, fontsize=14)
        plt.grid(alpha=0.3)  # Сетка слабая, как фон
        plt.tight_layout()
        save_path = PLOTS_DIR / 'boxplot_target_by_sector.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Коробка {TARGET} по секторам сохранена: {save_path}")

        # Коробка для ключевого (P/E)
        key_fich = 'Price/Earnings'
        if key_fich in df.columns:
            plt.figure()
            sns.boxplot(x='Sector', y=key_fich, data=df, notch=True, showfliers=False, linewidth=1.5)
            plt.title(f'Распределение {key_fich} по секторам', fontsize=16, fontweight='bold')
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.xlabel('Сектор', fontsize=14)
            plt.ylabel(key_fich, fontsize=14)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            save_path = PLOTS_DIR / 'boxplot_pe_by_sector.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Коробка {key_fich} по секторам сохранена: {save_path}")
    except Exception as e:
        logger.error(f"Ошибка в анализе по секторам: {e}. Проверьте столбец 'Sector'.")

def outliers_analysis(df):
    """
    Что делает: Находит странные значения (выбросы) с Z-score (>3 — очень странные) и IQR (1.5*IQR — умеренные).
    Показывает количество, рисует коробки.
    Выбросы — это числа, которые сильно отличаются. Z-score — как расстояние от среднего, IQR — как ящик, где выбросы за пределами.
    Рекомендация: Обрезать в предобработке, чтобы модель не путалась.
    """
    numerical_cols = [
        'Price', 'Price/Earnings', 'Dividend Yield', 'Earnings/Share',
        '52 Week Low', '52 Week High', 'Market Cap', 'EBITDA',
        'Price/Sales', 'Price/Book'
    ]  # Числовые столбцы.
    logger.info("Анализ выбросов для числовых признаков:")
    outliers_summary = {}  # Словарь для результатов
    for col in numerical_cols:
        if col in df.columns:
            try:
                # Z-score: расстояние от среднего в отклонениях
                z_scores = np.abs(stats.zscore(df[col].dropna()))  # Abs — берём положительное
                z_outliers = (z_scores > 3).sum()  # Считаем >3

                # IQR: ящик, где пределы — 1.5 раза размах
                Q1 = df[col].quantile(0.25)  # Нижняя четверть
                Q3 = df[col].quantile(0.75)  # Верхняя
                IQR = Q3 - Q1  # Размах.
                iqr_outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()  # За пределами

                outliers_summary[col] = {'z_outliers': z_outliers, 'iqr_outliers': iqr_outliers}
                logger.info(f"{col}: Z-score outliers = {z_outliers}, IQR outliers = {iqr_outliers}")
            except Exception as e:
                logger.error(f"Ошибка в анализе выбросов для {col}: {e}.")

    # График коробок для выбросов (горизонтальный, чтобы метки поместились)
    try:
        plt.figure(figsize=(14, 10))
        sns.boxplot(data=df[numerical_cols], orient='h', notch=True, showfliers=True, linewidth=1.5)  # Orient='h' — горизонтально
        plt.title('Коробки для поиска выбросов в числах', fontsize=16, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        save_path = PLOTS_DIR / 'outliers_boxplots.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"График коробок для выбросов сохранён: {save_path}")
    except Exception as e:
        logger.error(f"Ошибка в графике выбросов: {e}.")

def feature_eng_preview(df):
    """
    Что делает: Показывает пример новых признаков (ratios), считает их связь с целью
    Новые признаки — это как новые вопросы из старых. Ratio — соотношение, как цена за кг. Мы проверяем, полезны ли они (связь >0.5 — хорошо)
    Рекомендация: Добавить в предобработку, если полезно
    """
    try:
        # Пример: соотношение EBITDA к цене (из сильной связи в анализе)
        if 'EBITDA' in df and 'Price' in df:
            df['EBITDA_Price_Ratio'] = df['EBITDA'] / df['Price'].clip(lower=1e-6)  # Clip — обрезаем, чтобы не делить на 0
            logger.info("Создан пример признака: EBITDA_Price_Ratio")

        # Смотрим связь с целью
        new_cols = ['EBITDA_Price_Ratio']
        for col in new_cols:
            if col in df:
                corr = df[col].corr(df[TARGET])  # Corr — сила связи
                logger.info(f"Связь {col} с {TARGET}: {corr:.4f}")

        logger.info("Рекомендация: Если связь сильная, добавьте в предобработку для лучших моделей.")
    except Exception as e:
        logger.error(f"Ошибка в примере новых признаков: {e}.")

def main():
    """
    Что делает: Запускает весь анализ
    Вызывает все функции по порядку
    Это главная функция: загружает данные и запускает все проверки
    """
    try:
        df = load_data()  # Загружаем данные
        basic_info(df)  # Базовая информация
        analyze_categorical(df)  # Категории
        analyze_numerical(df)  # Числа
        correlation_analysis(df)  # Связи
        sector_analysis(df)  # По группам
        outliers_analysis(df)  # Выбросы
        feature_eng_preview(df)  # Пример новых признаков
        logger.info("Анализ завершён успешно. Все графики и записи сохранены.")
    except Exception as e:
        logger.error(f"Ошибка в главном анализе: {e}. Проверьте начало.")

if __name__ == "__main__":
    main()  # Запускаем, если файл открыт напрямую