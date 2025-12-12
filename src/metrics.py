"""
Модуль для оценки моделей.
Это как проверка домашнего задания: мы смотрим, насколько модель хорошо предсказывает, считаем оценки (метрики) и выбираем лучшую
Для новичков: Метрики — это отметки модели: R^2 — процент правильных ответов, MAE — средняя ошибка, RMSE — ошибка с акцентом на большие промахи. Мы сравниваем модели, рисуем графики и пишем отчёт
Если ошибка, программа запишет в лог и не остановится полностью
"""

# Настройки, с добавлением RANDOM_STATE для cv
from configs.config import RESULTS_DIR, PLOTS_DIR, logger, RANDOM_STATE
# Добавлен импорт cross_validate для интеграции CV в метрики
from sklearn.model_selection import cross_validate, KFold
# Оценки
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns  # Для красивых графиков
import matplotlib.pyplot as plt  # Для графиков
import numpy as np  # Для чисел
import pandas as pd  # Для таблиц
import sys  # Пути
from pathlib import Path  # Для адресов файлов

ROOT_DIR = Path(__file__).parent.parent  # Главная папка
sys.path.insert(0, str(ROOT_DIR))  # Добавляем путь
sys.path.insert(0, str(ROOT_DIR / 'src'))  # К модулям


# Делаем графики красивыми: белый фон с сеткой, цвета husl
sns.set_style("whitegrid")  # Фон
sns.set_palette("husl")  # Цвета мягкие


def evaluate_models(models, metrics_dict, X_train, X_test, y_train_log, y_test_log, y_test_real, names_test):
    """
    Что делает: Проверяет все модели, считает оценки на log и реальных данных
    Параметры: модели, их оценки, данные
    Возвращает: Таблицу результатов
    Мы предсказываем на тесте, сравниваем с правдой, считаем ошибки. Bias correction — поправка, чтобы предсказания были точнее, как калибровка весов
    """
    results = []  # Список для результатов

    # KFold вместо StratifiedKFold для continuous target (регрессия), чтобы cross_validate работал без ошибки target type
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, pipe in models.items():  # Для каждой модели
        try:
            # Кросс-валидация на train для надежных средних метрик (интегрировано для фикса overfitting)
            scores = cross_validate(pipe, X_train, y_train_log, cv=cv, scoring=['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error'], n_jobs=-1)
            r2_log_cv = scores['test_r2'].mean()  # Средний R² из CV на log
            # Средний MAE из CV
            mae_log_cv = -scores['test_neg_mean_absolute_error'].mean()
            # Средний RMSE из CV
            rmse_log_cv = np.sqrt(-scores['test_neg_mean_squared_error'].mean())

            # Поправка на train, чтобы модель не 'врала'
            # Предсказания на обучении
            y_train_pred_log = pipe.predict(X_train)
            # Поправка
            bias_correction = np.exp(y_train_log.mean() - y_train_pred_log.mean())

            # Предсказания на тесте
            y_pred_log = pipe.predict(X_test)
            # Обратный log, чтобы вернуться к реальным числам
            y_pred_real = np.expm1(y_pred_log) * bias_correction

            # Оценки на log (на тесте, для сравнения с CV)
            r2_log = r2_score(y_test_log, y_pred_log)  # Процент правильного
            # Средняя ошибка
            mae_log = mean_absolute_error(y_test_log, y_pred_log)
            # Корень ошибки
            rmse_log = np.sqrt(mean_squared_error(y_test_log, y_pred_log))

            # Оценки на реальных (для понимания в деньгах)
            r2_real = r2_score(y_test_real, y_pred_real)
            # В миллиардах долларов
            mae_real = mean_absolute_error(y_test_real, y_pred_real) / 1e9
            rmse_real = np.sqrt(mean_squared_error(
                y_test_real, y_pred_real)) / 1e9

            # Время обучения
            train_time = metrics_dict[name].get('Train_Time', 0.0)

            results.append({  # Добавляем в список, включая CV-метрики для полноты
                'Модель': name,
                'R^2 (log CV)': round(r2_log_cv, 4),  # Средний из CV
                'MAE (log CV)': round(mae_log_cv, 2),
                'RMSE (log CV)': round(rmse_log_cv, 2),
                'R^2 (log)': round(r2_log, 4),
                'MAE (log)': round(mae_log, 2),
                'RMSE (log)': round(rmse_log, 2),
                'R^2 (real)': round(r2_real, 4),
                'MAE (real, млрд $)': round(mae_real, 2),
                'RMSE (real, млрд $)': round(rmse_real, 2),
                'Время обучения (с)': round(train_time, 2)
            })
            logger.info(
                f"Оценка {name}: R^2_real={r2_real:.4f}, MAE_real={mae_real:.2f}, R^2_log_CV={r2_log_cv:.4f}")
        except Exception as e:
            logger.error(f"Ошибка оценки {name}: {e}. Проверьте предсказания.")

    df_results = pd.DataFrame(results)  # Делаем таблицу
    if df_results.empty:
        logger.warning("Нет результатов. Модели не сработали?")
    else:
        # Сохраняем в файл.
        df_results.to_csv(RESULTS_DIR / 'models_comparison.csv', index=False)
        logger.info(
            "Таблица сравнения сохранена в results/models_comparison.csv")

    return df_results


def plot_comparison(df_results):
    """
    Что делает: Рисует графики сравнения моделей, как столбики отметок
    Параметры: таблица результатов
    Графики — визуальная проверка, как диаграмма, где выше столбик — лучше модель по метрике
    """
    try:
        # Увеличено для новых метрик CV
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        axes = axes.flatten()  # Делаем список
        # Добавлены CV-метрики
        metrics = ['R^2 (log CV)', 'MAE (log CV)', 'RMSE (log CV)',
                   'R^2 (real)', 'MAE (real, млрд $)', 'RMSE (real, млрд $)']
        for i, metric in enumerate(metrics):  # Для каждой метрики
            # Столбики
            sns.barplot(x='Модель', y=metric, data=df_results, ax=axes[i])
            axes[i].set_title(f'Сравнение по {metric}')  # Заголовок.
            # Фиксируем метки
            axes[i].set_xticks(range(len(df_results['Модель'])))
            # Поворачиваем, чтобы читалось
            axes[i].set_xticklabels(df_results['Модель'], rotation=45)
            axes[i].grid(alpha=0.3)  # Сетка слабая

        plt.tight_layout()  # Чтобы поместилось
        save_path = PLOTS_DIR / 'models_comparison_plot.png'
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"График сравнения сохранён: {save_path}")
    except Exception as e:
        logger.error(f"Ошибка рисунка сравнения: {e}. Проверьте таблицу.")


def select_best_model(df_results):
    """
    Что делает: Выбирает лучшую модель по R^2, если равно — по времени
    Параметры: таблица результатов
    Возвращает: Имя лучшей модели
    Лучшая — с наивысшей отметкой (R^2). Если ничья, выбираем быструю
    """
    if df_results.empty:
        logger.warning("Нет данных для выбора. Нет оценок?")
        return None

    # Сортируем: лучше R^2, меньше времени
    df_sorted = df_results.sort_values(
        by=['R^2 (real)', 'Время обучения (с)'], ascending=[False, True])
    best_name = df_sorted.iloc[0]['Модель']  # Первая в списке
    logger.info(
        f"Лучшая модель: {best_name} (R^2_real={df_sorted.iloc[0]['R^2 (real)']:.4f})")
    return best_name


def generate_report(df_results, best_name, shap_insights=''):
    """
    Что делает: Пишет отчёт в файл report.md
    Параметры: таблица, лучшая модель, insights от SHAP
    Отчёт — итог, с таблицей оценок, лучшей моделью
    """
    try:
        report_path = RESULTS_DIR / 'report.md'  # Путь к файлу
        # Пишем в файл, utf-8 для кириллицы
        with open(report_path, 'w', encoding='utf-8') as f:
            # Заголовок
            f.write("# Отчёт по моделям прогнозирования Market Cap\n\n")
            f.write("## Сравнение моделей\n")  # Раздел
            # Таблица в формате markdown, как список
            f.write(df_results.to_markdown(index=False))
            f.write("\n\n## Лучшая модель\n")  # Раздел
            f.write(f"{best_name}\n\n")
            f.write("## Insights из SHAP\n")  # Раздел
            # Insights или сообщение.
            f.write(shap_insights or "SHAP анализ не выполнен.\n")
            # Раздел рекомендаций отключён по запросу пользователя
            # Если нужно вернуть рекомендации — добавьте их вручную или включите параметр include_recommendations.

        logger.info(f"Отчёт создан: {report_path}")
    except Exception as e:
        logger.error(f"Ошибка создания отчёта: {e}. Проверьте папку.")


def main(models, metrics_dict, X_train, X_test, y_train_log, y_test_log, y_test_real, names_test, shap_insights=''):
    """
    Что делает: Главная функция оценки
    Параметры: модели, оценки, данные, insights.
    Возвращает: Имя лучшей модели
    Здесь мы проверяем всех, рисуем, выбираем победителя и пишем отчёт
    """
    logger.info("Начало оценки моделей.")
    df_results = evaluate_models(models, metrics_dict, X_train, X_test,
                                 # Считаем оценки
                                 y_train_log, y_test_log, y_test_real, names_test)
    plot_comparison(df_results)  # Рисуем
    best_name = select_best_model(df_results)  # Выбираем
    generate_report(df_results, best_name, shap_insights)  # Пишем отчёт
    logger.info("Оценка моделей завершена.")
    return best_name


if __name__ == '__main__':
    # Для теста: импортируйте данные, результаты и вызовите main(...)
    pass
