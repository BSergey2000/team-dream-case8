"""
Модуль для анализа SHAP
Показывает, какие признаки важны для модели и почему она ошибается.
SHAP — способ понять, почему модель предсказала именно так, как 'рентген' предсказаний. Мы рисуем графики (рой пчёл для важности, водопад для ошибок), считаем влияние и пишем отчёт
Если ошибка, программа запишет в лог и не остановится полностью
"""

# Для обновления отчёта (пакетный импорт).
from src.metrics import generate_report
from configs.config import MODELS_DIR, PLOTS_DIR, logger  # Настройки
import warnings
import joblib  # Для загрузки моделей
import numpy as np  # Для чисел
import matplotlib.pyplot as plt  # Для рисунков
import shap  # Библиотека для анализа влияния
import sys  # Пути
from pathlib import Path  # Для адресов

ROOT_DIR = Path(__file__).parent.parent  # Главная папка
sys.path.insert(0, str(ROOT_DIR))  # Добавляем путь
sys.path.insert(0, str(ROOT_DIR / 'src'))  # К модулям


# Делаем графики большими и читаемыми
plt.rcParams['figure.figsize'] = (12, 8)  # Размер
plt.rcParams['font.size'] = 12  # Шрифт


def load_best_model(best_name):
    """
    Что делает: Загружает лучшую модель из файла
    Параметры: Имя модели
    Возвращает: Модель или ничего, если не нашлась
    """
    name_to_file = {
        'LinearRegression': 'linearregression_model.pkl',
        'GradientBoosting': 'gradientboosting_model.pkl',
        'XGBoost': 'xgboost_model.pkl',
        'LightGBM': 'lightgbm_model.pkl',
        'Ensemble': 'ensemble_model.pkl'
    }  # Словарь имён файлов
    try:
        # Пробуем явно по словарю, иначе формируем имя файла из имени модели
        # Находим файл по имени
        file_name = name_to_file.get(
            best_name, f"{best_name.lower().replace(' ', '_')}_model.pkl")
        pipe = joblib.load(MODELS_DIR / file_name)  # Загружаем
        logger.info(f"Загружена лучшая модель: {best_name} ({file_name})")
        return pipe
    except KeyError:
        logger.error(f"Имя модели неизвестно: {best_name}. Проверьте список.")
    except FileNotFoundError:
        logger.error(f"Файл модели не найден: {file_name}. Проверьте папку.")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}. Возможно, файл сломан.")
    return None


def compute_shap_values(pipe, X_train, X_test):
    """
    Что делает: Считает значения SHAP для теста, как меру влияния каждого признака
    Параметры: Конвейер, данные
    Возвращает: Значения SHAP, объяснитель, трансформированные данные и оригинальные имена признаков
    SHAP — показывает, насколько каждый 'вопрос' (признак) повлиял на ответ модели
    """
    try:
        preprocessor = pipe.named_steps['preprocess']  # Часть для подготовки
        model = pipe.named_steps['model']  # Сама модель

        # Сначала применяем FunctionTransformer для добавления производных признаков
        ft = preprocessor.named_steps.get('feature_engineer')
        if ft is not None:
            X_train_after_ft = ft.transform(X_train)
            X_test_after_ft = ft.transform(X_test)
        else:
            X_train_after_ft = X_train.copy()
            X_test_after_ft = X_test.copy()

        # Получаем имена признаков после FunctionTransformer (до ColumnTransformer)
        # Это будут оригинальные имена с добавленными производными признаками
        if hasattr(X_train_after_ft, 'columns'):
            original_feature_names = list(X_train_after_ft.columns)
        else:
            original_feature_names = None

        # Подготовленные данные для обучения через ColumnTransformer
        X_train_trans = preprocessor.transform(X_train)

        # Подготовленные для теста
        X_test_trans = preprocessor.transform(X_test)

        # Если модель — VotingRegressor или другой ансамбль, SHAP может не поддерживать напрямую
        # В этом случае используем KernelExplainer как fallback
        model_to_explain = model
        if hasattr(model, 'estimators_'):
            # Это VotingRegressor или подобный — попробуем использовать первый базовый estimator
            logger.warning(
                f"Модель {type(model).__name__} может не поддерживаться SHAP напрямую. Используем KernelExplainer.")
            # Для VotingRegressor берём первый estimator, или используем KernelExplainer на весь ансамбль
            # Здесь мы просто логируем и попытаемся KernelExplainer, который работает для любых моделей
            use_kernel_explainer = True
        else:
            use_kernel_explainer = False

        # Попытка создать explainer и посчитать shap. TreeExplainer в некоторых версиях
        # делает строгую проверку аддитивности; если она падает, повторим с отключённой проверкой или KernelExplainer
        try:
            if use_kernel_explainer:
                # KernelExplainer работает с любыми моделями (медленнее, но универсален)
                explainer = shap.KernelExplainer(
                    model.predict, shap.sample(X_train_trans, min(100, len(X_train_trans))))
            else:
                explainer = shap.Explainer(model, X_train_trans)
            shap_values = explainer(X_test_trans)
        except Exception as e:
            msg = str(e)
            # Если ошибка связана с проверкой аддитивности или типом модели,
            # пробуем KernelExplainer как универсальный fallback
            if 'Additivity check failed' in msg or 'additivity' in msg.lower() or 'not callable' in msg.lower():
                logger.warning(
                    f"Ошибка при создании explainer: {msg[:100]}... Используем KernelExplainer.")
                try:
                    # KernelExplainer — медленнее, но работает с любыми моделями (включая ансамбли)
                    explainer = shap.KernelExplainer(
                        model.predict, shap.sample(X_train_trans, min(100, len(X_train_trans))))
                    shap_values = explainer(shap.sample(
                        X_test_trans, min(50, len(X_test_trans))))
                except Exception as e2:
                    logger.error(
                        f"KernelExplainer также упал: {e2}. Ансамбль может быть несовместим с SHAP.")
                    raise
            else:
                logger.error(
                    f"Ошибка при создании explainer или расчёте SHAP: {e}")
                raise

        logger.info("SHAP-значения посчитаны успешно.")
        return shap_values, explainer, X_test_trans, original_feature_names
    except Exception as e:
        logger.error(f"Ошибка подсчёта SHAP: {e}. Проверьте модель.")
        return None, None, None, None


def plot_shap_summary(shap_values, X_test_trans, feature_names, preprocessor=None):
    """
    Параметры: Значения SHAP, данные, имена, препроцессор (опционально)
    График — где точки выше, признак важнее
    """
    try:
        # Если feature_names None или пусто, генерируем generic имена
        if feature_names is None or (isinstance(feature_names, list) and len(feature_names) == 0):
            n_features = X_test_trans.shape[1]
            actual_feature_names = [f'feature_{i}' for i in range(n_features)]
            logger.debug(
                f"Использованы автоматические имена признаков: {len(actual_feature_names)} шт.")
        else:
            actual_feature_names = feature_names

        # Рисуем. Некоторые версии shap/np могут генерировать FutureWarning
        # о глобальном RNG — подавляем конкретно эту ворнинг-линию локально
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            try:
                # Подход 1: Стандартный вызов (только shap_values, без X_test_trans)
                shap.summary_plot(
                    shap_values, feature_names=actual_feature_names, show=False)
            except Exception as e1:
                logger.debug(f"summary_plot подход 1 упал: {str(e1)[:60]}")
                try:
                    # Подход 2: С X_test_trans (может не работать с KernelExplainer)
                    vals = getattr(shap_values, 'values', shap_values)
                    shap.summary_plot(vals, X_test_trans,
                                      feature_names=actual_feature_names, show=False)
                except Exception as e2:
                    logger.debug(f"summary_plot подход 2 упал: {str(e2)[:60]}")
                    # Подход 3: Без feature_names
                    shap.summary_plot(shap_values, show=False)
        plt.title("Summary SHAP: Важность признаков", fontsize=16)
        save_path = PLOTS_DIR / 'shap_summary_plot.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"График summary сохранён: {save_path}")

        # ПРИМЕЧАНИЕ: dependence_plot может не работать с KernelExplainer из-за несовпадения размеров
        # (KernelExplainer может возвращать подвыборку данных). Пропускаем его в этом случае
        # Если нужен dependence_plot, используйте TreeExplainer с древовидными моделями
        logger.debug(
            "dependence_plot пропущен для совместимости с KernelExplainer")

    except Exception as e:
        logger.error(f"Ошибка рисунка summary: {e}. Проверьте значения.")


def plot_error_waterfalls(shap_values, explainer, X_test_trans, y_test_real, y_pred_real, names_test):
    """
    Что делает: Рисует водопады для 3 больших ошибок, как разбор полётов
    Параметры: Значения SHAP, данные
    Для новичков: Водопад — показывает, как признаки 'толкнули' предсказание в ошибку, как цепочка событий
    """
    try:
        errors = np.abs(y_pred_real - y_test_real)  # Ошибки, abs — без знака
        top_errors_idx = np.argsort(errors)[-3:]  # Топ-3 больших
        for i, idx in enumerate(top_errors_idx):  # Для каждой ошибки
            real_val = y_test_real.iloc[idx] / 1e9  # В млрд
            pred_val = y_pred_real[idx] / 1e9
            # Процент ошибки.
            error_pct = (pred_val - real_val) / real_val * \
                100 if real_val != 0 else 0

            # Рисуем водопад (убираем feature_names — в зависимости от версии shap этот аргумент может отсутствовать)
            shap.plots.waterfall(shap_values[idx], max_display=10, show=False)

            plt.title(
                f"Водопад {i}: {names_test.iloc[idx]} (Реальное: {real_val:.1f} млрд, Предсказ: {pred_val:.1f} млрд, Ошибка: {error_pct:.1f}%)")
            # Безопасное имя для файла.
            safe_name = names_test.iloc[idx].replace(' ', '_').replace('/', '_')
            save_path = PLOTS_DIR / f'shap_waterfall_error_{i}_{safe_name}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(
                f"График водопада для ошибки {i} сохранён: {save_path}")
    except Exception as e:
        logger.error(
            f"Ошибка в plot_error_waterfalls: {e}. Проверьте данные или индексы.")


def generate_shap_insights(shap_values, feature_names):
    """
    Что делает: Создаёт текст с выводами из SHAP, как список важных фактов
    Параметры: Значения SHAP, имена
    Возвращает: Текст с выводами
    Insights — ключевые находки, как "EBITDA сильно влияет, потому что корреляция высокая из анализа"
    """
    try:
        # Среднее влияние каждого признака по абсолютным значениям
        # shap_values может быть объектом Explanation или numpy-массивом, поэтому используем .values если есть
        importances = np.abs(
            getattr(shap_values, 'values', shap_values)).mean(axis=0)
        top_indices = np.argsort(importances)[-5:]  # Топ-5 важных
        insights = "Ключевые находки из SHAP-анализа:\n"  # Начало текста
        for idx in top_indices[::-1]:  # От самого важного
            feat = feature_names[idx]  # Имя признака
            imp = importances[idx]  # Влияние
            insights += f"- {feat}: среднее влияние {imp:.4f} (из анализа связей, e.g., EBITDA с ценой компании).\n"
        logger.info("Находки SHAP созданы.")
        return insights
    except Exception as e:
        logger.error(f"Ошибка создания находок: {e}. Проверьте значения.")
        return "Находки SHAP не созданы из-за ошибки."


def main(best_name, X_train, X_test, y_train_log, y_test_log, y_test_real, names_test, feature_names, df_results):
    """
    Что делает: Главная функция для SHAP, как сбор отчёта детектива
    Параметры: Имя лучшей, данные, результаты
    Для новичков: Загружаем модель, считаем влияние, рисуем графики, пишем выводы и добавляем в отчёт
    """
    logger.info("Начало анализа SHAP.")
    pipe = load_best_model(best_name)  # Загружаем лучшую
    if pipe is None:
        return  # Если не загрузилась, останавливаемся

    # Предсказания для проверки ошибок
    y_train_pred_log = pipe.predict(X_train)  # На обучении
    # Поправка смещения между средними лог-предсказаниями (bias correction)
    # чтобы вернуть предсказания в реальный масштаб корректно
    bias_correction = np.exp(y_train_log.mean() - y_train_pred_log.mean())
    y_pred_log = pipe.predict(X_test)  # На тесте
    y_pred_real = np.expm1(y_pred_log) * bias_correction  # Обратный log

    # Считаем SHAP
    shap_values, explainer, X_test_trans, ct_feature_names = compute_shap_values(
        pipe, X_train, X_test)
    if shap_values is None:
        return

    # Используем имена признаков из ColumnTransformer, если доступны; иначе используем переданные
    actual_feature_names = ct_feature_names if ct_feature_names is not None else feature_names

    # Графики (передаём препроцессор для получения корректных имён признаков после ColumnTransformer)
    preprocessor = pipe.named_steps.get('preprocess')
    plot_shap_summary(shap_values, X_test_trans,
                      actual_feature_names, preprocessor=preprocessor)
    plot_error_waterfalls(shap_values, explainer, X_test_trans,
                          y_test_real, y_pred_real, names_test)

    # Выводы и отчёт
    insights = generate_shap_insights(shap_values, actual_feature_names)
    try:
        generate_report(df_results, best_name, insights)
    except Exception as e:
        # Чаще всего ошибка — отсутствие дополнительных опциональных зависимостей (например, tabulate)
        err_msg = str(e)
        logger.error(f"Ошибка создания отчёта: {err_msg}")
        # Если это отсутствие tabulate — даём понятную подсказку и проверяем папки с результатами
        try:
            import tabulate  # не убирать вверх
        except Exception:
            logger.error(
                "Отсутствует необязимая зависимость 'tabulate'. Установите её: pip install tabulate")

        # Логируем содержимое ключевых директорий, чтобы вы могли быстро проверить файлы
        try:
            plots = list(PLOTS_DIR.glob('*')) if PLOTS_DIR.exists() else []
            models = list(MODELS_DIR.glob('*')) if MODELS_DIR.exists() else []
            results_dir = Path.cwd() / 'results'
            results = list(results_dir.glob(
                '*')) if results_dir.exists() else []
            logger.info(f"Содержимое {PLOTS_DIR}: {[p.name for p in plots]}")
            logger.info(f"Содержимое {MODELS_DIR}: {[m.name for m in models]}")
            logger.info(f"Содержимое results/: {[r.name for r in results]}")
        except Exception as _:
            logger.warning(
                "Не удалось проверить содержимое директорий результатов.")

    logger.info("Анализ SHAP завершён.")


if __name__ == '__main__':
    # Для теста: импортируйте данные, результаты и вызовите main(...).
    pass
