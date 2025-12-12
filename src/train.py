"""
Модуль для обучения моделей.
Здесь мы определяем модели, оптимизация, и проверяем результаты (метрики).
Модель — это программа, которая учится на данных. Мы используем разные модели (простые, как LinearRegression — прямая линия, или сложные, как LightGBM — дерево решений). Optuna — помощник, который пробует разные настройки, чтобы модель была лучше. Pipeline — конвейер, где данные готовятся и модель учится вместе.
Если ошибка, программа запишет в лог и попробует не остановиться
"""

# Конвейер подготовки (пакетный импорт для стабильности)
from src.preprocess import build_preprocess_pipeline
# Настройки
from configs.config import MODELS_DIR, RANDOM_STATE, logger
import lightgbm as lgb  # Быстрая модель для больших данных
import xgboost as xgb  # Сильная модель для деревьев
from sklearn.pipeline import Pipeline  # Конвейер шагов
# Оценки
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# Для проверки на частях
from sklearn.model_selection import cross_validate, KFold
# Добавлен VotingRegressor для фикса ошибки с Ensemble
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
# Простая модель, как прямая линия
from sklearn.linear_model import LinearRegression
import optuna  # Для поиска лучших настроек
import joblib  # Для сохранения моделей
import numpy as np  # Для чисел
import time  # Для измерения времени
import sys  # Пути
from pathlib import Path  # Для адресов файлов

ROOT_DIR = Path(__file__).parent.parent  # Главная папка
sys.path.insert(0, str(ROOT_DIR))  # Добавляем путь
sys.path.insert(0, str(ROOT_DIR / 'src'))  # К модулям


# Делаем тихо, чтобы не было лишних слов в логе
optuna.logging.set_verbosity(optuna.logging.WARNING)
xgb.set_config(verbosity=0)


def define_models():
    """
    Что делает: Создаёт список моделей для обучения.
    Возвращает: Словарь с моделями.
    Модели — LinearRegression — простой, как 2+2. GradientBoosting — учит на ошибках. Мы добавляем команду (ensemble), где модели голосуют вместе.
    """
    try:
        models = {
            'LinearRegression': LinearRegression(),  # Простая модель
            # Учится шаг за шагом
            'GradientBoosting': GradientBoostingRegressor(random_state=RANDOM_STATE),
            # Быстрая и сильная
            'XGBoost': xgb.XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
            # Ещё быстрее
            'LightGBM': lgb.LGBMRegressor(random_state=RANDOM_STATE, verbose=-1),
        }
        # Команда моделей — голосуют за ответ
        models['Ensemble'] = VotingRegressor(estimators=[
            ('gb', models['GradientBoosting']),
            ('xgb', models['XGBoost']),
            ('lgbm', models['LightGBM'])
        ])
        logger.info("Базовые модели созданы: {}".format(list(models.keys())))
        return models
    except Exception as e:
        logger.error(f"Ошибка создания моделей: {e}. Проверьте библиотеки.")
        raise


def optimize_model(model_name, model, X_train, y_train_log, preprocessor=None):
    """
    Что делает: Подбирает лучшие настройки для модели с Optuna
    Параметры: Имя, модель, данные
    Возвращает: Лучшие параметры или пусто при ошибке
    Optuna — как проба разных рецептов, чтобы найти вкусный. Мы пробуем числа (параметры), учим модель, смотрим оценку
    """
    def objective(trial):  # Функция пробы
        params = {}  # Настройки
        if model_name == 'GradientBoosting':
            params = {
                # Сколько шагов
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20)
            }
        elif model_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0)
            }
        elif model_name == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100)
            }
        elif model_name == 'Ensemble':
            # Весы для GB, XGB, LGBM
            weights = [trial.suggest_float(
                f'w{i}', 0.1, 1.0) for i in range(3)]
            # Создаём локальную переменную ensemble_model, чтобы не затенять внешнюю переменную
            ensemble_model = VotingRegressor(estimators=[('gb', GradientBoostingRegressor(random_state=RANDOM_STATE)),
                                                         ('xgb', xgb.XGBRegressor(
                                                             random_state=RANDOM_STATE)),
                                                         ('lgbm', lgb.LGBMRegressor(random_state=RANDOM_STATE))],
                                             weights=weights)
            # Определение pipe внутри objective для Ensemble, используя переданный preprocessor, чтобы не пересобирать его многократно
            used_preprocessor = preprocessor if preprocessor is not None else build_preprocess_pipeline()
            pipe = Pipeline(
                [('preprocess', used_preprocessor), ('model', ensemble_model)])
            cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            scores = cross_validate(
                pipe, X_train, y_train_log, cv=cv, scoring='r2', n_jobs=-1)
            return scores['test_score'].mean()

        # Определение pipe внутри objective для остальных моделей, чтобы избежать NameError и обеспечить работу cross_validate
        used_preprocessor = preprocessor if preprocessor is not None else build_preprocess_pipeline()
        pipe = Pipeline([('preprocess', used_preprocessor),
                         ('model', model.set_params(**params))])

        # Проверка на 5 частях (CV)
        # KFold — делит данные на 5 частей
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        # Учим и проверяем
        scores = cross_validate(
            pipe, X_train, y_train_log, cv=cv, scoring='r2', n_jobs=-1)
        return scores['test_score'].mean()  # Средняя оценка

    try:
        # Для простых моделей без гиперпараметров (например LinearRegression) пропускаем Optuna
        if model_name == 'LinearRegression':
            return {}

        # Создаём исследование Optuna: direction='maximize' означает
        # что мы хотим максимизировать метрику R^2 в кросс-валидации
        study = optuna.create_study(direction='maximize')
        # Запускаем оптимизацию. n_trials задаёт число проверок гиперпараметров
        # n_jobs=-1 позволяет параллельно исполнять несколько испытаний (используйте с осторожностью)
        study.optimize(objective, n_trials=10, n_jobs=-1)  # 10 попыток
        best_params = study.best_params  # Лучшие настройки
        logger.info(
            f"Оптимизация {model_name} завершена: best_params={best_params}, best_R^2={study.best_value:.4f}")
        return best_params
    except Exception as e:
        logger.error(
            f"Ошибка подбора настроек {model_name}: {e}. Возможно, данные не подходят.")
        return {}


def train_model(model_name, model, best_params, X_train, y_train_log, X_test, y_test_log, preprocessor=None):
    """
    Что делает: Учит модель с лучшими настройками, считает оценки
    Параметры: Имя, модель, настройки, данные
    Возвращает: Обученный конвейер, оценки или ничего при ошибке
    Метрики — отметки: R^2 — процент правильного, MAE — средняя ошибка.
    """
    start_time = time.time()  # Засекаем начало
    # Для Ensemble best_params содержит веса (w0, w1, w2). VotingRegressor.set_params не принимает такие имена,
    # поэтому для Ensemble нужно явно собрать VotingRegressor с найденными весами
    if model_name == 'Ensemble':
        try:
            # Ожидаем параметры в формате {'w0': ..., 'w1': ..., 'w2': ...}
            weights = [v for k, v in sorted(
                best_params.items())] if best_params else None
            # Создаём новые базовые оценки с фиксированным random_state
            estimators = [
                ('gb', GradientBoostingRegressor(random_state=RANDOM_STATE)),
                ('xgb', xgb.XGBRegressor(random_state=RANDOM_STATE, verbosity=0)),
                ('lgbm', lgb.LGBMRegressor(random_state=RANDOM_STATE, verbose=-1))
            ]
            if weights:
                model_instance = VotingRegressor(
                    estimators=estimators, weights=weights)
            else:
                model_instance = VotingRegressor(estimators=estimators)

            used_preprocessor = preprocessor if preprocessor is not None else build_preprocess_pipeline()
            pipe = Pipeline([
                ('preprocess', used_preprocessor),  # Подготовка.
                ('model', model_instance)  # Модель ансамбля.
            ])
        except Exception as e:
            logger.error(
                f"Ошибка при создании Ensemble с лучшими параметрами: {e}")
            return None, {}
    else:
        used_preprocessor = preprocessor if preprocessor is not None else build_preprocess_pipeline()
        pipe = Pipeline([
            ('preprocess', used_preprocessor),  # Подготовка
            ('model', model.set_params(**best_params))  # Модель
        ])

    try:
        pipe.fit(X_train, y_train_log)  # Учим
        train_time = time.time() - start_time  # Время урока

        y_pred_log = pipe.predict(X_test)  # Предсказания

        r2 = r2_score(y_test_log, y_pred_log)  # Процент правильного
        mae = mean_absolute_error(y_test_log, y_pred_log)  # Средняя ошибка
        # Корень ошибки
        rmse = np.sqrt(mean_squared_error(y_test_log, y_pred_log))

        # Словарь оценок
        metrics = {'R2': r2, 'MAE': mae,
                   'RMSE': rmse, 'Train_Time': train_time}
        logger.info(f"{model_name} обучена: метрики={metrics}")

        save_path = MODELS_DIR / \
            f"{model_name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(pipe, save_path)  # Сохраняем, как файл
        logger.info(f"Модель сохранена: {save_path}")

        return pipe, metrics
    except Exception as e:
        logger.error(f"Ошибка обучения {model_name}: {e}. Проверьте данные.")
        return None, {}


def main(X_train, X_test, y_train_log, y_test_log, preprocessor=None):
    """
    Что делает: Главная функция, запускает обучение всех моделей
    Параметры: Данные из подготовки
    Возвращает: Словарь с моделями и оценками
    Вызывает модели, учит их, собирает результаты.
    """
    logger.info("Начало обучения моделей.")
    models = define_models()  # Создаём список.
    trained_models = {}  # Словарь готовых.
    all_metrics = {}  # Оценки.

    for name, model in models.items():  # Для каждой модели.
        # Подбираем настройки.
        best_params = optimize_model(
            name, model, X_train, y_train_log, preprocessor=preprocessor)
        trained_model, metrics = train_model(
            # Учим.
            name, model, best_params, X_train, y_train_log, X_test, y_test_log, preprocessor=preprocessor)
        if trained_model:
            trained_models[name] = trained_model
            all_metrics[name] = metrics

    logger.info("Обучение моделей завершено.")
    return {'models': trained_models, 'metrics': all_metrics}


if __name__ == '__main__':
    # Для теста: импортируйте данные из подготовки и вызовите main(...).
    pass
