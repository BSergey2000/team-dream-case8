# Обучаем 4 модели и сохраняем их в /models

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from .config import MODELS_DIR

def train_all_models(X_train, X_test, y_train_log, y_test_log):
    models = {}

    print("Обучение моделей...")

    # 1. Линейная регрессия — базовая модель
    lr = LinearRegression()
    lr.fit(X_train, y_train_log)
    joblib.dump(lr, MODELS_DIR / "linear_regression.pkl")
    models["Linear Regression"] = lr

    # 2. Gradient Boosting из sklearn — часто лучший на малых данных
    gb = GradientBoostingRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        min_samples_leaf=3,
        random_state=42
    )
    gb.fit(X_train, y_train_log)
    joblib.dump(gb, MODELS_DIR / "gradient_boosting.pkl")
    models["Gradient Boosting"] = gb

    # 3. XGBoost — популярный бустинг
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train_log)
    joblib.dump(xgb_model, MODELS_DIR / "xgboost.pkl")
    models["XGBoost"] = xgb_model

    # 4. LightGBM — быстрый и мощный
    lgb_model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=7,
        num_leaves=64,
        min_child_samples=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train_log,
                  eval_set=[(X_test, y_test_log)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    joblib.dump(lgb_model, MODELS_DIR / "lightgbm.pkl")
    models["LightGBM"] = lgb_model

    print("Все 4 модели обучены и сохранены в папке models/")
    return models