from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error
import numpy as np


def train_models(X, y):
    print("\nОбучение моделей + кросс-валидация (5-fold) + hold-out 20%...")

    # Честный сплит: 80% — обучение, 20% — финальный тест
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )

    models = {
        'Linear Regression': LinearRegression(),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                random_state=42, tree_method='hist')
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    final_models = {}

    for name, model in models.items():
        print(f"→ {name} (CV на train-части)...")

        # R² по кросс-валидации
        cv_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')

        # MAE в млрд $ на исходной шкале
        mae_vals = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            mae_vals.append(mean_absolute_error(np.expm1(y_val), np.expm1(pred)) / 1e9)

        cv_mae_mean = np.mean(mae_vals)

        cv_results[name] = {
            'cv_r2_mean': cv_r2.mean(),
            'cv_r2_std': cv_r2.std(),
            'cv_mae_original_mean': cv_mae_mean
        }

        print(f"   R² = {cv_r2.mean():.4f} ± {cv_r2.std():.4f} | MAE ≈ {cv_mae_mean:.1f} млрд $")

        # Финальное обучение на всей train-части
        model.fit(X_train, y_train)
        final_models[name] = model

    # Возвращаем модели + ЧЕСТНУЮ тестовую выборку
    results = {}
    for name, model in final_models.items():
        results[name] = {
            'model': model,
            'X_test': X_test,
            'y_test': y_test
        }

    return results, cv_results, X_test, y_test