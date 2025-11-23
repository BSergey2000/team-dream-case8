# Делает SHAP-анализ для лучшей модели: график №04 + №05 для топ-3 ошибок

import shap
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from .config import MODELS_DIR, PLOTS_DIR
from .data_preparation import load_and_prepare_data


def generate_shap_analysis(best_model_name: str = "Gradient Boosting"):
    print("\nГенерация SHAP-анализа...")

    # Какой файл соответствует названию модели
    name_to_file = {
        "Linear Regression": "linear_regression.pkl",
        "Gradient Boosting": "gradient_boosting.pkl",
        "XGBoost": "xgboost.pkl",
        "LightGBM": "lightgbm.pkl",
    }

    model = joblib.load(MODELS_DIR / name_to_file[best_model_name])
    print(f"   → Загружена модель: {best_model_name}")

    # Загружаем те же данные, что использовались везде
    data = load_and_prepare_data()
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train_log = data["y_train_log"]
    y_test_real = data["y_test_real"]
    names_test = data["names_test"]

    # Исправляем смещение предсказаний (как в evaluation)
    train_pred = model.predict(X_train)
    correction = np.exp(y_train_log.mean() - train_pred.mean())
    test_pred_log = model.predict(X_test)
    test_pred = np.expm1(test_pred_log) * correction

    # Находим самые большие ошибки
    errors = np.abs(y_test_real - test_pred)
    top_n = min(3, len(errors))
    top_pos = np.argsort(errors.values)[-top_n:][::-1]  # позиции в тесте

    # Считаем SHAP-значения
    print("   → Считаем SHAP...")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    # График №04 — общая важность признаков
    plt.figure()
    shap.summary_plot(shap_values, X_test, max_display=12, show=False)
    plt.savefig(PLOTS_DIR / "04_shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("   → 04_shap_summary.png")

    # Графики №05 — почему модель ошиблась на конкретных компаниях
    for i, pos in enumerate(top_pos, 1):
        name = names_test.iloc[pos]
        real = y_test_real.iloc[pos] / 1e9
        pred = test_pred[pos] / 1e9
        error_pct = errors.iloc[pos] / y_test_real.iloc[pos] * 100

        plt.figure()
        shap.waterfall_plot(
            shap.Explanation(
                base_values=explainer.expected_value,
                values=shap_values.values[pos],
                data=X_test.iloc[pos],
                feature_names=X_test.columns
            ),
            max_display=10,
            show=False
        )
        plt.title(f"{i}. {name}\nРеал: {real:.1f} млрд | Прогноз: {pred:.1f} млрд | Ошибка: {error_pct:.1f}%")
        safe_name = "".join(c for c in str(name).split()[0] if c.isalnum())
        plt.savefig(PLOTS_DIR / f"05_shap_waterfall_{safe_name}_rank{i}.png", dpi=300, bbox_inches="tight")
        plt.close()

    print(f"   → Графики 05_shap_waterfall (топ-{top_n}) сохранены")
    print("SHAP-анализ завершён\n")