 # Сравнивает модели, делает таблицу и график №03, возвращает имя лучшей

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from .config import RESULTS_DIR, PLOTS_DIR


def evaluate_and_compare(models, X_train, X_test, y_train_log, y_test_real, names_test):
    results = []

    print("Оценка моделей на тесте...")

    for name, model in models.items():
        # Исправляем систематическую ошибку модели
        train_pred = model.predict(X_train)
        correction = np.exp(y_train_log.mean() - train_pred.mean())

        test_pred_log = model.predict(X_test)
        test_pred = np.expm1(test_pred_log) * correction

        # Считаем метрики качества
        r2 = r2_score(y_test_real, test_pred)
        mae = mean_absolute_error(y_test_real, test_pred) / 1e9

        results.append({"Модель": name, "R²": round(r2, 4), "MAE (млрд $)": round(mae, 2)})

    # Сохраняем таблицу с результатами
    comparison = pd.DataFrame(results)
    comparison.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)

    # График №03 — сравнение всех моделей
    plt.figure(figsize=(11, 6))
    x = np.arange(len(comparison))
    plt.bar(x - 0.2, comparison["R²"], width=0.4, label="R²", color="skyblue", edgecolor="black")
    plt.bar(x + 0.2, comparison["MAE (млрд $)"], width=0.4, label="MAE (млрд $)", color="lightcoral", edgecolor="black")
    plt.xticks(x, comparison["Модель"], rotation=15, ha="right")
    plt.title("Сравнение моделей")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_models_comparison.png", dpi=300, bbox_inches='tight')  # ← теперь №03
    plt.close()

    best = comparison.loc[comparison["R²"].idxmax(), "Модель"]
    print(f"Лучшая модель по R²: {best}")
    return best