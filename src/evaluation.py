import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import shap
from xgboost import plot_importance
from sklearn.metrics import r2_score, mean_absolute_error
from src.config import MODELS_PATH, PLOTS_PATH

def evaluate_and_save(results, df, cv_results=None, X_test=None, y_test=None):
    print("\nОценка моделей...")

    comparison = []
    predictions = {}

    # 1. Оценка всех моделей на hold-out выборке
    for name, item in results.items():
        model = item['model']
        y_pred = model.predict(item['X_test'])
        y_test_current = item['y_test']

        r2 = r2_score(y_test_current, y_pred)
        mae = mean_absolute_error(np.expm1(y_test_current), np.expm1(y_pred)) / 1e9

        comparison.append({
            'Модель': name,
            'R² (на log-шкале)': round(r2, 4),
            'MAE (млрд $)': round(mae, 1)
        })
        predictions[name] = (y_test_current, y_pred)

        print(f"{name}: R² = {r2:.4f} (log), MAE = {mae:.1f} млрд $")

    # Выводим кросс-валидацию (если есть)
    if cv_results:
        print("\n" + "="*60)
        print("КРОСС-ВАЛИДАЦИЯ (5-FOLD) — СТАБИЛЬНОСТЬ РЕЗУЛЬТАТОВ".center(60))
        print("="*60)
        for name, res in cv_results.items():
            print(f"{name:18} → R² = {res['cv_r2_mean']:.4f} ± {res['cv_r2_std']:.4f}  |  MAE ≈ {res['cv_mae_original_mean']:.1f} млрд $")
        print("="*60)

    # Красивая табличка сравнения моделей
    comp_df = pd.DataFrame(comparison)
    print("\n" + "="*50)
    print(comp_df.to_string(index=False))
    print("="*50)

    # 2. Сохраняем лучшую модель
    best_model = results['XGBoost']['model']
    joblib.dump(best_model, os.path.join(MODELS_PATH, 'best_xgboost_model.pkl'))
    joblib.dump(X_test, os.path.join(MODELS_PATH, 'X_test.pkl'))
    print(f"\nЛучшая модель сохранена → {os.path.join(MODELS_PATH, 'best_xgboost_model.pkl')}")

    # 3. Feature Importance
    print("Генерация красивого Feature Importance...")

    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'axes.titlesize': 22,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'font.family': 'DejaVu Sans',
        'savefig.dpi': 400,
        'savefig.bbox': 'tight'
    })

    fig, ax = plt.subplots(figsize=(12, 9))

    # Берём важность по gain и сортируем
    importance = best_model.get_booster().get_score(importance_type='gain')
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    features = [x[0] for x in importance[::-1]]
    scores = [x[1] for x in importance[::-1]]

    # Красивые цвета
    colors = ['#e74c3c' if i == len(scores) - 1 else '#3498db' for i in range(len(scores))]

    bars = ax.barh(features, scores, color=colors, edgecolor='black', linewidth=1.2, height=0.7)

    # Подписываем значения на столбцах
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{width:.4f}', ha='left', va='center', fontsize=13, fontweight='bold')

    ax.set_title('Топ-10 самых важных признаков\n(XGBoost, gain importance)',
                 fontsize=24, fontweight='bold', pad=30)
    ax.set_xlabel('Importance score (gain)', fontsize=16)
    ax.grid(axis='x', alpha=0.3)
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, '03_feature_importance_xgboost.png'),
                facecolor='white', edgecolor='none')
    plt.close()
    print("Красивый Feature Importance сохранён → 03_feature_importance_xgboost.png")

    # 4. SHAP Summary Plot
    print("Генерация SHAP значений (может занять 10–20 секунд)...")
    explainer = shap.Explainer(best_model)
    shap_values = explainer(results['XGBoost']['X_test'])

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, results['XGBoost']['X_test'], show=False)
    plt.title('SHAP Summary Plot — влияние признаков', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, '04_shap_summary_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("SHAP Summary Plot сохранён — это ваш главный козырь на защите!")

    # 5. ФИНАЛЬНАЯ ТАБЛИЧКА ТОЛЬКО ДЛЯ XGBoost
    print("\n" + "="*90)
    print("ПРЕДСКАЗАНИЯ ЛУЧШЕЙ МОДЕЛИ (XGBoost) НА ОТЛОЖЕННОЙ ВЫБОРКЕ (20%)".center(90))
    print("="*90)

    y_pred_log = best_model.predict(results['XGBoost']['X_test'])
    y_pred = np.expm1(y_pred_log) / 1e9
    y_true = np.expm1(results['XGBoost']['y_test']) / 1e9

    test_df = df.iloc[results['XGBoost']['y_test'].index].copy()
    test_df['Реальная кап-я, млрд $'] = y_true.round(2)
    test_df['Прогноз, млрд $'] = y_pred.round(2)
    test_df['Ошибка, %'] = (abs(y_true - y_pred) / y_true * 100).round(2)

    top_predictions = test_df.nlargest(12, 'Реальная кап-я, млрд $')[[
        'Name', 'Реальная кап-я, млрд $', 'Прогноз, млрд $', 'Ошибка, %'
    ]].copy()

    # Укорачиваем длинные имена
    top_predictions['Name'] = top_predictions['Name'].str.split().str[:3].str.join(' ')

    print(top_predictions.to_string(index=False))
    print("="*90)

    # Сохраняем табличку
    top_predictions.to_csv(os.path.join(PLOTS_PATH, 'top_companies_predictions.csv'), index=False)
    print("Финальная табличка сохранена → results/plots/top_companies_predictions.csv")

    # 6. График сравнения моделей
    plt.figure(figsize=(10, 6))
    plt.bar(comp_df['Модель'], comp_df['R² (на log-шкале)'], color=['#95a5a6', '#3498db', '#e74c3c'])
    plt.title('Сравнение моделей по R² (hold-out 20%)', fontsize=14, fontweight='bold')
    plt.ylabel('R²')
    plt.ylim(0, 1)
    for i, v in enumerate(comp_df['R² (на log-шкале)']):
        plt.text(i, v + 0.01, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("\nВсе графики и модель успешно сохранены!")
    print("ГОТОВО!")