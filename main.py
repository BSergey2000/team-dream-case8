# Один запуск — всё готово

from src.eda import generate_eda_plots
from src.data_preparation import load_and_prepare_data
from src.models_training import train_all_models
from src.evaluation import evaluate_and_compare
from src.shap_analysis import generate_shap_analysis

def main():
    print("="*70)
    print("ПРОГНОЗИРОВАНИЕ СТОИМОСТИ КОМПАНИЙ — ПОЛНОЕ СРАВНЕНИЕ")
    print("="*70)

    generate_eda_plots()
    data = load_and_prepare_data()

    models = train_all_models(
        data["X_train"],
        data["X_test"],
        data["y_train_log"],
        data["y_test_log"]
    )

    best_model_name = evaluate_and_compare(
        models,
        data["X_train"],
        data["X_test"],
        data["y_train_log"],
        data["y_test_real"],
        data["names_test"]
    )

    print(f"SHAP-анализ для лучшей модели: {best_model_name}")
    generate_shap_analysis(best_model_name)

    print("="*70)
    print("ГОТОВО! ВСЕ МОДЕЛИ СОХРАНЕНЫ, СРАВНЕНИЕ ПРОВЕДЕНО!")
    print("="*70)

if __name__ == "__main__":
    main()