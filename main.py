from src.data_preprocessing import load_and_prepare_data
from src.eda import run_eda
from src.modeling import train_models
from src.evaluation import evaluate_and_save

if __name__ == "__main__":
    print("ПРОГНОЗИРОВАНИЕ СТОИМОСТИ КОМПАНИИ")
    print("=" * 60)

    # 1. Загрузка и подготовка
    df, X_scaled, y, scaler = load_and_prepare_data()

    # 2. EDA
    run_eda(df)

    # 3. Обучение
    results, cv_results, X_test, y_test = train_models(X_scaled, y)

    # 4. Оценка и сохранение
    evaluate_and_save(results, df, cv_results, X_test, y_test)

    print("\nГОТОВО! Всё сохранено в папках results/ и models/")