"""
Модуль для запуска сервера с моделью.
Берется лучшая модель, FastAPI — принимает вводные данные и выдает предсказание
Сервер — ждёт запросов от пользователя, /predict — адрес, куда отправлять данные, чтобы получить ответ (стоимость компании). Мы загружаем лучшую модель, проверяем, считаем и возвращаем.
Запуск: uvicorn src.deploy:app --reload
"""

from configs.config import MODELS_DIR, logger  # Настройки
import pandas as pd  # Для таблиц
import numpy as np  # Для чисел
import joblib  # Для загрузки модели
from pydantic import BaseModel, Field  # Pydantic — проверка данных
from fastapi import FastAPI, HTTPException # FastAPI — сервер для запросов, HTTPException — ошибки
import sys  # Помощник для путей
from pathlib import Path  # Для адресов

ROOT_DIR = Path(__file__).parent.parent  # Главная папка
sys.path.insert(0, str(ROOT_DIR))  # Добавляем путь
sys.path.insert(0, str(ROOT_DIR / 'src'))  # К модулям


# Создание
app = FastAPI(title="Market Cap Predictor",
              description="Сервер для предсказания капитализации компаний.")

# Храним выбранное имя лучшей модели для диагностических ответов
BEST_MODEL_NAME = None


class CompanyInput(BaseModel):
    """
    Что делает: Шаблон для данных от юзера.
    """
    # Используем безопасные python-идентификаторы и задаём alias'ы, чтобы можно было принимать поля с оригинальными названиями из CSV/JSON
    price: float = Field(alias="Price")  # Цена акции, число
    price_earnings: float = Field(alias="Price/Earnings")  # P/E, соотношение
    dividend_yield: float = Field(alias="Dividend Yield")  # Доходность
    earnings_share: float = Field(alias="Earnings/Share")  # Прибыль на акцию
    week_52_low: float = Field(alias="52 Week Low")  # Минимум за год
    week_52_high: float = Field(alias="52 Week High")  # Максимум
    ebitda: float = Field(alias="EBITDA")  # Прибыль до вычетов
    price_sales: float = Field(alias="Price/Sales")  # P/S
    price_book: float = Field(alias="Price/Book")  # P/B
    sector: str = Field(alias="Sector")  # Сектор, как группа компаний


def load_best_model():
    """
    Что делает: Загружает лучшую модель
    Возвращает: Модель или ошибку.
    """
    # Пытаемся выбрать лучшую модель автоматически по файлу results/models_comparison.csv
    try:
        comp_file = ROOT_DIR / 'results' / 'models_comparison.csv'
        if comp_file.exists():
            import pandas as _pd
            df = _pd.read_csv(comp_file)
            # Найдём колонку с R^2 для real
            r2_col = None
            for c in df.columns:
                if 'r^2' in c.lower() and 'real' in c.lower():
                    r2_col = c
                    break
            if r2_col is None:
                for c in df.columns:
                    if 'real' in c.lower():
                        r2_col = c
                        break
            if r2_col is not None and not df.empty:
                best_idx = df[r2_col].idxmax()
                model_name = df.iloc[best_idx, 0]
            else:
                model_name = 'LightGBM'
        else:
            model_name = 'LightGBM'

        model_file = MODELS_DIR / \
            f"{str(model_name).lower().replace(' ', '')}_model.pkl"
        try:
            pipe = joblib.load(model_file)
            logger.info(f"Загружена модель: {model_name} из {model_file}")
            # Сохраняем имя выбранной модели в глобальную переменную
            global BEST_MODEL_NAME
            BEST_MODEL_NAME = str(model_name)
            return pipe
        except FileNotFoundError:
            logger.error(
                f"Файл модели не найден: {model_file}. Обучите модель заново.")
            raise HTTPException(status_code=500, detail="Модель не найдена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise HTTPException(
                status_code=500, detail="Ошибка загрузки модели")
    except Exception as e:
        logger.error(f"Ошибка при выборе лучшей модели: {e}")
        raise HTTPException(status_code=500, detail="Ошибка выбора модели")


# Загружаем модель
model_pipe = load_best_model()


@app.get("/predict_test")
def predict_test():
    """
    Быстрый тестовый эндпоинт для браузера: возвращает предсказание для фиксированного тестового пейлоада
    с использованием автоматически выбранной лучшей модели
    """
    try:
        sample = {
            "Price": 100.0,
            "Price/Earnings": 20.0,
            "Dividend Yield": 1.5,
            "Earnings/Share": 3.0,
            "52 Week Low": 50.0,
            "52 Week High": 150.0,
            "EBITDA": 1000000000.0,
            "Price/Sales": 4.0,
            "Price/Book": 3.0,
            "Sector": "Technology"
        }
        # Создаём DataFrame из примера — структура колонок совпадает с тем, что ожидает препроцессор
        df = pd.DataFrame([sample])
        pred_log = model_pipe.predict(df)
        pred_real = np.expm1(pred_log)[0]
        return {"predicted_market_cap": float(pred_real), "model": BEST_MODEL_NAME}
    except Exception as e:
        logger.error(f"Ошибка в predict_test: {e}")
        raise HTTPException(
            status_code=500, detail="Ошибка предсказания тестового примера")


@app.post("/predict")
def predict(input_data: CompanyInput):
    """
    Что делает: Принимает данные, предсказывает стоимость
    Параметры: input_data — форма с данными
    Возвращает: Словарь с предсказанием
    Это адрес /predict, куда отправлять данные (как POST-запрос в браузере). Мы проверяем, превращаем в таблицу, предсказываем, возвращаем число в миллиардах
    """
    try:
        # Превращаем Pydantic модель в DataFrame, используя alias'ы — это гарантирует соответствие исходным CSV-именам
        df = pd.DataFrame([input_data.dict(by_alias=True)])

        # Предсказание: конвейер делает всё
        pred_log = model_pipe.predict(df)
        # Обратный log, чтобы нормальное число
        pred_real = np.expm1(pred_log)[0]

        logger.info(
            f"Предсказание для сектора {input_data.sector}: {pred_real:.2f} млрд $")
        return {"predicted_market_cap": float(pred_real)}  # Возвращаем ответ
    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}. Проверьте данные в запросе.")
        raise HTTPException(status_code=500, detail="Ошибка предсказания")


@app.get("/")
def root():
    """
    Что делает: Главная страница для проверки
    Возвращает: Сообщение
    Это адрес /, куда зайти, чтобы увидеть, работает ли сервер
    """
    return {"message": "Сервер для предсказания капитализации. Используйте /predict."}


if __name__ == '__main__':
    import uvicorn  # Запускаем сервер
    # Открываем порт 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)