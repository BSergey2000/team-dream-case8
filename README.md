# README для проекта «Прогнозирование стоимости компании"

## Краткое описание

Это проект полного цикла (end-to-end) для прогнозирования Market Cap компаний по финансовым показателям. Включает:
- EDA (анализ данных),
- предобработку (Pipeline с FunctionTransformer + ColumnTransformer),
- подбор и обучение моделей (Optuna, LightGBM, XGBoost, GradientBoosting, Ensemble),
- оценку (метрики, CV),
- объяснимость (SHAP),
- простой API для предсказаний (FastAPI).


## Быстрый старт (Windows / PowerShell)

1) Клонируйте репозиторий и перейдите в папку проекта:

```powershell
git clone https://github.com/BSergey2000/team-dream-case8.git
cd team-dream-case8
```

2) Создайте и активируйте виртуальное окружение:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3) Установите зависимости:

```powershell
pip install -r requirements.txt
```

4) (Рекомендуется) Установите дополнительно `tabulate` для корректной генерации markdown-таблиц в отчёте:

```powershell
pip install tabulate
```

5) Поместите `financials.csv` в папку `data/`.

## Запуск проекта

### Полный пайплайн (обучение, анализ, отчёт)

```powershell
python main.py
```

Что получится:
- логи: `logs/project.log`
- модели: `models/*.pkl`
- графики и отчёт: `results/plots/`, `results/report.md`

Если при создании отчёта появится лог о нехватке зависимости `tabulate`, установите её (см. выше).

### API (FastAPI)

Запустите сервер:

```powershell
uvicorn src.deploy:app --reload --host 0.0.0.0 --port 8000
```

- Swagger UI: `http://localhost:8000/docs`
- Быстрый тест в браузере: `http://localhost:8000/predict_test` — возвращает тестовое предсказание и имя модели.
- Пример POST (PowerShell):

```powershell
Invoke-RestMethod -Method POST -Uri http://localhost:8000/predict -ContentType 'application/json' -Body '{"Price":100.0, "Price/Earnings":20.0, "Dividend Yield":1.5, "Earnings/Share":5.0, "52 Week Low":80.0, "52 Week High":120.0, "EBITDA":1000000000.0, "Price/Sales":3.0, "Price/Book":4.0, "Sector":"Information Technology"}'
```

Ответ: JSON `{"predicted_market_cap": <float>}`

### Docker

```powershell
docker build -t market-cap-predictor .
docker run -p 8000:8000 market-cap-predictor
```

## Структура проекта

- `configs/config.py` — константы и логгер
- `src/eda.py` — EDA (графики и статистика)
- `src/preprocess.py` — подготовка данных (Pipeline)
- `src/train.py` — модели и оптимизация (Optuna)
- `src/metrics.py` — оценка, сравнение моделей и генерация отчёта
- `src/shap_analysis.py` — SHAP анализ и графики
- `src/deploy.py` — FastAPI сервер для предсказаний
- `main.py` — оркестратор пайплайна
- `data/financials.csv` — входной датасет
- `results/` — графики и отчёт (игнорируется в git)
- `models/` — сохранённые модели (игнорируются в git)

## Примечания по данным

Ожидаемый CSV содержит признаки: Price, Price/Earnings, Dividend Yield, Earnings/Share, 52 Week Low/High, Market Cap (target), EBITDA, Price/Sales, Price/Book, Sector и др. В коде предусмотрена обработка пропусков и защита от деления на ноль при генерации новых признаков.

## Отладка и проверки

- Для быстрого smoke-test можно импортировать модули Python (проверка синтаксиса и I/O):

```powershell
python -c "import importlib; importlib.import_module('src.preprocess'); importlib.import_module('src.train')"
```

- Логи помогут понять ошибки: `logs/project.log`.


## Лицензия

MIT License

## .gitignore

В проекте есть файл `.gitignore`, который исключает временные и большие артефакты (виртуальное окружение, логи, модели, результаты и т.п.).