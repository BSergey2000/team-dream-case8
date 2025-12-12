# ## Dockerfile для деплоя API
# Build: docker build -t market-cap-predictor .
# Run: docker run -p 8000:8000 market-cap-predictor

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["uvicorn", "src.deploy:app", "--host", "0.0.0.0", "--port", "8000"]