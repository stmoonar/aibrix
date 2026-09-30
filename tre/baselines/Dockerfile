FROM python:3.11-slim

WORKDIR /app/tre

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/tre/common:/app/tre/deploy:/app/tre/baselines

COPY requirements-runtime.txt requirements-test.txt ./
RUN pip install --no-cache-dir -r requirements-test.txt

COPY common ./common
COPY deploy ./deploy
COPY baselines ./baselines

EXPOSE 8080
CMD ["python", "-m", "tre_baselines.main"]
