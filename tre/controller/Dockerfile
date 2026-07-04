FROM python:3.11-slim

WORKDIR /app/tre

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/tre/common:/app/tre/deploy:/app/tre/controller:/app/tre/service-manager

COPY requirements-runtime.txt requirements-test.txt ./
RUN pip install --no-cache-dir -r requirements-test.txt

COPY common ./common
COPY deploy ./deploy
COPY controller ./controller
COPY service-manager ./service-manager

CMD ["python", "-m", "tre_controller"]
