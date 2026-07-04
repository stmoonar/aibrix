FROM python:3.11-slim

WORKDIR /app/tre

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/tre/common:/app/tre/deploy:/app/tre/ui

COPY requirements-runtime.txt requirements-test.txt ./
RUN pip install --no-cache-dir -r requirements-test.txt

COPY common ./common
COPY deploy ./deploy
COPY ui ./ui

EXPOSE 8000
CMD ["uvicorn", "tre_ui.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
