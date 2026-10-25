FROM python:3.11-slim

WORKDIR /app/tre

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# replayer: trace_oracle imports tre_replayer to re-schedule segment traces (PreServe Tier-1).
ENV PYTHONPATH=/app/tre/common:/app/tre/deploy:/app/tre/replayer:/app/tre/baselines

# One BLAS thread: the mLSTM forward pass is tiny and runs inside the tick (v1 D27).
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

COPY requirements-runtime.txt requirements-test.txt ./
COPY baselines/requirements-baselines.txt ./
RUN pip install --no-cache-dir -r requirements-test.txt -r requirements-baselines.txt

COPY common ./common
COPY deploy ./deploy
COPY replayer ./replayer
COPY baselines ./baselines

EXPOSE 8080
CMD ["python", "-m", "tre_baselines.main"]
