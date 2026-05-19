FROM python:3.11-slim as builder
WORKDIR /app
COPY week2/backend/requirements.txt .

RUN apt-get update && apt-get install -y libgomp1 libgcc-s1 && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 libgcc-s1 curl && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY week2/backend/ .
COPY week2/metadata /metadata
RUN mkdir -p /data/processed /app/frontend/public

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]