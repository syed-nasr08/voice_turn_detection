# Turn-detector inference service (CPU-only, INT8 ONNX)
# Build from repo root:  docker build -f serving/Dockerfile -t turn-detector .
# Run:                   docker run -p 8000:8000 turn-detector
FROM python:3.11-slim

WORKDIR /app

COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model is baked into the image: no startup download, instant readiness.
COPY serving/ serving/

ENV MODEL_DIR=/app/serving/model \
    MODEL_FILE=model.int8.onnx \
    ORT_THREADS=2

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=2s --start-period=15s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
