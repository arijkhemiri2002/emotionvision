# EmotionVision API - FastAPI + ONNX Runtime (no torch -> small image)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Code: API + Grad-CAM helpers only
COPY api/ api/
COPY src/__init__.py src/__init__.py
COPY src/explain/__init__.py src/explain/cam.py src/explain/

# Model artifacts (ONNX from `dvc repro`, face detector versioned with Git)
COPY models/emotionvision.onnx models/model_config.json models/
COPY models/face_detector/yunet.onnx models/face_detector/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
