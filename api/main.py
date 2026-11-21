"""EmotionVision API (FastAPI, async).

Run locally:
    uvicorn api.main:app --reload --port 8000
Docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from api.inference import EmotionEngine
from api.schemas import Health, ModelInfo, PredictionResponse

MODEL_PATH = os.getenv("MODEL_PATH", "models/emotionvision.onnx")
CONFIG_PATH = os.getenv("CONFIG_PATH", "models/model_config.json")
DETECTOR_PATH = os.getenv("DETECTOR_PATH", "models/face_detector/yunet.onnx")
UNCERTAINTY_THRESHOLD = float(os.getenv("UNCERTAINTY_THRESHOLD", "0.40"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "8"))
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ONNX model and face detector once at startup
    state["engine"] = EmotionEngine(MODEL_PATH, CONFIG_PATH, DETECTOR_PATH, UNCERTAINTY_THRESHOLD)
    yield
    state.clear()


app = FastAPI(
    title="EmotionVision API",
    description="Facial expression recognition (7 classes) with Grad-CAM explanations - ONNX Runtime.",
    version="1.0.0",
    lifespan=lifespan,
)


def _engine() -> EmotionEngine:
    engine = state.get("engine")
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return engine


async def _read_image(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported type {file.content_type}. "
                                                    f"Allowed: {sorted(ALLOWED_TYPES)}")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File larger than {MAX_UPLOAD_MB} MB")
    return data


async def _run(file: UploadFile, explain: bool, max_faces: int) -> dict:
    data = await _read_image(file)
    try:
        # CPU-bound ONNX inference runs in a worker thread so the event loop stays free
        return await run_in_threadpool(_engine().predict, data, explain, max_faces)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/health", response_model=Health, tags=["system"])
async def health() -> Health:
    return Health(status="ok", model_loaded="engine" in state)


@app.get("/model/info", response_model=ModelInfo, tags=["system"])
async def model_info() -> ModelInfo:
    e = _engine()
    return ModelInfo(model=e.model_name, classes=e.classes, input_size=e.input_size,
                     onnx_path=e.onnx_path, uncertainty_threshold=e.uncertainty_threshold,
                     providers=e.session.get_providers())


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(file: UploadFile = File(..., description="Image (jpg, png, webp, bmp)"),
                  max_faces: int = Query(5, ge=1, le=20)) -> dict:
    """Detect faces and return the expression + class probabilities for each face."""
    return await _run(file, explain=False, max_faces=max_faces)


@app.post("/explain", response_model=PredictionResponse, tags=["inference"])
async def explain(file: UploadFile = File(...),
                  max_faces: int = Query(5, ge=1, le=20)) -> dict:
    """Same as /predict + a Grad-CAM heatmap (base64 PNG) for each face."""
    return await _run(file, explain=True, max_faces=max_faces)
