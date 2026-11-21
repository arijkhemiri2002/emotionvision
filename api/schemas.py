"""Pydantic schemas of the EmotionVision API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: int = Field(..., ge=0, description="left (pixels)")
    y: int = Field(..., ge=0, description="top (pixels)")
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)


class FacePrediction(BaseModel):
    box: BoundingBox
    detection_score: float = Field(..., ge=0, le=1, description="face detector confidence")
    label: str = Field(..., examples=["happiness"])
    confidence: float = Field(..., ge=0, le=1)
    uncertain: bool = Field(..., description="True if confidence < uncertainty threshold")
    probabilities: dict[str, float]
    heatmap_png_base64: str | None = Field(
        None, description="Grad-CAM overlay on the face crop (only for /explain)")


class PredictionResponse(BaseModel):
    faces: list[FacePrediction]
    n_faces: int
    face_detected: bool = Field(..., description="False -> no face found, whole image was classified")
    image_width: int
    image_height: int
    inference_ms: float


class ModelInfo(BaseModel):
    model: str
    classes: list[str]
    input_size: int
    onnx_path: str
    uncertainty_threshold: float
    providers: list[str]


class Health(BaseModel):
    status: str
    model_loaded: bool
