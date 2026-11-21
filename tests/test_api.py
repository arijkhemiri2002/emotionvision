"""FastAPI endpoints (dummy ONNX model, real YuNet detector)."""

import base64

import pytest

from tests.conftest import CLASSES


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    info = client.get("/model/info").json()
    assert info["classes"] == CLASSES
    assert info["input_size"] == 32
    assert "CPUExecutionProvider" in info["providers"]


def test_predict_schema(client, png_bytes):
    r = client.post("/predict", files={"file": ("x.png", png_bytes, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["image_width"] == 160 and body["image_height"] == 120
    # random noise has no face -> whole image classified
    assert body["face_detected"] is False and body["n_faces"] == 0
    face = body["faces"][0]
    assert face["label"] in CLASSES
    assert set(face["probabilities"]) == set(CLASSES)
    assert sum(face["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert face["confidence"] == max(face["probabilities"].values())
    assert face["heatmap_png_base64"] is None


def test_explain_returns_png_heatmap(client, png_bytes):
    face = client.post("/explain", files={"file": ("x.png", png_bytes, "image/png")}).json()["faces"][0]
    raw = base64.b64decode(face["heatmap_png_base64"])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_unsupported_type(client):
    r = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_corrupted_image(client):
    r = client.post("/predict", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_empty_file(client):
    r = client.post("/predict", files={"file": ("x.png", b"", "image/png")})
    assert r.status_code == 400


def test_file_too_large(client, monkeypatch):
    import api.main

    monkeypatch.setattr(api.main, "MAX_UPLOAD_MB", 0.0001)
    r = client.post("/predict", files={"file": ("x.png", b"\x00" * 1000, "image/png")})
    assert r.status_code == 413


def test_max_faces_validation(client, png_bytes):
    r = client.post("/predict?max_faces=0", files={"file": ("x.png", png_bytes, "image/png")})
    assert r.status_code == 422
