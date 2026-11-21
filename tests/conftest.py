"""Shared fixtures. CI has no DVC data and no trained model, so the API tests use a tiny
ONNX model built on the fly with the same interface (input -> logits, cams)."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]
INPUT_SIZE = 32


def make_dummy_onnx(path: Path, n_classes: int = 7) -> None:
    """input (B,3,S,S) -> 1x1 Conv -> cams (B,7,S,S) -> spatial mean -> logits (B,7)."""
    rng = np.random.default_rng(0)
    w = helper.make_tensor("W", TensorProto.FLOAT, [n_classes, 3, 1, 1],
                           rng.normal(size=(n_classes, 3, 1, 1)).astype(np.float32).ravel())
    axes = helper.make_tensor("axes", TensorProto.INT64, [2], [2, 3])
    nodes = [
        helper.make_node("Conv", ["input", "W"], ["cams"]),
        helper.make_node("ReduceMean", ["cams", "axes"], ["logits"], keepdims=0),
    ]
    graph = helper.make_graph(
        nodes, "dummy",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["B", 3, INPUT_SIZE, INPUT_SIZE])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["B", n_classes]),
         helper.make_tensor_value_info("cams", TensorProto.FLOAT, ["B", n_classes, INPUT_SIZE, INPUT_SIZE])],
        initializer=[w, axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


@pytest.fixture(scope="session")
def model_files(tmp_path_factory) -> dict[str, str]:
    d = tmp_path_factory.mktemp("model")
    make_dummy_onnx(d / "model.onnx")
    cfg = {"model": "dummy", "classes": CLASSES, "input_size": INPUT_SIZE,
           "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "grayscale_to_rgb": True}
    (d / "model_config.json").write_text(json.dumps(cfg))
    return {"onnx": str(d / "model.onnx"), "config": str(d / "model_config.json"),
            "detector": str(ROOT / "models" / "face_detector" / "yunet.onnx")}


@pytest.fixture()
def client(model_files, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MODEL_PATH", model_files["onnx"])
    monkeypatch.setenv("CONFIG_PATH", model_files["config"])
    monkeypatch.setenv("DETECTOR_PATH", model_files["detector"])
    import api.main

    importlib.reload(api.main)  # re-read the env variables
    with TestClient(api.main.app) as c:
        yield c


@pytest.fixture()
def png_bytes() -> bytes:
    img = (np.random.default_rng(1).random((120, 160, 3)) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()
