"""Grad-CAM helpers."""

import numpy as np

from src.explain.cam import cam_for_class, normalize_cam, overlay


def test_normalize_range():
    cam = normalize_cam(np.array([[-1.0, 0.0], [2.0, 4.0]]))
    assert cam.min() == 0.0 and cam.max() == 1.0


def test_normalize_constant_map_is_zero():
    assert np.all(normalize_cam(np.ones((4, 4))) == 0)


def test_cam_for_class_shape_and_range():
    cams = np.random.default_rng(0).normal(size=(7, 7, 7)).astype(np.float32)
    heat = cam_for_class(cams, 3, (224, 224))
    assert heat.shape == (224, 224)
    assert heat.min() >= 0 and heat.max() <= 1


def test_contrastive_removes_component_shared_by_all_classes():
    rng = np.random.default_rng(0)
    specific = np.zeros((7, 7, 7), dtype=np.float32)
    specific[2, 1, 1] = 5.0                    # class 2 relies on cell (1, 1)
    shared = np.zeros((7, 7), dtype=np.float32)
    shared[6, 6] = 50.0                        # strong corner artifact in every class
    cams = specific + shared + rng.normal(scale=0.01, size=(7, 7, 7)).astype(np.float32)

    raw = cam_for_class(cams, 2, (7, 7), mode="raw")
    con = cam_for_class(cams, 2, (7, 7), mode="contrastive")
    assert np.unravel_index(raw.argmax(), raw.shape) == (6, 6)   # raw is fooled by the artifact
    assert np.unravel_index(con.argmax(), con.shape) == (1, 1)   # contrastive finds the real region


def test_overlay_shape_and_dtype():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    out = overlay(img, np.ones((64, 64), dtype=np.float32))
    assert out.shape == (64, 64, 3) and out.dtype == np.uint8
