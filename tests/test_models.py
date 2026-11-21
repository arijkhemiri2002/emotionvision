"""Models and the ONNX export wrapper."""

import pytest
import torch

from src.models.export_onnx import LogitsAndCAM
from src.models.factory import build_model, set_backbone_trainable


@pytest.mark.parametrize("name", ["baseline_cnn", "efficientnet_b0"])
def test_output_shape(name):
    model = build_model(name, num_classes=7, pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 64, 64)).shape == (2, 7)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model("vgg_unknown", 7)


def test_freeze_backbone_keeps_head_trainable():
    model = build_model("efficientnet_b0", 7, pretrained=False)
    set_backbone_trainable(model, False)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("classifier") for n in trainable)


@pytest.mark.parametrize("name", ["baseline_cnn", "efficientnet_b0"])
def test_cam_wrapper_is_consistent_with_logits(name):
    """CAM property of a GAP+Linear head: spatial mean of CAM_c + bias_c == logit_c."""
    model = build_model(name, 7, pretrained=False).eval()
    wrapper = LogitsAndCAM(model).eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        logits, cams = wrapper(x)
        ref = model(x)
    assert torch.allclose(logits, ref, atol=1e-4)
    assert torch.allclose(cams.mean(dim=(2, 3)) + wrapper.head.bias, logits, atol=1e-4)
