"""Tests du modèle et de la boucle de conduite (sans simulateur)."""

import base64
from io import BytesIO

import numpy as np
import torch
from PIL import Image

from drive import Pilot, decode_image
from sdc.config import INPUT_SHAPE, RAW_HEIGHT, RAW_WIDTH
from sdc.model import PilotNet, count_parameters, load_model, save_model


def encode(image):
    """Encode une image comme le simulateur : JPEG en base64."""
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_model_matches_nvidia_architecture():
    model = PilotNet()
    out = model(torch.zeros((2, *INPUT_SHAPE), dtype=torch.uint8))
    assert tuple(out.shape) == (2,)
    # Environ 250 000 poids, comme dans l'article.
    assert 240_000 < count_parameters(model) < 260_000
    convs = [m.out_channels for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    assert convs == [24, 36, 48, 64, 64]


def test_model_saves_and_reloads(tmp_path):
    model = PilotNet(dropout=0.3).eval()
    path = tmp_path / "model.pt"
    save_model(model, path)
    reloaded = load_model(path)
    batch = torch.randint(0, 255, (1, *INPUT_SHAPE), dtype=torch.uint8)
    with torch.no_grad():
        torch.testing.assert_close(model(batch), reloaded(batch))
    assert reloaded.dropout == 0.3
    assert not reloaded.training


def test_decode_image_roundtrip():
    image = np.full((RAW_HEIGHT, RAW_WIDTH, 3), 90, dtype=np.uint8)
    decoded = decode_image(encode(image))
    assert decoded.shape == image.shape
    assert abs(int(decoded.mean()) - 90) <= 2


def test_pilot_step_returns_bounded_commands(tmp_path):
    pilot = Pilot(PilotNet().eval(), max_speed=10, min_speed=5, record_dir=tmp_path / "run")
    image = np.random.default_rng(0).integers(0, 255, (RAW_HEIGHT, RAW_WIDTH, 3), dtype=np.uint8)
    steering, throttle = pilot.step({"image": encode(image), "speed": "3.0",
                                     "steering_angle": "0", "throttle": "0"})
    assert -1.0 <= steering <= 1.0
    assert throttle > 0  # sous la consigne : on accélère
    assert len(list((tmp_path / "run").glob("*.jpg"))) == 1
