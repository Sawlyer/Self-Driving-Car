"""Tests du prétraitement, de l'augmentation, des données et des métriques."""

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from sdc.augment import augment, choose_camera, random_flip, random_translate
from sdc.config import INPUT_SHAPE, RAW_HEIGHT, RAW_WIDTH
from sdc.control import PIController, SteeringSmoother, kmh_to_mph, mph_to_kmh, target_speed
from sdc.dataset import DrivingDataset, load_driving_log, load_samples
from sdc.metrics import autonomy, regression_report
from sdc.preprocess import load_image, preprocess


def raw_image(value=128):
    """Image brute de la taille produite par le simulateur."""
    return np.full((RAW_HEIGHT, RAW_WIDTH, 3), value, dtype=np.uint8)


def test_preprocess_shape_and_dtype():
    out = preprocess(raw_image())
    assert out.shape == INPUT_SHAPE
    assert out.dtype == np.uint8


def test_load_image_returns_rgb(tmp_path):
    image = np.zeros((RAW_HEIGHT, RAW_WIDTH, 3), dtype=np.uint8)
    image[..., 0] = 255  # rouge en RGB
    path = tmp_path / "red.png"
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    loaded = load_image(path)
    assert loaded[0, 0].tolist() == [255, 0, 0]


def test_load_image_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "absent.jpg")


def test_choose_camera_correction():
    picked = set()
    rng = np.random.default_rng(0)
    for _ in range(50):
        camera, steering = choose_camera("c", "l", "r", 0.1, rng, correction=0.2)
        picked.add((camera, round(steering, 6)))
    assert picked == {("c", 0.1), ("l", 0.3), ("r", -0.1)}


def test_flip_mirrors_image_and_negates_steering():
    image = raw_image(0)
    image[:, 0] = 255  # colonne de gauche blanche
    for seed in range(10):
        out, steering = random_flip(image, 0.4, np.random.default_rng(seed))
        if steering == -0.4:
            assert out[:, -1].min() == 255
            break
    else:
        pytest.fail("aucun retournement en 10 tirages")


def test_translate_adjusts_steering_with_shift_direction():
    rng = np.random.default_rng(1)
    _, steering = random_translate(raw_image(), 0.0, rng, max_shift_x=50, max_shift_y=0)
    assert steering != 0.0
    assert abs(steering) <= 50 * 0.004 + 1e-9


def test_augment_keeps_steering_in_range():
    rng = np.random.default_rng(2)
    for _ in range(20):
        image, steering = augment(raw_image(), 0.95, rng)
        assert image.shape == (RAW_HEIGHT, RAW_WIDTH, 3)
        assert -1.0 <= steering <= 1.0


@pytest.fixture
def recording(tmp_path):
    """Faux enregistrement au format du simulateur (chemins Windows absolus)."""
    img_dir = tmp_path / "IMG"
    img_dir.mkdir()
    lines = []
    for i, angle in enumerate([0.0, 0.0, 0.25, -0.3]):
        names = [f"{cam}_{i}.jpg" for cam in ("center", "left", "right")]
        for name in names:
            cv2.imwrite(str(img_dir / name), raw_image(40 * i))
        paths = [f"C:\\Users\\someone\\rec\\IMG\\{n}" for n in names]
        lines.append(", ".join(paths + [str(angle), "0.5", "0", "20.0"]))
    (tmp_path / "driving_log.csv").write_text("\n".join(lines) + "\n")
    return tmp_path


def test_load_driving_log_resolves_absolute_windows_paths(recording):
    frame = load_driving_log(recording)
    assert len(frame) == 4
    assert frame["center"][0] == str(recording / "IMG" / "center_0.jpg")
    assert frame["right"][3] == str(recording / "IMG" / "right_3.jpg")
    assert frame["steering"].tolist() == [0.0, 0.0, 0.25, -0.3]


def test_load_driving_log_skips_udacity_header(tmp_path):
    (tmp_path / "IMG").mkdir()
    (tmp_path / "driving_log.csv").write_text(
        "center,left,right,steering,throttle,brake,speed\n"
        "IMG/c.jpg, IMG/l.jpg, IMG/r.jpg, 0.1, 0, 0, 22.1\n"
    )
    frame = load_driving_log(tmp_path)
    assert len(frame) == 1
    assert frame["left"][0] == str(tmp_path / "IMG" / "l.jpg")


def test_keep_straight_drops_only_straight_rows(recording):
    frame = load_samples([recording], keep_straight=0.0)
    assert sorted(frame["steering"].tolist()) == [-0.3, 0.25]


def test_dataset_batches(recording):
    frame = load_driving_log(recording)
    images, angles = next(iter(DataLoader(DrivingDataset(frame, training=True), batch_size=3)))
    assert tuple(images.shape) == (3, *INPUT_SHAPE)
    assert images.dtype == torch.uint8
    assert tuple(angles.shape) == (3,)
    val = DataLoader(DrivingDataset(frame, training=False), batch_size=4)
    _, val_angles = next(iter(val))
    # En validation : caméra centrale, aucune transformation de l'angle.
    assert val_angles.tolist() == pytest.approx([0.0, 0.0, 0.25, -0.3])


def test_pi_controller_accelerates_below_set_point_and_saturates():
    controller = PIController(set_point=10)
    assert controller.update(0) > 0
    assert controller.update(30) < 0
    assert PIController(kp=10, set_point=10).update(0) == 1.0


def test_autonomy_formula():
    assert autonomy(0, 60) == 100.0
    assert autonomy(1, 60) == pytest.approx(90.0)
    assert autonomy(20, 60) == 0.0
    with pytest.raises(ValueError):
        autonomy(1, 0)


def test_regression_report():
    report = regression_report([0.0, 1.0], [0.0, 0.5])
    assert report["mse"] == pytest.approx(0.125)
    assert report["mae"] == pytest.approx(0.25)


def test_target_speed_slows_down_in_turns():
    assert target_speed(0.0, 30, 15) == 30
    assert target_speed(-0.15, 30, 15, full_turn=0.3) == pytest.approx(22.5)
    assert target_speed(0.8, 30, 15) == 15


def test_steering_smoother_and_unit_conversion():
    smoother = SteeringSmoother(alpha=0.5)
    assert smoother.update(0.4) == 0.4
    assert smoother.update(0.0) == pytest.approx(0.2)
    assert mph_to_kmh(kmh_to_mph(48.0)) == pytest.approx(48.0)
    assert mph_to_kmh(30) == pytest.approx(48.28, abs=0.01)
