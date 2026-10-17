"""Pilote la voiture du simulateur Udacity en mode autonome.

Usage ::

    python drive.py models/model.pt
    python drive.py models/model.pt --max-speed 48 --min-speed 25 --record runs/tour1

Les vitesses se donnent en km/h. Le simulateur plafonne la voiture à
30 mph, soit environ 48 km/h : au-delà, la consigne est ramenée à ce
maximum.

Lancer ce script, puis le simulateur, choisir une piste et cliquer sur
« Autonomous Mode ». Le simulateur se connecte en Socket.IO sur le port
4567 et envoie un événement ``telemetry`` à chaque image. On répond par un
événement ``steer`` contenant l'angle prédit et l'accélérateur.
"""

import argparse
import base64
import csv
import time
import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path

# Masque l'avertissement d'obsolescence qu'eventlet affiche à l'import.
warnings.filterwarnings("ignore", message=r"\s*Eventlet is deprecated")

# Les imports suivants doivent venir après le filtre.
# ruff: noqa: E402
import eventlet
import eventlet.wsgi
import numpy as np
import socketio
import torch
from PIL import Image

from sdc.config import SIMULATOR_PORT, STEERING_MAX, STEERING_MIN
from sdc.control import (
    SIMULATOR_MAX_SPEED_MPH,
    PIController,
    SteeringSmoother,
    kmh_to_mph,
    mph_to_kmh,
    target_speed,
)
from sdc.model import default_device, load_model
from sdc.preprocess import preprocess


def decode_image(encoded):
    """Décode l'image JPEG en base64 envoyée par le simulateur (RGB)."""
    return np.asarray(Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB"))


class Pilot:
    """Associe le modèle de braquage et le régulateur de vitesse.

    Args:
        model: modèle PyTorch en mode évaluation, qui prend un lot d'images
            ``uint8`` de forme ``(lot, 66, 200, 3)``.
        max_speed: vitesse visée en ligne droite, en mph.
        min_speed: vitesse visée dans les virages les plus serrés, en mph.
        smoothing: poids de l'angle précédent dans le lissage (0 = aucun).
        record_dir: dossier où sauvegarder les images reçues (optionnel).
    """

    def __init__(self, model, max_speed=25.0, min_speed=15.0, smoothing=0.3, record_dir=None):
        self.model = model
        self.device = next(model.parameters()).device
        self.max_speed = min(max_speed, SIMULATOR_MAX_SPEED_MPH)
        self.min_speed = min(min_speed, self.max_speed)
        self.controller = PIController(set_point=self.max_speed)
        self.smoother = SteeringSmoother(smoothing)
        self.record_dir = Path(record_dir) if record_dir else None
        self.log_writer = None
        if self.record_dir:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            log_file = open(self.record_dir / "telemetry.csv", "w", newline="", encoding="utf-8")
            self.log_writer = csv.writer(log_file)
            self.log_writer.writerow(["time", "steering", "throttle", "speed"])
            self._log_file = log_file
        self.start = time.monotonic()

    def predict_steering(self, image):
        """Prédit l'angle de braquage pour une image brute RGB du simulateur.

        Applique exactement le même prétraitement qu'à l'entraînement.
        """
        batch = torch.from_numpy(preprocess(image)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            steering = self.model(batch).item()
        return float(np.clip(steering, STEERING_MIN, STEERING_MAX))

    def step(self, telemetry):
        """Calcule ``(angle, accélérateur)`` à partir d'un message ``telemetry``."""
        image = decode_image(telemetry["image"])
        speed = float(telemetry["speed"])
        steering = self.smoother.update(self.predict_steering(image))
        # On ralentit avant que le virage ne fasse sortir la voiture.
        self.controller.set_point = target_speed(steering, self.max_speed, self.min_speed)
        throttle = self.controller.update(speed)
        if self.record_dir:
            self._record(image, steering, throttle, speed)
        return steering, throttle

    def _record(self, image, steering, throttle, speed):
        """Sauvegarde l'image reçue et la commande envoyée."""
        stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
        Image.fromarray(image).save(self.record_dir / f"{stamp}.jpg")
        elapsed = time.monotonic() - self.start
        self.log_writer.writerow(
            [f"{elapsed:.3f}", f"{steering:.4f}", f"{throttle:.4f}", f"{speed:.2f}"]
        )
        self._log_file.flush()


def create_server(pilot):
    """Crée le serveur Socket.IO et branche les gestionnaires d'événements."""
    sio = socketio.Server(async_mode="eventlet")

    def send_control(steering, throttle):
        # Le simulateur attend des chaînes de caractères.
        sio.emit("steer", data={"steering_angle": str(steering), "throttle": str(throttle)},
                 skip_sid=True)

    @sio.on("connect")
    def connect(sid, environ):
        print(f"Simulateur connecté ({sid})")
        send_control(0.0, 0.0)

    @sio.on("telemetry")
    def telemetry(sid, data):
        if not data:
            # Pas de données : le simulateur est en mode manuel.
            sio.emit("manual", data={}, skip_sid=True)
            return
        steering, throttle = pilot.step(data)
        print(f"angle {steering:+.3f}  accélérateur {throttle:+.3f}  "
              f"vitesse {mph_to_kmh(float(data['speed'])):5.1f} km/h")
        send_control(steering, throttle)

    @sio.on("disconnect")
    def disconnect(sid):
        print("Simulateur déconnecté")

    return sio


def parse_args():
    """Lit les options de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Conduite autonome dans le simulateur Udacity")
    parser.add_argument("model", help="modèle entraîné (.pt produit par train.py)")
    parser.add_argument("--max-speed", type=float, default=45.0,
                        help="vitesse visée en ligne droite, en km/h (48 au maximum)")
    parser.add_argument("--min-speed", type=float, default=25.0,
                        help="vitesse visée dans les virages serrés, en km/h")
    parser.add_argument("--smoothing", type=float, default=0.3,
                        help="lissage de l'angle entre deux images (0 = aucun)")
    parser.add_argument("--record", default=None,
                        help="dossier où enregistrer images et télémétrie de la conduite")
    parser.add_argument("--port", type=int, default=SIMULATOR_PORT)
    return parser.parse_args()


def main():
    """Charge le modèle puis attend la connexion du simulateur."""
    args = parse_args()
    # On ouvre le port avant de charger le modèle : si un autre drive.py
    # tourne déjà, on le signale tout de suite avec un message clair.
    try:
        listener = eventlet.listen(("", args.port))
    except OSError:
        raise SystemExit(
            f"Le port {args.port} est déjà utilisé, sans doute par un autre drive.py "
            "encore ouvert. Ferme-le (Ctrl+C dans son terminal) puis relance."
        ) from None
    device = default_device()
    model = load_model(args.model, device)
    print(f"Modèle chargé sur {device}")
    pilot = Pilot(model, max_speed=kmh_to_mph(args.max_speed), min_speed=kmh_to_mph(args.min_speed),
                  smoothing=args.smoothing, record_dir=args.record)
    print(f"Vitesse visée : {mph_to_kmh(pilot.max_speed):.0f} km/h en ligne droite, "
          f"{mph_to_kmh(pilot.min_speed):.0f} km/h dans les virages")
    # Un premier appel à vide initialise CUDA avant l'arrivée du simulateur,
    # pour que la première image ne subisse pas de latence.
    pilot.predict_steering(np.zeros((160, 320, 3), dtype=np.uint8))

    app = socketio.WSGIApp(create_server(pilot))
    print(f"En attente du simulateur sur le port {args.port}...")
    eventlet.wsgi.server(listener, app, log_output=False)


if __name__ == "__main__":
    main()
