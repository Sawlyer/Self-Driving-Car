"""Lecture des enregistrements du simulateur et génération des lots.

Le mode Training du simulateur produit un dossier contenant :

* ``IMG/`` : les images des caméras centre, gauche et droite ;
* ``driving_log.csv`` : une ligne par instant, avec les chemins des trois
  images puis ``steering``, ``throttle``, ``brake`` et ``speed``.

Les chemins écrits par le simulateur sont absolus (par exemple
``C:\\Users\\...\\IMG\\center_...jpg``) et deviennent faux dès que le dossier
est déplacé. On ne garde donc que le nom du fichier et on le cherche dans le
sous-dossier ``IMG`` de l'enregistrement.
"""

from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from sdc.augment import augment, choose_camera
from sdc.config import STEERING_CORRECTION
from sdc.preprocess import load_image, preprocess

COLUMNS = ["center", "left", "right", "steering", "throttle", "brake", "speed"]


def _resolve(image_path, image_dir):
    """Rattache un chemin écrit par le simulateur au dossier ``IMG`` local."""
    # PureWindowsPath comprend à la fois les « \\ » et les « / ».
    return str(image_dir / PureWindowsPath(image_path.strip()).name)


def load_driving_log(data_dir):
    """Charge le ``driving_log.csv`` d'un enregistrement.

    Gère les deux formats rencontrés : les fichiers du simulateur (sans
    en-tête, chemins absolus) et le jeu d'exemple Udacity (avec en-tête,
    chemins relatifs précédés d'espaces).

    Args:
        data_dir: dossier contenant ``driving_log.csv`` et ``IMG/``.

    Returns:
        ``DataFrame`` avec les colonnes ``center``, ``left``, ``right``
        (chemins locaux) et ``steering`` (float).
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / "driving_log.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"driving_log.csv introuvable dans {data_dir}")

    frame = pd.read_csv(csv_path, header=None, names=COLUMNS, skipinitialspace=True)
    # Le jeu d'exemple Udacity commence par une ligne d'en-tête : on la retire.
    if str(frame.iloc[0]["center"]).strip().lower() == "center":
        frame = frame.iloc[1:].reset_index(drop=True)

    image_dir = data_dir / "IMG"
    for camera in ("center", "left", "right"):
        frame[camera] = frame[camera].astype(str).map(lambda p: _resolve(p, image_dir))
    frame["steering"] = frame["steering"].astype(float)
    return frame[["center", "left", "right", "steering"]]


def load_samples(data_dirs, keep_straight=1.0, seed=0):
    """Charge et concatène plusieurs enregistrements.

    Args:
        data_dirs: liste de dossiers d'enregistrement.
        keep_straight: proportion des lignes en ligne droite (angle quasi nul)
            à conserver. Une valeur inférieure à 1 réduit le biais vers
            l'angle zéro, très majoritaire sur la piste 1.
        seed: graine du tirage des lignes conservées.

    Returns:
        ``DataFrame`` de tous les échantillons retenus.
    """
    frame = pd.concat([load_driving_log(d) for d in data_dirs], ignore_index=True)
    if keep_straight < 1.0:
        rng = np.random.default_rng(seed)
        straight = frame["steering"].abs() < 0.01
        keep = ~straight | (rng.random(len(frame)) < keep_straight)
        frame = frame[keep].reset_index(drop=True)
    return frame


def split_samples(frame, val_split=0.2, seed=0):
    """Sépare les échantillons en entraînement et validation."""
    return train_test_split(frame, test_size=val_split, random_state=seed, shuffle=True)


class DrivingDataset(Dataset):
    """Jeu de données PyTorch construisant les exemples à la volée.

    Les images ne tiennent pas toutes en mémoire une fois augmentées : on les
    charge et on les transforme au moment où le ``DataLoader`` les demande.

    En entraînement, chaque échantillon passe par le choix aléatoire de
    caméra et la chaîne d'augmentation. En validation, on utilise uniquement
    la caméra centrale sans transformation, pour mesurer l'erreur dans les
    conditions de la conduite réelle.

    Args:
        frame: échantillons issus de :func:`load_samples`.
        training: active le choix de caméra et l'augmentation.
        correction: correction d'angle des caméras latérales.
        seed: graine de base du tirage aléatoire.
    """

    def __init__(self, frame, training=True, correction=STEERING_CORRECTION, seed=0):
        self.frame = frame.reset_index(drop=True)
        self.training = training
        self.correction = correction
        self.seed = seed
        self._rng = None

    def __len__(self):
        return len(self.frame)

    @property
    def rng(self):
        """Générateur aléatoire propre à chaque processus de chargement.

        Les workers du ``DataLoader`` sont des copies du jeu de données : sans
        graine distincte, ils produiraient tous les mêmes augmentations.
        """
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            worker_seed = info.seed if info is not None else self.seed
            self._rng = np.random.default_rng(worker_seed % 2**32)
        return self._rng

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        if not self.training:
            image, steering = load_image(row["center"]), row["steering"]
        else:
            path, steering = choose_camera(
                row["center"], row["left"], row["right"], row["steering"], self.rng, self.correction
            )
            image, steering = augment(load_image(path), steering, self.rng)
        return torch.from_numpy(preprocess(image)), torch.tensor(steering, dtype=torch.float32)
