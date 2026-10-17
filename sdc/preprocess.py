"""Prétraitement commun à l'entraînement et à la conduite.

Les images sont toujours manipulées en RGB, en ``uint8``. La normalisation
des pixels est faite dans le modèle (couche ``Rescaling``), ce qui garantit
qu'elle est identique à l'entraînement et en conduite.
"""

import cv2
import numpy as np

from sdc.config import CROP_BOTTOM, CROP_TOP, IMAGE_HEIGHT, IMAGE_WIDTH


def load_image(path):
    """Charge une image depuis le disque et la renvoie en RGB.

    OpenCV lit en BGR alors que le simulateur envoie du RGB en mode
    autonome : on convertit ici pour que les deux chemins coïncident.

    Args:
        path: chemin du fichier image.

    Returns:
        Tableau ``uint8`` de forme ``(hauteur, largeur, 3)`` en RGB.

    Raises:
        FileNotFoundError: si l'image est introuvable ou illisible.
    """
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Image introuvable ou illisible : {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def crop(image):
    """Retire le ciel en haut de l'image et le capot en bas."""
    return image[CROP_TOP : image.shape[0] - CROP_BOTTOM, :, :]


def resize(image):
    """Redimensionne l'image à la taille d'entrée du réseau (66 x 200)."""
    # cv2.resize attend (largeur, hauteur) et non (hauteur, largeur).
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)


def preprocess(image):
    """Applique la chaîne complète : recadrage puis redimensionnement.

    Args:
        image: image RGB brute du simulateur, de forme ``(160, 320, 3)``.

    Returns:
        Image RGB ``uint8`` de forme ``(66, 200, 3)``.
    """
    return resize(crop(np.asarray(image)))
