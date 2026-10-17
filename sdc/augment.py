"""Augmentation de données pour l'entraînement.

Sur une piste en boucle, la plupart des images ont un angle proche de zéro
et la voiture tourne majoritairement dans le même sens. Sans augmentation,
le réseau apprend à prédire « tout droit » partout. Les fonctions ci-dessous
fabriquent des exemples supplémentaires dont l'angle est cohérent avec
l'image :

* caméras latérales avec une correction d'angle ;
* miroir horizontal avec inversion de l'angle ;
* translation horizontale avec un angle ajusté ;
* variation de luminosité, qui ne change pas l'angle.

Chaque fonction reçoit un générateur ``numpy.random.Generator`` pour que les
tests puissent la rendre déterministe.
"""

import cv2
import numpy as np

from sdc.config import STEERING_CORRECTION, STEERING_MAX, STEERING_MIN

# Ajustement d'angle par pixel de translation horizontale.
STEERING_PER_PIXEL = 0.004


def choose_camera(center, left, right, steering, rng, correction=STEERING_CORRECTION):
    """Tire au hasard l'une des trois caméras et corrige l'angle.

    La caméra de gauche voit la route comme si la voiture était décalée vers
    la gauche : pour revenir au centre il faut braquer à droite, donc on
    ajoute ``correction`` (angle positif = droite dans le simulateur). C'est
    l'inverse pour la caméra de droite.

    Args:
        center, left, right: chemins (ou images) des trois caméras.
        steering: angle enregistré pour la caméra centrale.
        rng: générateur aléatoire numpy.
        correction: valeur ajoutée ou retirée pour les caméras latérales.

    Returns:
        Couple ``(source, angle)`` pour la caméra choisie.
    """
    choice = rng.integers(3)
    if choice == 1:
        return left, steering + correction
    if choice == 2:
        return right, steering - correction
    return center, steering


def random_flip(image, steering, rng):
    """Retourne l'image horizontalement une fois sur deux en inversant l'angle.

    Cela équilibre les virages à gauche et à droite.
    """
    if rng.random() < 0.5:
        return np.ascontiguousarray(image[:, ::-1, :]), -steering
    return image, steering


def random_translate(image, steering, rng, max_shift_x=50, max_shift_y=10):
    """Décale l'image et ajuste l'angle en fonction du décalage horizontal.

    Un décalage vers la droite simule une voiture placée plus à gauche sur la
    route : il faut alors braquer davantage à droite.
    """
    shift_x = rng.uniform(-max_shift_x, max_shift_x)
    shift_y = rng.uniform(-max_shift_y, max_shift_y)
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    height, width = image.shape[:2]
    shifted = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    return shifted, steering + shift_x * STEERING_PER_PIXEL


def random_brightness(image, rng, low=0.5, high=1.3):
    """Multiplie la luminosité (canal V en HSV) par un facteur aléatoire.

    Rend le modèle moins sensible aux ombres et aux zones plus sombres de la
    piste.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(low, high), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def augment(image, steering, rng):
    """Applique la chaîne d'augmentation complète à une image brute RGB.

    Le choix de caméra est fait en amont (voir :func:`choose_camera`) car il
    nécessite de charger un fichier différent.

    Returns:
        Couple ``(image, angle)``, l'angle étant borné à ``[-1, 1]``.
    """
    image, steering = random_flip(image, steering, rng)
    image, steering = random_translate(image, steering, rng)
    image = random_brightness(image, rng)
    return image, float(np.clip(steering, STEERING_MIN, STEERING_MAX))
