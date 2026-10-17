"""Constantes partagées entre l'entraînement et la conduite autonome.

Tout ce qui touche à la forme des images vit ici : ``train.py`` et
``drive.py`` doivent appliquer exactement le même prétraitement, sinon le
modèle reçoit en conduite des images qu'il n'a jamais vues.
"""

# Taille des images brutes produites par le simulateur Udacity.
RAW_HEIGHT, RAW_WIDTH = 160, 320

# Lignes retirées en haut (ciel, arbres) et en bas (capot de la voiture).
CROP_TOP, CROP_BOTTOM = 60, 25

# Entrée du réseau NVIDIA : 66 x 200 pixels, 3 canaux RGB.
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

# Correction d'angle appliquée aux caméras latérales : l'image de gauche
# montre la route comme si la voiture était décalée vers la gauche, il faut
# donc braquer un peu plus à droite pour revenir au centre (et inversement).
STEERING_CORRECTION = 0.2

# L'angle de braquage du simulateur est normalisé dans [-1, 1].
STEERING_MIN, STEERING_MAX = -1.0, 1.0

# Port sur lequel le simulateur cherche le serveur en mode autonome.
SIMULATOR_PORT = 4567
