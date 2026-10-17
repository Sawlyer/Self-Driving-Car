"""Régulation de la vitesse en mode autonome.

Le réseau ne prédit que l'angle de braquage. L'accélération est confiée à un
contrôleur proportionnel-intégral (PI) qui maintient une vitesse de
consigne. Cette consigne baisse dans les virages, comme le ferait un
conducteur : plus la voiture va vite, plus une erreur de braquage l'emmène
loin du centre avant que le modèle ne la corrige.

Le simulateur exprime la vitesse en miles par heure et la plafonne à
30 mph (environ 48 km/h).
"""

KMH_PER_MPH = 1.609344

# Vitesse maximale atteignable dans le simulateur Udacity, en mph.
SIMULATOR_MAX_SPEED_MPH = 30.0


def kmh_to_mph(speed_kmh):
    """Convertit une vitesse de km/h en mph (unité du simulateur)."""
    return speed_kmh / KMH_PER_MPH


def mph_to_kmh(speed_mph):
    """Convertit une vitesse de mph (unité du simulateur) en km/h."""
    return speed_mph * KMH_PER_MPH


def target_speed(steering, max_speed, min_speed, full_turn=0.3):
    """Vitesse de consigne en fonction de l'angle de braquage.

    En ligne droite on vise ``max_speed`` ; la consigne baisse linéairement
    avec ``|steering|`` jusqu'à ``min_speed``, atteinte pour un braquage de
    ``full_turn`` ou plus.
    """
    ratio = min(abs(steering) / full_turn, 1.0)
    return max_speed - (max_speed - min_speed) * ratio


class SteeringSmoother:
    """Lissage exponentiel de l'angle de braquage.

    À vitesse élevée, les petites oscillations d'une image à l'autre font
    zigzaguer la voiture. ``alpha`` est le poids de l'ancienne valeur :
    0 désactive le lissage, 0.5 fait une moyenne avec l'angle précédent.
    """

    def __init__(self, alpha=0.3):
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha doit être dans [0, 1[.")
        self.alpha = alpha
        self.value = None

    def update(self, steering):
        """Renvoie l'angle lissé après ajout de la nouvelle prédiction."""
        if self.value is None:
            self.value = steering
        else:
            self.value = self.alpha * self.value + (1.0 - self.alpha) * steering
        return self.value


class PIController:
    """Contrôleur PI simple.

    ``commande = kp * erreur + ki * somme(erreurs)``, avec
    ``erreur = consigne - mesure``.

    Args:
        kp: gain proportionnel.
        ki: gain intégral.
        set_point: vitesse visée (unités du simulateur, environ mph).
        output_limits: bornes de la commande renvoyée.
    """

    def __init__(self, kp=0.1, ki=0.002, set_point=9.0, output_limits=(-1.0, 1.0)):
        self.kp = kp
        self.ki = ki
        self.set_point = float(set_point)
        self.low, self.high = output_limits
        self.integral = 0.0

    def update(self, measurement):
        """Calcule la commande d'accélérateur pour la vitesse mesurée."""
        error = self.set_point - float(measurement)
        self.integral += error
        output = self.kp * error + self.ki * self.integral
        if not self.low <= output <= self.high:
            # Anti-emballement : on n'accumule pas l'erreur quand la commande
            # est saturée, sinon l'intégrale explose dans les montées.
            self.integral -= error
            output = min(max(output, self.low), self.high)
        return output
