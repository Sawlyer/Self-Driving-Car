"""Métriques d'évaluation de la conduite."""

import numpy as np

# Temps compté par NVIDIA pour chaque intervention humaine : reprendre la
# main, recentrer la voiture puis rendre le contrôle.
SECONDS_PER_INTERVENTION = 6.0


def autonomy(interventions, elapsed_seconds):
    """Calcule le score d'autonomie défini par NVIDIA.

    ``autonomie = (1 - interventions * 6 / durée) * 100``

    Un tour complet sans intervention vaut 100 %. Le résultat est borné à 0
    quand les interventions « coûtent » plus que la durée du tour.

    Args:
        interventions: nombre de fois où il a fallu reprendre la main.
        elapsed_seconds: durée du tour en secondes.

    Returns:
        Pourcentage d'autonomie entre 0 et 100.
    """
    if elapsed_seconds <= 0:
        raise ValueError("La durée doit être strictement positive.")
    if interventions < 0:
        raise ValueError("Le nombre d'interventions ne peut pas être négatif.")
    score = (1.0 - interventions * SECONDS_PER_INTERVENTION / elapsed_seconds) * 100.0
    return max(score, 0.0)


def regression_report(y_true, y_pred):
    """Renvoie MSE, MAE et RMSE entre angles réels et prédits."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    errors = y_pred - y_true
    mse = float(np.mean(errors**2))
    return {"mse": mse, "mae": float(np.mean(np.abs(errors))), "rmse": mse**0.5}
