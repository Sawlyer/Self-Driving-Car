"""Outils d'évaluation hors simulateur.

Deux commandes :

* ``offline`` compare les angles prédits aux angles enregistrés par un
  conducteur humain sur un enregistrement (caméra centrale uniquement) et
  trace les deux courbes ;
* ``autonomy`` calcule le score d'autonomie NVIDIA d'un tour en mode
  autonome.

Exemples ::

    python evaluate.py offline models/model.pt --data-dir data/mes_tours
    python evaluate.py autonomy --interventions 1 --duration 75

Le vrai test reste un tour complet dans le simulateur : une faible erreur
hors ligne ne garantit pas que la voiture reste sur la route, car les
petites erreurs s'accumulent en boucle fermée.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from sdc.dataset import DrivingDataset, load_samples  # noqa: E402
from sdc.metrics import autonomy, regression_report  # noqa: E402
from sdc.model import default_device, load_model  # noqa: E402


def evaluate_offline(model_path, data_dirs, plot_path, batch_size=256, workers=4):
    """Prédit l'angle pour chaque image centrale et compare à la vérité terrain.

    Returns:
        Dictionnaire ``mse``, ``mae``, ``rmse``.
    """
    device = default_device()
    model = load_model(model_path, device)
    frame = load_samples(data_dirs)
    loader = DataLoader(DrivingDataset(frame, training=False), batch_size=batch_size,
                        num_workers=workers)
    with torch.inference_mode():
        predictions = torch.cat([model(images.to(device)).cpu() for images, _ in loader]).numpy()
    report = regression_report(frame["steering"], predictions)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(frame["steering"].to_numpy(), label="humain", linewidth=0.8)
    ax.plot(predictions, label="modèle", linewidth=0.8, alpha=0.8)
    ax.set_xlabel("image")
    ax.set_ylabel("angle de braquage")
    ax.set_title(f"Angles prédits et enregistrés (MAE {report['mae']:.3f})")
    ax.legend()
    fig.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    return report


def parse_args():
    """Lit les options de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Évaluation du modèle de conduite")
    commands = parser.add_subparsers(dest="command", required=True)

    offline = commands.add_parser("offline", help="erreur sur un enregistrement")
    offline.add_argument("model")
    offline.add_argument("--data-dir", nargs="+", default=["data/udacity_sample"])
    offline.add_argument("--plot", default="models/predictions.png")

    score = commands.add_parser("autonomy", help="score d'autonomie NVIDIA")
    score.add_argument("--interventions", type=int, required=True)
    score.add_argument("--duration", type=float, required=True, help="durée du tour en secondes")
    return parser.parse_args()


def main():
    """Point d'entrée."""
    args = parse_args()
    if args.command == "offline":
        report = evaluate_offline(args.model, args.data_dir, args.plot)
        print("  ".join(f"{name.upper()} {value:.4f}" for name, value in report.items()))
        print(f"Courbe enregistrée dans {args.plot}")
    else:
        print(f"Autonomie : {autonomy(args.interventions, args.duration):.1f} %")


if __name__ == "__main__":
    main()
