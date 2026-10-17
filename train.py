"""Entraîne le réseau NVIDIA sur un ou plusieurs enregistrements du simulateur.

Exemple ::

    python train.py --data-dir data/udacity_sample data/mes_tours --epochs 30

L'entraînement se fait sur le GPU CUDA s'il est disponible. Le meilleur
modèle (plus faible perte de validation) est enregistré dans
``models/model.pt``, accompagné de l'historique ``history.csv`` et de la
courbe ``loss.png``.
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Pas de fenêtre : on écrit directement un PNG.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from sdc.config import STEERING_CORRECTION  # noqa: E402
from sdc.dataset import DrivingDataset, load_samples, split_samples  # noqa: E402
from sdc.model import PilotNet, count_parameters, default_device, save_model  # noqa: E402


def parse_args():
    """Lit les options de la ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", nargs="+", default=["data/udacity_sample"],
                        help="dossiers contenant driving_log.csv et IMG/")
    parser.add_argument("--output", default="models/model.pt", help="chemin du modèle produit")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="part des données gardée pour la validation (0.1 à 0.2)")
    parser.add_argument("--correction", type=float, default=STEERING_CORRECTION,
                        help="correction d'angle des caméras latérales")
    parser.add_argument("--keep-straight", type=float, default=0.5,
                        help="proportion des images en ligne droite conservées")
    parser.add_argument("--repeat", type=int, default=3,
                        help="nombre de passages augmentés sur chaque image par époque")
    parser.add_argument("--patience", type=int, default=8,
                        help="époques sans amélioration avant arrêt anticipé")
    parser.add_argument("--workers", type=int, default=8, help="processus de chargement des images")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def run_epoch(model, loader, device, optimizer=None):
    """Parcourt un ``DataLoader`` une fois et renvoie ``(MSE, MAE)`` moyennes.

    Avec un optimiseur, le modèle est entraîné ; sans, il est seulement
    évalué (dropout désactivé, pas de gradient).
    """
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.MSELoss()
    total_mse = total_mae = count = 0.0
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            predictions = model(images)
            loss = loss_fn(predictions, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_mse += loss.item() * len(targets)
            total_mae += (predictions - targets).abs().sum().item()
            count += len(targets)
    return total_mse / count, total_mae / count


def plot_history(history, path):
    """Trace les pertes d'entraînement et de validation par époque.

    Une perte de validation qui remonte pendant que celle d'entraînement
    continue de baisser signale un surapprentissage.
    """
    epochs = range(1, len(history["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["loss"], label="entraînement")
    ax.plot(epochs, history["val_loss"], label="validation")
    ax.set_xlabel("époque")
    ax.set_ylabel("MSE")
    ax.set_title("Perte du modèle")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    """Charge les données, entraîne le modèle et sauvegarde les résultats."""
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = default_device()
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Entraînement sur {name}")

    samples = load_samples(args.data_dir, keep_straight=args.keep_straight, seed=args.seed)
    train_frame, val_frame = split_samples(samples, args.val_split, args.seed)
    # Chaque image est vue plusieurs fois par époque, avec une augmentation
    # différente à chaque passage.
    train_frame = pd.concat([train_frame] * args.repeat, ignore_index=True)
    print(f"{len(samples)} échantillons : {len(train_frame)} en entraînement "
          f"(x{args.repeat}), {len(val_frame)} en validation")

    loader_options = {"batch_size": args.batch_size, "num_workers": args.workers,
                      "pin_memory": device.type == "cuda",
                      "persistent_workers": args.workers > 0}
    train_loader = DataLoader(DrivingDataset(train_frame, training=True, correction=args.correction,
                                             seed=args.seed), shuffle=True, **loader_options)
    val_loader = DataLoader(DrivingDataset(val_frame, training=False), shuffle=False,
                            **loader_options)

    model = PilotNet(dropout=args.dropout).to(device)
    print(f"Modèle PilotNet : {count_parameters(model):,} poids".replace(",", " "))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    history = {"loss": [], "mae": [], "val_loss": [], "val_mae": [], "lr": []}
    best_val, epochs_without_progress = float("inf"), 0

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        loss, mae = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_mae = run_epoch(model, val_loader, device)
        scheduler.step(val_loss)
        for key, value in zip(history, (loss, mae, val_loss, val_mae,
                                         optimizer.param_groups[0]["lr"]), strict=True):
            history[key].append(value)

        improved = val_loss < best_val
        if improved:
            # On ne garde que le modèle qui généralise le mieux.
            best_val, epochs_without_progress = val_loss, 0
            save_model(model, output)
        else:
            epochs_without_progress += 1
        print(f"Époque {epoch:3d}/{args.epochs}  perte {loss:.4f}  val {val_loss:.4f}  "
              f"val MAE {val_mae:.4f}  ({time.perf_counter() - start:.0f} s)"
              f"{'  * sauvegardé' if improved else ''}", flush=True)
        if epochs_without_progress >= args.patience:
            print(f"Arrêt anticipé : pas d'amélioration depuis {args.patience} époques.")
            break

    with open(output.with_name("history.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *history])
        rows = zip(*history.values(), strict=True)
        writer.writerows([epoch, *values] for epoch, values in enumerate(rows, start=1))
    plot_history(history, output.with_name("loss.png"))
    print(f"Meilleure perte de validation : {best_val:.4f}. Modèle enregistré dans {output}")


if __name__ == "__main__":
    main()
