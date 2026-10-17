"""Architecture du réseau NVIDIA (Bojarski et al., 2016) en PyTorch.

Référence : « End to End Learning for Self-Driving Cars »,
https://arxiv.org/abs/1604.07316

Le réseau prend une image RGB 66 x 200 et renvoie directement l'angle de
braquage. Les convolutions extraient les bords de route, les couches denses
font office de contrôleur.
"""

from pathlib import Path

import torch
from torch import nn

from sdc.config import INPUT_SHAPE


class PilotNet(nn.Module):
    """Réseau NVIDIA « PilotNet ».

    Structure :

    * normalisation des pixels dans ``[-1, 1]`` (faite dans ``forward``) ;
    * 3 convolutions 5x5 de pas 2 (24, 36, 48 filtres) ;
    * 2 convolutions 3x3 (64, 64 filtres) ;
    * dropout puis couches denses 100, 50, 10 et une sortie linéaire.

    Les activations ELU évitent les neurones morts de ReLU sur une régression
    dont la cible peut être négative. On obtient environ 250 000 poids.

    Le modèle reçoit directement les images ``uint8`` au format
    ``(lot, hauteur, largeur, canaux)`` produites par le prétraitement : la
    conversion et la normalisation font partie du modèle, ce qui garantit
    qu'elles sont identiques à l'entraînement et en conduite.

    Args:
        dropout: taux de dropout avant et après l'aplatissement.
    """

    def __init__(self, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, 3), nn.ELU(),
            nn.Conv2d(64, 64, 3), nn.ELU(),
            nn.Dropout(dropout),
        )
        # Pour une entrée 66 x 200, la sortie des convolutions fait 64 x 1 x 18.
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1 * 18, 100), nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, 1),
        )

    def forward(self, images):
        """Prédit l'angle de braquage.

        Args:
            images: tenseur ``uint8`` (ou flottant dans ``[0, 255]``) de forme
                ``(lot, 66, 200, 3)``.

        Returns:
            Tenseur de forme ``(lot,)`` contenant les angles prédits.
        """
        # (N, H, W, C) -> (N, C, H, W), puis pixels ramenés dans [-1, 1].
        x = images.permute(0, 3, 1, 2).float() / 127.5 - 1.0
        return self.regressor(self.features(x)).squeeze(1)


def count_parameters(model):
    """Nombre de poids entraînables du modèle."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path):
    """Enregistre les poids et les hyperparamètres nécessaires au rechargement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "dropout": model.dropout,
                "input_shape": INPUT_SHAPE}, path)


def load_model(path, device="cpu"):
    """Recharge un modèle enregistré par :func:`save_model`, en mode évaluation."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = PilotNet(dropout=checkpoint["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def default_device():
    """Renvoie le GPU CUDA s'il est disponible, sinon le CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
