"""Télécharge le simulateur Udacity et le jeu de données d'exemple.

Usage ::

    python scripts/download.py simulator   # simulateur Windows, Linux ou macOS
    python scripts/download.py data        # jeu d'exemple Udacity (piste 1)
    python scripts/download.py all

Les archives sont extraites dans ``simulator/`` et ``data/udacity_sample/``,
deux dossiers ignorés par git. Un élément déjà présent n'est pas
retéléchargé.
"""

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Version 2 du simulateur « Term 1 », publiée par Udacity sur S3.
SIMULATOR_URLS = {
    "win32": "https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-windows.zip",
    "linux": "https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-linux.zip",
    "darwin": "https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-mac.zip",
}
SAMPLE_DATA_URL = "https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip"


def download(url, destination):
    """Télécharge ``url`` vers ``destination`` en affichant la progression."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    def progress(blocks, block_size, total):
        if total > 0:
            done = min(blocks * block_size / total, 1.0)
            print(f"\r  {done:6.1%} de {total / 1e6:.0f} Mo", end="", flush=True)

    print(f"Téléchargement de {url}")
    urllib.request.urlretrieve(url, destination, progress)
    print()


def extract(archive, target):
    """Extrait une archive zip en ignorant les métadonnées macOS."""
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.namelist() if not m.startswith("__MACOSX")]
        zf.extractall(target, members)


def get_simulator():
    """Installe le simulateur correspondant au système courant."""
    target = ROOT / "simulator"
    if any(target.glob("*/beta_simulator*")):
        print(f"Simulateur déjà présent dans {target}")
        return
    platform = "linux" if sys.platform.startswith("linux") else sys.platform
    url = SIMULATOR_URLS.get(platform)
    if url is None:
        raise SystemExit(f"Pas de build du simulateur pour la plateforme {sys.platform}")
    archive = target / Path(url).name
    download(url, archive)
    extract(archive, target)
    archive.unlink()
    print(f"Simulateur extrait dans {target}")


def get_sample_data():
    """Installe le jeu d'exemple Udacity dans ``data/udacity_sample``."""
    target = ROOT / "data" / "udacity_sample"
    if (target / "driving_log.csv").exists():
        print(f"Jeu d'exemple déjà présent dans {target}")
        return
    archive = ROOT / "data" / "udacity_sample.zip"
    download(SAMPLE_DATA_URL, archive)
    extract(archive, ROOT / "data")
    # L'archive contient un dossier nommé simplement « data ».
    shutil.move(str(ROOT / "data" / "data"), str(target))
    archive.unlink()
    print(f"Jeu d'exemple extrait dans {target}")


def main():
    """Point d'entrée."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["simulator", "data", "all"])
    args = parser.parse_args()
    if args.what in ("simulator", "all"):
        get_simulator()
    if args.what in ("data", "all"):
        get_sample_data()


if __name__ == "__main__":
    main()
