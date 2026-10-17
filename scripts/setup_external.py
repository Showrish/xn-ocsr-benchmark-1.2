"""One-time setup: download BioMiner, the public BioVista benchmark, and model weights.

    python scripts/setup_external.py                  # clone + biovista + weights
    python scripts/setup_external.py --steps mineru   # after the BioMiner env exists

Steps:
  clone     clone the official BioMiner repo into paths.biominer_repo_dir
  biovista  download the six public BioVista label sets from the authors' Google Drive
  weights   download MolDetv2 and MolGlyph weights from Hugging Face
  teammate  copy teammate Jirapa Bangvak's public DECIMER results (same BioVista
            sample) into paths.teammate_dir for the side-by-side comparison
  mineru    copy MinerU's config to the home folder and download its models
            (runs inside the BioMiner environment, so create that env first)
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

# Make the src package importable when this script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DEFAULT_CONFIG, load_config, setup_logging  # noqa: E402

LOGGER = logging.getLogger("setup")

BIOMINER_GIT_URL = "https://github.com/jiaxianyan/BioMiner.git"

# Public BioVista label files, from the "Statistics and Access of BioVista" table
# in the BioMiner README. The papers themselves are not distributed (copyright).
BIOVISTA_FILES = {
    "bioactivity_triplet_extraction": "14icAlg_jEy3uE7FOcDD-EYn3UIKQxC__",
    "structure_bioactivity_annotation": "1cLC3NePtxctIurnCVaNLw0JzBWYrWTVW",
    "molecule_detection": "1SvsdgrizDqg5V-MGXEPGMmu4JmmhiFae",
    "ocsr": "11iCH0j7U9iFgwXahjkWZouNyqATfEPoL",
    "full_structure_coreference": "14_BREaWgJOIjgVAkCEMwbXKRKsmqpTgu",
    "markush_enumeration": "19WrleiHQY6v6srZEWlKvl-uPlx0wiaLa",
}

# Hugging Face repo -> (file name BioMiner expects in its scripts/ folder, name hint)
MODEL_WEIGHTS = {
    "UniParser/MolDetv2": ("moldet_v2_yolo11n_960_doc.pt", "960"),
    "jiaxianustc/MolGlyph": ("molglyph_large.pt", "large"),
}


def clone_biominer(repo_dir: Path) -> None:
    """Clone BioMiner unless it is already present."""
    if (repo_dir / ".git").exists():
        LOGGER.info("BioMiner already cloned at %s", repo_dir)
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", BIOMINER_GIT_URL, str(repo_dir)], check=True)


def download_biovista(repo_dir: Path) -> None:
    """Download and unpack the BioVista label sets into BioMiner/BioVista, as its README asks."""
    import gdown  # imported here so the other steps do not need it

    target_dir = repo_dir / "BioVista"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, file_id in BIOVISTA_FILES.items():
        LOGGER.info("Downloading BioVista %s", name)
        downloaded = gdown.download(id=file_id, output=str(target_dir) + "/", quiet=False)
        if downloaded is None:
            raise RuntimeError(f"Could not download BioVista file {name} ({file_id})")
        try:
            shutil.unpack_archive(downloaded, target_dir)
            LOGGER.info("Unpacked %s", Path(downloaded).name)
        except (shutil.ReadError, ValueError):
            LOGGER.info("Kept %s as is (not an archive)", Path(downloaded).name)


def download_weights(repo_dir: Path) -> None:
    """Download MolDetv2 and MolGlyph weights and save them under the names BioMiner expects."""
    from huggingface_hub import snapshot_download

    for hf_repo, (target_name, name_hint) in MODEL_WEIGHTS.items():
        target_path = repo_dir / "scripts" / target_name
        if target_path.exists():
            LOGGER.info("%s already present", target_name)
            continue
        snapshot_dir = Path(snapshot_download(repo_id=hf_repo, allow_patterns=["*.pt", "*.pth"]))
        candidates = sorted(snapshot_dir.rglob("*.pt")) + sorted(snapshot_dir.rglob("*.pth"))
        if not candidates:
            raise FileNotFoundError(f"No weight file found in Hugging Face repo {hf_repo}")
        # Prefer the file whose name matches the expected variant
        chosen = next((path for path in candidates if name_hint in path.name), candidates[0])
        shutil.copy(chosen, target_path)
        LOGGER.info("Saved %s as %s", chosen.name, target_path)


def copy_teammate_results(repo_url: str, file_names: list[str], clone_dir: Path, target_dir: Path) -> None:
    """Clone the teammate's public repo and copy her result files, wherever they sit in it."""
    if not (clone_dir / ".git").exists():
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(clone_dir)], check=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    for file_name in file_names:
        matches = sorted(clone_dir.rglob(file_name))
        if not matches:
            LOGGER.warning("%s not found in %s", file_name, repo_url)
            continue
        shutil.copy(matches[0], target_dir / file_name)
        LOGGER.info("Copied %s", matches[0].relative_to(clone_dir))


def setup_mineru(repo_dir: Path, biominer_python: list[str]) -> None:
    """Copy MinerU's config to the home folder and download MinerU's models (BioMiner README)."""
    shutil.copy(repo_dir / "magic-pdf.json", Path.home() / "magic-pdf.json")
    subprocess.run(biominer_python + ["download_models.py"], cwd=repo_dir, check=True)
    LOGGER.info(
        "If MinerU's models failed to download, get them from "
        "https://huggingface.co/jiaxianustc/BioMiner-MinerU-Model (see BioMiner README)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["clone", "biovista", "weights", "teammate", "mineru"],
        default=["clone", "biovista", "weights", "teammate"],
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    repo_dir = config["paths"]["biominer_repo_dir"]

    if "clone" in args.steps:
        clone_biominer(repo_dir)
    if "biovista" in args.steps:
        download_biovista(repo_dir)
    if "weights" in args.steps:
        download_weights(repo_dir)
    if "teammate" in args.steps:
        copy_teammate_results(
            config["teammate"]["decimer_repo"],
            config["teammate"]["files"],
            repo_dir.parent / "teammate-decimer",
            config["paths"]["teammate_dir"],
        )
    if "mineru" in args.steps:
        setup_mineru(repo_dir, list(config["environments"]["biominer"]))


if __name__ == "__main__":
    main()
