"""Pipeline step: run MolScribe on molecule images.

Two datasets:
  biovista_ocsr  The public BioVista OCSR images, using the same 100-image sample as
                 Jirapa's DECIMER benchmark, so the two tools read identical images.
                 Output: results/biovista_ocsr/predictions_molscribe.csv
  course         DECIMER crops of the six course papers (Jirapa's segmentation).
                 Output: results/predictions_molscribe.csv in the shared schema

Runs inside the xn-molscribe environment (envs/molscribe.yml).
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from src import EXTRA_COLUMNS, PREDICTION_COLUMNS
from src.biovista_ocsr import load_records, select_sample
from src.config import DEFAULT_CONFIG, load_config, resolve_path, setup_logging
from src.data import load_crops

LOGGER = logging.getLogger("run_molscribe")
TOOL_NAME = "MolScribe"


def pick_device(setting: str) -> torch.device:
    """Use the GPU when one is available, unless the config forces a device."""
    if setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(setting)


def load_model(hf_repo: str, checkpoint: str, device: torch.device) -> MolScribe:
    """Download the pretrained checkpoint from Hugging Face (cached after the first run)."""
    # MolScribe imports matplotlib when it loads. Notebooks such as Colab set a
    # notebook-only plotting backend that this environment does not have, which
    # breaks that import, so switch to the headless backend first.
    os.environ["MPLBACKEND"] = "Agg"
    from molscribe import MolScribe

    checkpoint_path = hf_hub_download(repo_id=hf_repo, filename=checkpoint)
    LOGGER.info("Loaded checkpoint %s on %s", checkpoint, device)
    return MolScribe(checkpoint_path, device=device)


def predict_image(model: MolScribe, image_path: Path) -> dict:
    """Predict one image. Returns SMILES, confidence, runtime, and any error message."""
    start = time.perf_counter()
    try:
        output = model.predict_image_file(str(image_path), return_confidence=True)
        smiles, confidence, error = output.get("smiles", ""), output.get("confidence"), ""
    except Exception as exc:  # one unreadable image should not stop the whole run
        smiles, confidence, error = "", None, f"{type(exc).__name__}: {exc}"
        LOGGER.warning("Failed on %s: %s", image_path.name, error)
    return {
        "predicted_smiles": smiles,
        "confidence": confidence,
        "runtime_sec": round(time.perf_counter() - start, 3),
        "error": error,
    }


def run_biovista_ocsr(model: MolScribe, config: dict) -> None:
    """Benchmark MolScribe on the BioVista OCSR sample and save one row per image."""
    paths, settings = config["paths"], config["biovista_ocsr"]
    records = load_records(paths["biovista_ocsr_dir"], settings["labels_file"], settings.get("smiles_column") or "smiles")
    sample_ids_file = resolve_path(settings["sample_ids_file"]) if settings.get("sample_ids_file") else None
    sample = select_sample(records, settings["n_samples"], settings["seed"], sample_ids_file)

    rows = []
    for index, record in enumerate(sample.itertuples(index=False), start=1):
        result = predict_image(model, Path(record.image_path))
        rows.append({"sample_id": record.sample_id, "group": record.group, "tool": TOOL_NAME, **result})
        if index % 25 == 0:
            LOGGER.info("Processed %d of %d images", index, len(sample))

    output_dir = paths["results_dir"] / "biovista_ocsr"
    output_dir.mkdir(parents=True, exist_ok=True)
    # The manifest records exactly which images and answers were used, for scoring and reproducibility
    sample.assign(image_file=sample["image_path"].map(lambda p: Path(p).name)).drop(columns="image_path").to_csv(
        output_dir / "sample_manifest.csv", index=False
    )
    columns = ["sample_id", "group", "tool", "predicted_smiles", "confidence", "runtime_sec", "error"]
    pd.DataFrame(rows, columns=columns).to_csv(output_dir / "predictions_molscribe.csv", index=False)
    LOGGER.info("Saved %d predictions to %s", len(rows), output_dir / "predictions_molscribe.csv")


def run_course_crops(model: MolScribe, config: dict) -> None:
    """Run MolScribe on the DECIMER crops of the course papers, in the shared schema."""
    paths, crop_settings = config["paths"], config["crops"]
    crops = load_crops(
        crops_dir=paths["decimer_crops_dir"],
        paper_ids=config["papers"],
        filename_pattern=crop_settings["filename_pattern"],
        extensions=crop_settings["extensions"],
        manifest_csv=crop_settings.get("manifest_csv") or None,
    )
    if not crops:
        raise SystemExit("No crops found. Check paths.decimer_crops_dir and crops.filename_pattern.")

    rows = []
    for index, crop in enumerate(crops, start=1):
        result = predict_image(model, crop.image_path)
        rows.append(
            {
                "paper_id": crop.paper_id,
                "page": crop.page,
                "crop_id": crop.crop_id,
                "tool": TOOL_NAME,
                "label": "",
                "source": crop.image_path.name,
                **result,
            }
        )
        if index % 25 == 0:
            LOGGER.info("Processed %d of %d crops", index, len(crops))

    output_path = paths["results_dir"] / "predictions_molscribe.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=PREDICTION_COLUMNS + EXTRA_COLUMNS).to_csv(output_path, index=False)
    LOGGER.info("Saved %d predictions to %s", len(rows), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml")
    parser.add_argument("--dataset", choices=["biovista_ocsr", "course"], required=True)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    settings = config["molscribe"]
    model = load_model(settings["hf_repo"], settings["checkpoint"], pick_device(settings["device"]))

    if args.dataset == "biovista_ocsr":
        run_biovista_ocsr(model, config)
    else:
        run_course_crops(model, config)


if __name__ == "__main__":
    main()
