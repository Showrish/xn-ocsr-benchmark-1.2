"""Phase 2, stage 3: recognition (3a) and the Markush branch (3b).

Reads the gated manifest from stage 2 (molecule detection and gate) and writes the
two hand-off files agreed in the Assignment 1.4 plan (Table 3). Stage 4
(validation and combination) reads both.

  predictions_molscribe.csv   3a: one row per crop labelled "molecule"
      paper_id, page, crop_id, tool, predicted_smiles, confidence, runtime_sec, error
  markush_compounds.csv       3b: one row per crop labelled "markush"
      paper_id, page, crop_id, compound_label, smiles, method

Crops labelled "not_molecule" are skipped. Until stage 2's gate is ready,
scripts/make_stand_in_manifest.py builds a manifest from the DECIMER crops.

Version 1 of the Markush branch keeps the scaffold, turns R-group placeholders into
valid attachment points, and flags the crop. Expanding R-group tables into full
compounds is the next step.

Runs inside the xn-molscribe environment (envs/molscribe.yml).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_CONFIG, load_config, resolve_path, setup_logging
from src.markush import attachment_points, normalize_attachment_points
from src.run_molscribe import TOOL_NAME, load_model, pick_device, predict_image

LOGGER = logging.getLogger("stage3")

REQUIRED_COLUMNS = {"paper_id", "page", "crop_id", "crop_path", "label"}
PREDICTION_COLUMNS = ["paper_id", "page", "crop_id", "tool", "predicted_smiles", "confidence", "runtime_sec", "error"]
MARKUSH_COLUMNS = ["paper_id", "page", "crop_id", "compound_label", "smiles", "method"]


def load_gated_manifest(path: Path) -> pd.DataFrame:
    """Read stage 2's manifest and check it has the agreed columns."""
    manifest = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"{path.name} is missing the columns {sorted(missing)}")
    # Crop paths may be written relative to the manifest's own folder
    manifest["crop_path"] = [
        p if Path(p).is_absolute() else str(path.parent / p) for p in manifest["crop_path"].astype(str)
    ]
    manifest["label"] = manifest["label"].astype(str).str.strip().str.lower()
    LOGGER.info("Gated crops by label: %s", manifest["label"].value_counts().to_dict())
    return manifest


def run_stage3(manifest: pd.DataFrame, model, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recognize molecule crops, start the Markush branch, and write both hand-off files."""
    predictions, markush = [], []
    for row in manifest.itertuples(index=False):
        if row.label not in ("molecule", "markush"):
            continue  # not_molecule: dropped by the gate
        ids = {"paper_id": row.paper_id, "page": row.page, "crop_id": row.crop_id}
        result = predict_image(model, Path(row.crop_path))
        if row.label == "molecule":
            predictions.append({**ids, "tool": TOOL_NAME, **result})
        else:
            scaffold = normalize_attachment_points(result["predicted_smiles"])
            method = "scaffold_flagged" if attachment_points(scaffold) else "flagged_no_attachment_point"
            markush.append({**ids, "compound_label": "", "smiles": scaffold, "method": method})

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(predictions, columns=PREDICTION_COLUMNS)
    markush = pd.DataFrame(markush, columns=MARKUSH_COLUMNS)
    predictions.to_csv(output_dir / "predictions_molscribe.csv", index=False)
    markush.to_csv(output_dir / "markush_compounds.csv", index=False)
    LOGGER.info("Wrote %d predictions and %d Markush rows to %s", len(predictions), len(markush), output_dir)
    return predictions, markush


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml")
    parser.add_argument("--manifest", help="Gated manifest to use instead of paths.gated_manifest")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    paths, settings = config["paths"], config["molscribe"]
    manifest_path = resolve_path(args.manifest) if args.manifest else paths["gated_manifest"]
    manifest = load_gated_manifest(manifest_path)
    model = load_model(settings["hf_repo"], settings["checkpoint"], pick_device(settings["device"]))
    run_stage3(manifest, model, paths["stage3_dir"])


if __name__ == "__main__":
    main()
