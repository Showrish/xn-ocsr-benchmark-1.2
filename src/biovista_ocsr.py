"""Load the public BioVista OCSR set: cropped molecule images with their correct SMILES.

The sample matches teammate Jirapa Bangvak's DECIMER benchmark (100 images, seed 42),
so DECIMER and MolScribe are compared on identical images, as planned in our 1.1
proposal. When her per-image file is present, its sample IDs fix the exact images.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
ID_HINTS = ("file", "image", "img", "name", "path", "id")


def structure_group(smiles: str) -> str:
    """Approximate BioVista's Full / Chiral / Markush groups from the SMILES text.

    Markush structures contain a variable group (*), chiral ones contain @.
    This is the same approximation Jirapa used, so the groups line up.
    """
    if "*" in smiles:
        return "markush"
    if "@" in smiles:
        return "chiral"
    return "full"


def _detect_columns(labels: pd.DataFrame, id_column: str, smiles_column: str) -> tuple[str, str]:
    """Find the SMILES and image ID columns, unless the config names them."""
    if not smiles_column:
        smiles_column = next((c for c in labels.columns if "smi" in str(c).lower()), "")
    if not id_column:
        id_column = next(
            (c for c in labels.columns if c != smiles_column and any(h in str(c).lower() for h in ID_HINTS)),
            next((c for c in labels.columns if c != smiles_column), ""),
        )
    if not smiles_column or not id_column:
        raise ValueError(
            f"Could not detect columns in {list(labels.columns)}. "
            "Set biovista_ocsr.id_column and biovista_ocsr.smiles_column in config.yaml."
        )
    LOGGER.info("Using '%s' as the image column and '%s' as the SMILES column", id_column, smiles_column)
    return id_column, smiles_column


def load_records(ocsr_dir: Path, labels_file: str, id_column: str = "", smiles_column: str = "") -> pd.DataFrame:
    """Return one row per labelled image: sample_id, smiles, group, image_path."""
    labels = pd.read_csv(ocsr_dir / labels_file)
    id_column, smiles_column = _detect_columns(labels, id_column, smiles_column)

    # Index every image under the OCSR folder by file name without extension
    images = {path.stem: path for path in ocsr_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES}

    records = pd.DataFrame(
        {
            "sample_id": labels[id_column].astype(str).map(lambda value: Path(value).stem),
            "smiles": labels[smiles_column].astype(str),
        }
    )
    records["image_path"] = records["sample_id"].map(images)
    missing = records["image_path"].isna().sum()
    if missing:
        LOGGER.warning("%d labelled images were not found on disk and are skipped", missing)
    records = records.dropna(subset=["image_path"]).reset_index(drop=True)
    records["group"] = records["smiles"].map(structure_group)
    LOGGER.info("Loaded %d BioVista OCSR images", len(records))
    return records


def select_sample(records: pd.DataFrame, n_samples: int, seed: int, sample_ids_file: Path | None = None) -> pd.DataFrame:
    """Pick the benchmark sample. Uses the teammate's sample IDs when available."""
    if sample_ids_file and Path(sample_ids_file).exists():
        teammate_ids = pd.read_csv(sample_ids_file)["sample_id"].astype(str).map(lambda v: Path(v).stem)
        sample = records[records["sample_id"].isin(set(teammate_ids))]
        if sample.empty and teammate_ids.str.isdigit().all():
            # The IDs may be row positions in the label file rather than file names
            sample = records.iloc[[int(i) for i in teammate_ids if int(i) < len(records)]]
        if not sample.empty:
            LOGGER.info("Using the %d images from %s", len(sample), Path(sample_ids_file).name)
            return sample.reset_index(drop=True)
        LOGGER.warning("No sample IDs from %s matched the labels, drawing a seeded sample instead", sample_ids_file)

    if n_samples <= 0 or n_samples >= len(records):
        return records
    LOGGER.info("Drawing %d random images with seed %d", n_samples, seed)
    return records.sample(n=n_samples, random_state=seed).reset_index(drop=True)
