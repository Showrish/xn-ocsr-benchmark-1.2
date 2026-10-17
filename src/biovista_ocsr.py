"""Load the public BioVista OCSR set: cropped molecule images with their correct SMILES.

Images are matched to their labels with the BioMiner authors' own rule, taken from
BioVista/component_infernece/ocsr/molscribe_inference_batch.py in their repo: an image
named <pdb_idx>_<pdb_name>_..._<page>-<index>.png belongs to the label row whose
pdf_name is <pdb_idx>_<pdb_name> and whose index is <index>.

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


def _stem(value) -> str:
    """File name without folder or extension, so 'a/b/img_1.png' and 'img_1' compare equal."""
    return Path(str(value)).stem


def _image_key(stem: str) -> tuple[str, int] | None:
    """(pdf_name, index) for a BioVista image name, using the authors' parsing rule."""
    items = stem.split("_")
    if len(items) < 3 or "-" not in items[-1]:
        return None
    index = items[-1].split("-")[1]
    return (f"{items[0]}_{items[1]}", int(index)) if index.isdigit() else None


def load_records(ocsr_dir: Path, labels_file: str, smiles_column: str = "smiles") -> pd.DataFrame:
    """Return one row per labelled image: sample_id, row_index, pdf_name, paper_index, smiles, group, image_path."""
    labels = pd.read_csv(ocsr_dir / labels_file)
    missing_columns = {"pdf_name", "index", smiles_column} - set(labels.columns)
    if missing_columns:
        raise ValueError(f"BioVista labels are missing {sorted(missing_columns)}. Found: {list(labels.columns)}")

    # Index every image by (pdf_name, index), exactly as the authors' script does
    images = {}
    for path in sorted(ocsr_dir.rglob("*")):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            key = _image_key(path.stem)
            if key is not None:
                images[key] = path

    records = pd.DataFrame(
        {
            "row_index": range(len(labels)),
            "pdf_name": labels["pdf_name"].astype(str),
            "paper_index": labels["index"].astype(int),
            "smiles": labels[smiles_column].astype(str),
        }
    )
    records["image_path"] = [images.get(key) for key in zip(records["pdf_name"], records["paper_index"])]

    missing = int(records["image_path"].isna().sum())
    if missing:
        LOGGER.warning("%d of %d labelled images were not found on disk and are skipped", missing, len(records))
    records = records.dropna(subset=["image_path"]).reset_index(drop=True)
    if records.empty:
        raise ValueError(f"No BioVista images matched their labels in {ocsr_dir}")

    records["sample_id"] = records["image_path"].map(lambda path: path.stem)
    records["group"] = records["smiles"].map(structure_group)
    LOGGER.info("Matched %d BioVista images to their labels", len(records))
    return records


def id_aliases(table: pd.DataFrame) -> dict[str, str]:
    """Every way another run might name an image, mapped to our sample_id.

    Covers the image file name, the row number in the label file, and the
    paper name plus index written with common separators.
    """
    aliases: dict[str, str] = {}
    for row in table.itertuples(index=False):
        sample_id = str(row.sample_id)
        names = {sample_id, str(row.row_index)}
        for separator in ("_", "-", ":", "/"):
            names.add(f"{row.pdf_name}{separator}{row.paper_index}")
        for name in names:
            aliases.setdefault(name, sample_id)
    return aliases


def select_sample(records: pd.DataFrame, n_samples: int, seed: int, sample_ids_file: Path | None = None) -> pd.DataFrame:
    """Pick the benchmark sample. Uses the teammate's sample IDs when available."""
    if sample_ids_file and Path(sample_ids_file).exists():
        teammate = pd.read_csv(sample_ids_file, dtype=str)
        aliases = id_aliases(records)
        wanted = {aliases.get(_stem(value)) for value in teammate["sample_id"]} - {None}
        if wanted:
            sample = records[records["sample_id"].isin(wanted)].reset_index(drop=True)
            LOGGER.info("Using the %d images from %s", len(sample), Path(sample_ids_file).name)
            _check_teammate_answers(teammate, sample, aliases)
            return sample
        LOGGER.warning(
            "None of the sample IDs in %s matched (for example %s), drawing a seeded sample instead",
            Path(sample_ids_file).name,
            list(teammate["sample_id"].head(3)),
        )
    elif sample_ids_file:
        LOGGER.warning("%s not found, drawing a seeded sample instead", sample_ids_file)

    if n_samples <= 0 or n_samples >= len(records):
        return records
    LOGGER.info("Drawing %d random images with seed %d", n_samples, seed)
    return records.sample(n=n_samples, random_state=seed).reset_index(drop=True)


def _check_teammate_answers(teammate: pd.DataFrame, sample: pd.DataFrame, aliases: dict[str, str]) -> None:
    """Confirm the teammate's images line up with ours by comparing the correct SMILES she recorded."""
    truth_column = next(
        (c for c in teammate.columns
         if "smi" in c.lower() and any(word in c.lower() for word in ("true", "target", "gold", "label", "answer", "correct"))),
        None,
    )
    if truth_column is None:
        LOGGER.info("Teammate file has no answer column, so the image match cannot be double-checked")
        return
    ours = dict(zip(sample["sample_id"], sample["smiles"]))
    pairs = [(aliases.get(_stem(sid)), smiles) for sid, smiles in zip(teammate["sample_id"], teammate[truth_column])]
    agree = sum(1 for sid, smiles in pairs if sid is not None and ours.get(sid) == smiles)
    if agree < 0.9 * len(pairs):
        LOGGER.warning(
            "Only %d of %d teammate answers match ours: the DECIMER comparison may not be on the same images",
            agree, len(pairs),
        )
    else:
        LOGGER.info("Checked against the teammate's answers: %d of %d match", agree, len(pairs))
