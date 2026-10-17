"""Load the public BioVista OCSR set: cropped molecule images with their correct SMILES.

The sample matches teammate Jirapa Bangvak's DECIMER benchmark (100 images, seed 42),
so DECIMER and MolScribe are compared on identical images, as planned in our 1.1
proposal. When her per-image file is present, its sample IDs fix the exact images.
"""

from __future__ import annotations

import logging
import re
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


def _index_images(ocsr_dir: Path) -> dict[str, Path]:
    """Every image under the OCSR folder, keyed by file name without extension."""
    return {path.stem: path for path in sorted(ocsr_dir.rglob("*")) if path.suffix.lower() in IMAGE_SUFFIXES}


def _best_id_column(labels: pd.DataFrame, smiles_column: str, images: dict[str, Path]) -> tuple[str | None, int]:
    """The column whose values match the most image file names, and how many it matches."""
    best_column, best_hits = None, 0
    for column in labels.columns:
        if column == smiles_column:
            continue
        hits = int(labels[column].map(_stem).isin(images).sum())
        if hits > best_hits:
            best_column, best_hits = column, hits
    return best_column, best_hits


def _images_by_row_number(n_rows: int, images: dict[str, Path]) -> list[Path] | None:
    """Match rows to images named with a running number (e.g. 00012.png), counting from 0 or 1."""
    numbered: dict[int, Path] = {}
    for stem, path in images.items():
        match = re.search(r"(\d+)(?!.*\d)", stem)  # last number in the file name
        if match:
            numbered.setdefault(int(match.group(1)), path)
    for start in (0, 1):
        mapped = [numbered.get(row + start) for row in range(n_rows)]
        if all(mapped):
            LOGGER.info("Matched images to label rows by their running number (starting at %d)", start)
            return mapped
    return None


BIOVISTA_IMAGE_NAME = re.compile(r"^(?P<pdf>.+)_image_molminer_bbox_(?P<page>\d+)-(?P<box>\d+)$")
MOLMINER_ID = re.compile(r"P(?P<page>\d+)-(?P<box>\d+)")


def _images_by_biovista_name(labels: pd.DataFrame, images: dict[str, Path]) -> list[Path | None] | None:
    """Match rows to BioVista's image names, built from the paper name and MolMiner box ID.

    Images are named <pdf_name>_image_molminer_bbox_<page>-<box>, while the labels give
    molminer_id as P<page>-<box>. The two may count pages and boxes from different
    starting numbers, so every offset is tried and the one matching the most images wins.
    """
    if not {"pdf_name", "molminer_id"} <= set(labels.columns):
        return None
    by_key = {}
    for stem, path in images.items():
        match = BIOVISTA_IMAGE_NAME.match(stem)
        if match:
            by_key[(match["pdf"], int(match["page"]), int(match["box"]))] = path
    ids = labels["molminer_id"].astype(str).str.extract(MOLMINER_ID)

    best_hits, best_paths, best_offsets = 0, None, None
    for page_offset in (1, 0):
        for box_offset in (1, 0):
            paths = [
                by_key.get((pdf, int(page) - page_offset, int(box) - box_offset)) if pd.notna(page) else None
                for pdf, page, box in zip(labels["pdf_name"].astype(str), ids["page"], ids["box"])
            ]
            hits = sum(path is not None for path in paths)
            if hits > best_hits:
                best_hits, best_paths, best_offsets = hits, paths, (page_offset, box_offset)
    if not best_hits:
        return None
    if best_hits < 0.9 * len(labels):
        LOGGER.warning("Only %d of %d labels matched an image by name, so the sample will be smaller", best_hits, len(labels))
    LOGGER.info(
        "Matched %d of %d images by paper name and MolMiner ID (page offset %d, box offset %d)",
        best_hits, len(labels), *best_offsets,
    )
    return best_paths


def load_records(ocsr_dir: Path, labels_file: str, id_column: str = "", smiles_column: str = "") -> pd.DataFrame:
    """Return one row per labelled image: sample_id, row_index, smiles, group, image_path."""
    labels = pd.read_csv(ocsr_dir / labels_file)
    images = _index_images(ocsr_dir)

    smiles_column = smiles_column or next((c for c in labels.columns if "smi" in str(c).lower()), "")
    if not smiles_column:
        raise ValueError(f"No SMILES column found in {list(labels.columns)}. Set biovista_ocsr.smiles_column.")

    records = pd.DataFrame({"row_index": range(len(labels)), "smiles": labels[smiles_column].astype(str)})

    # Keep BioVista's own identifiers, so every result can be traced back to its paper
    for column in ("pdf_name", "molminer_id"):
        if column in labels.columns:
            records[column] = labels[column].astype(str)

    # Strategy 1: BioVista's naming scheme (paper name + MolMiner box ID)
    by_name = None if id_column else _images_by_biovista_name(labels, images)
    # Strategy 2: a column that holds the image file names (or the one set in the config)
    chosen, hits = (id_column, None) if id_column else _best_id_column(labels, smiles_column, images)
    if by_name is not None:
        records["image_path"] = by_name
        records["sample_id"] = [path.stem if path is not None else None for path in by_name]
    elif chosen and (hits is None or hits > 0):
        records["sample_id"] = labels[chosen].map(_stem)
        records["image_path"] = records["sample_id"].map(images)
        LOGGER.info("Matched images through the '%s' column", chosen)
    else:
        # Strategy 3: images named by row number
        by_row = _images_by_row_number(len(labels), images)
        if by_row is None:
            raise ValueError(
                "Could not match BioVista labels to image files.\n"
                f"  Label columns: {list(labels.columns)}\n"
                f"  First label rows:\n{labels.head(3).to_string()}\n"
                f"  Images found: {len(images)}, for example {list(images)[:5]}\n"
                "Set biovista_ocsr.id_column in config/config.yaml, or send this message for a fix."
            )
        records["image_path"] = by_row
        records["sample_id"] = [path.stem for path in by_row]

    missing = records["image_path"].isna().sum()
    if missing:
        LOGGER.warning("%d labelled images were not found on disk and are skipped", missing)
    records = records.dropna(subset=["image_path"]).reset_index(drop=True)
    if records.empty:
        raise ValueError(f"No BioVista images could be loaded from {ocsr_dir}")
    records["group"] = records["smiles"].map(structure_group)
    LOGGER.info("Loaded %d BioVista OCSR images", len(records))
    return records


def select_sample(records: pd.DataFrame, n_samples: int, seed: int, sample_ids_file: Path | None = None) -> pd.DataFrame:
    """Pick the benchmark sample. Uses the teammate's sample IDs when available."""
    if sample_ids_file and Path(sample_ids_file).exists():
        teammate_ids = pd.read_csv(sample_ids_file, dtype={"sample_id": str})["sample_id"].map(_stem)
        sample = records[records["sample_id"].isin(set(teammate_ids))]
        if sample.empty and teammate_ids.str.isdigit().all():
            # Her IDs may be row numbers in the label file rather than file names
            sample = records[records["row_index"].isin({int(i) for i in teammate_ids})]
        if not sample.empty:
            LOGGER.info("Using the %d images from %s", len(sample), Path(sample_ids_file).name)
            _check_teammate_answers(sample_ids_file, sample)
            return sample.reset_index(drop=True)
        LOGGER.warning(
            "No sample IDs from %s matched (for example %s), drawing a seeded sample instead",
            Path(sample_ids_file).name,
            list(teammate_ids[:3]),
        )

    if n_samples <= 0 or n_samples >= len(records):
        return records
    LOGGER.info("Drawing %d random images with seed %d", n_samples, seed)
    return records.sample(n=n_samples, random_state=seed).reset_index(drop=True)


def _check_teammate_answers(sample_ids_file: Path, sample: pd.DataFrame) -> None:
    """Confirm the teammate's images line up with ours by comparing the correct SMILES she recorded."""
    teammate = pd.read_csv(sample_ids_file, dtype=str)
    truth_column = next(
        (c for c in teammate.columns
         if "smi" in c.lower() and any(word in c.lower() for word in ("true", "target", "gold", "label", "answer", "correct"))),
        None,
    )
    if truth_column is None:
        LOGGER.info("Teammate file has no answer column, so the image match cannot be double-checked")
        return
    recorded = dict(zip(teammate["sample_id"].map(_stem), teammate[truth_column]))
    ours = dict(zip(sample["sample_id"], sample["smiles"]))
    ours.update(dict(zip(sample["row_index"].astype(str), sample["smiles"])))
    agree = sum(1 for sid, smiles in recorded.items() if ours.get(sid) == smiles)
    if agree < 0.9 * len(recorded):
        LOGGER.warning(
            "Only %d of %d teammate answers match ours: the DECIMER comparison may not be on the same images",
            agree, len(recorded),
        )
    else:
        LOGGER.info("Checked against the teammate's answers: %d of %d match", agree, len(recorded))
