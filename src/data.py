"""Data loading: course ground truth, course PDFs, and the shared DECIMER crops."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

# Column names used in the course ground truth file
GT_PAPER_COL = "paperID"
GT_NAME_COL = "name of molecule"
GT_SMILES_COL = "canonical_SMILES"


def load_ground_truth(csv_path: Path, paper_ids: list[int]) -> pd.DataFrame:
    """Load the course ground truth and keep only the papers being benchmarked.

    Returns columns: paper_id, molecule_name, smiles.
    """
    table = pd.read_csv(csv_path)
    missing = {GT_PAPER_COL, GT_NAME_COL, GT_SMILES_COL} - set(table.columns)
    if missing:
        raise ValueError(f"Ground truth file {csv_path} is missing columns: {sorted(missing)}")

    table = table.rename(
        columns={GT_PAPER_COL: "paper_id", GT_NAME_COL: "molecule_name", GT_SMILES_COL: "smiles"}
    )
    table["paper_id"] = table["paper_id"].astype(int)
    table = table[table["paper_id"].isin(paper_ids)].reset_index(drop=True)
    LOGGER.info("Loaded %d ground truth molecules across %d papers", len(table), table["paper_id"].nunique())
    return table


def find_paper_pdfs(papers_dir: Path, paper_ids: list[int]) -> dict[int, Path]:
    """Map each paper ID to its PDF. Course PDFs are named <paper_id>.pdf."""
    pdfs: dict[int, Path] = {}
    for paper_id in paper_ids:
        pdf_path = papers_dir / f"{paper_id}.pdf"
        if pdf_path.exists():
            pdfs[paper_id] = pdf_path
        else:
            LOGGER.warning("Paper %s not found at %s, skipping it", paper_id, pdf_path)
    return pdfs


@dataclass(frozen=True)
class Crop:
    """One cropped molecule image produced by DECIMER segmentation."""

    paper_id: int
    page: int | None
    crop_id: str
    image_path: Path


def load_crops(
    crops_dir: Path,
    paper_ids: list[int],
    filename_pattern: str,
    extensions: list[str],
    manifest_csv: str | Path | None = None,
) -> list[Crop]:
    """Collect DECIMER crops for the selected papers.

    Uses a manifest CSV (paper_id, page, crop_id, image_path) when one is given.
    Otherwise it walks crops_dir and reads the paper and page from each file path.
    """
    if manifest_csv:
        return _crops_from_manifest(Path(manifest_csv), crops_dir, paper_ids)

    if not crops_dir.exists():
        raise FileNotFoundError(f"Crops folder not found: {crops_dir}")

    pattern = re.compile(filename_pattern, flags=re.IGNORECASE)
    allowed_suffixes = {suffix.lower() for suffix in extensions}
    crops: list[Crop] = []

    for image_path in sorted(crops_dir.rglob("*")):
        if image_path.suffix.lower() not in allowed_suffixes:
            continue
        relative = image_path.relative_to(crops_dir)
        paper_id, page = _parse_crop_path(relative, pattern, paper_ids)
        if paper_id is None or paper_id not in paper_ids:
            continue
        # The relative path (without extension) is unique, so it doubles as the crop ID
        crop_id = relative.with_suffix("").as_posix().replace("/", "__")
        crops.append(Crop(paper_id=paper_id, page=page, crop_id=crop_id, image_path=image_path))

    _log_crop_counts(crops)
    return crops


def _parse_crop_path(relative: Path, pattern: re.Pattern, paper_ids: list[int]) -> tuple[int | None, int | None]:
    """Read paper and page numbers from a crop path, falling back to folder names."""
    match = pattern.search(relative.as_posix())
    if match:
        page = match.groupdict().get("page")
        return int(match.group("paper_id")), int(page) if page else None

    # Fallback: crops stored in one folder per paper, e.g. 19/crop_5.png
    for part in relative.parts[:-1]:
        if part.isdigit() and int(part) in paper_ids:
            return int(part), None
    LOGGER.warning("Could not read a paper ID from %s, skipping it", relative)
    return None, None


def _crops_from_manifest(manifest_path: Path, crops_dir: Path, paper_ids: list[int]) -> list[Crop]:
    """Build the crop list from a manifest CSV, with image paths relative to crops_dir."""
    manifest = pd.read_csv(manifest_path)
    crops = []
    for row in manifest.itertuples(index=False):
        if int(row.paper_id) not in paper_ids:
            continue
        page = int(row.page) if pd.notna(row.page) else None
        image_path = Path(row.image_path)
        if not image_path.is_absolute():
            image_path = crops_dir / image_path
        crops.append(Crop(int(row.paper_id), page, str(row.crop_id), image_path))
    _log_crop_counts(crops)
    return crops


def _log_crop_counts(crops: list[Crop]) -> None:
    """Log how many crops were found per paper, so missing papers are easy to spot."""
    counts: dict[int, int] = {}
    for crop in crops:
        counts[crop.paper_id] = counts.get(crop.paper_id, 0) + 1
    LOGGER.info("Loaded %d crops: %s", len(crops), dict(sorted(counts.items())))
