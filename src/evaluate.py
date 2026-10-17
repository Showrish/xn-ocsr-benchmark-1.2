"""Pipeline step: score every tool and save the results.

Two evaluations, each run when its predictions exist:

1. BioVista OCSR images (public). One correct SMILES per image. Scores every
   results/biovista_ocsr/predictions_<tool>.csv, plus teammate Jirapa Bangvak's
   DECIMER predictions on the same images when they are present.
   Writes results/biovista_ocsr/: metrics_per_image.csv, metrics_by_group.csv, comparison.csv

2. Course papers. Several molecules per paper. Scores every results/predictions_<tool>.csv
   with per-paper precision, recall, and F1, averaged with equal weight.
   Writes results/: metrics_per_paper.csv, metrics_summary.csv, comparison.csv, near_misses.csv

Both feed results/summary.json and the readable results/summary.md.
Runs inside the xn-scoring environment (envs/scoring.yml).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_CONFIG, load_config, setup_logging
from src.data import load_ground_truth
from src.metrics import TanimotoScorer, canonical_smiles, set_scores

LOGGER = logging.getLogger("evaluate")

# BioVista OCSR results reported in the BioMiner README for all 8,861 images.
# Our 100-image sample is a check against these, not an exact reproduction.
PUBLISHED_OCSR = {
    "MolScribe": {"all": 0.455, "full": 0.703, "chiral": 0.481, "markush": 0.156},
    "DECIMER": {"all": 0.298, "full": 0.545, "chiral": 0.326, "markush": 0.000},
}

COURSE_SUMMARY_METRICS = [
    "precision", "recall", "f1",
    "precision_no_stereo", "recall_no_stereo", "f1_no_stereo",
    "parseable_rate", "mean_best_tanimoto", "runtime_sec",
]


# ---------------------------------------------------------------------------
# 1. BioVista OCSR images: one answer per image
# ---------------------------------------------------------------------------

def score_image(predicted: str, truth: str, scorer: TanimotoScorer) -> dict:
    """Exact match (with and without stereo), validity, and similarity for one image."""
    predicted_canonical = canonical_smiles(predicted)
    truth_canonical = canonical_smiles(truth)
    valid = predicted_canonical is not None
    similarity = 0.0  # invalid predictions get zero similarity
    if valid and truth_canonical is not None:
        similarity, _ = scorer.best_match(predicted_canonical, {truth_canonical: scorer.fingerprint(truth_canonical)})
    return {
        "exact_match": int(valid and predicted_canonical == truth_canonical),
        "exact_match_no_stereo": int(
            valid and canonical_smiles(predicted, keep_stereo=False) == canonical_smiles(truth, keep_stereo=False)
        ),
        "valid": int(valid),
        "tanimoto": similarity,
    }


def load_image_predictions(results_dir: Path, teammate_dir: Path) -> list[pd.DataFrame]:
    """Our predictions on the BioVista sample, plus the teammate's DECIMER file if present."""
    files = sorted((results_dir / "biovista_ocsr").glob("predictions_*.csv"))
    teammate_file = teammate_dir / "predictions_decimer.csv"
    if teammate_file.exists():
        files.append(teammate_file)
    tables = []
    for path in files:
        try:
            table = pd.read_csv(path, dtype={"sample_id": str})
        except pd.errors.EmptyDataError:
            table = pd.DataFrame()
        if table.empty:
            LOGGER.warning("%s is empty, skipping it", path.name)
            continue
        table["sample_id"] = table["sample_id"].astype(str).map(lambda value: Path(value).stem)
        tables.append(table)
        LOGGER.info("Loaded %s (%d rows)", path.name, len(table))
    return tables


def evaluate_images(results_dir: Path, teammate_dir: Path, scorer: TanimotoScorer) -> dict | None:
    """Score every tool on the BioVista OCSR sample. Returns tables for the summary."""
    manifest_path = results_dir / "biovista_ocsr" / "sample_manifest.csv"
    if not manifest_path.exists():
        LOGGER.info("No BioVista OCSR sample found, skipping the image evaluation")
        return None
    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})

    per_image_rows = []
    for predictions in load_image_predictions(results_dir, teammate_dir):
        tool = str(predictions["tool"].dropna().iloc[0])
        if not predictions["sample_id"].isin(manifest["sample_id"]).any() and predictions["sample_id"].str.isdigit().all():
            # IDs given as row numbers in the label file: translate them to our image IDs
            row_to_id = dict(zip(manifest["row_index"].astype(int).astype(str), manifest["sample_id"]))
            predictions = predictions.assign(sample_id=predictions["sample_id"].map(row_to_id))
        merged = manifest.merge(predictions[["sample_id", "predicted_smiles", "runtime_sec"]], on="sample_id", how="left")
        if merged["predicted_smiles"].notna().sum() == 0:
            LOGGER.warning("%s has no predictions for the sampled images, skipping it", tool)
            continue
        for row in merged.itertuples(index=False):
            predicted = row.predicted_smiles if isinstance(row.predicted_smiles, str) else ""
            per_image_rows.append(
                {
                    "tool": tool,
                    "sample_id": row.sample_id,
                    "group": row.group,
                    "true_smiles": row.smiles,
                    "predicted_smiles": predicted,
                    "runtime_sec": row.runtime_sec,
                    **score_image(predicted, row.smiles, scorer),
                }
            )

    per_image = pd.DataFrame(per_image_rows)
    by_group = pd.concat(
        [per_image.assign(group="all"), per_image], ignore_index=True
    ).groupby(["tool", "group"], as_index=False).agg(
        images=("sample_id", "count"),
        exact_match=("exact_match", "mean"),
        exact_match_no_stereo=("exact_match_no_stereo", "mean"),
        valid_rate=("valid", "mean"),
        mean_tanimoto=("tanimoto", "mean"),
        mean_runtime_sec=("runtime_sec", "mean"),
    )
    by_group["published_exact_match"] = [
        PUBLISHED_OCSR.get(tool, {}).get(group) for tool, group in zip(by_group["tool"], by_group["group"])
    ]
    group_order = {"all": 0, "full": 1, "chiral": 2, "markush": 3}
    by_group = by_group.sort_values(["group", "tool"], key=lambda col: col.map(group_order) if col.name == "group" else col)
    comparison = by_group.pivot(index="group", columns="tool", values="exact_match").reset_index()
    comparison.columns = ["group"] + [f"exact_match_{tool}" for tool in comparison.columns[1:]]

    output_dir = results_dir / "biovista_ocsr"
    per_image.to_csv(output_dir / "metrics_per_image.csv", index=False)
    by_group.to_csv(output_dir / "metrics_by_group.csv", index=False)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    LOGGER.info("Scored %d tools on %d BioVista images", per_image["tool"].nunique(), len(manifest))
    return {"by_group": by_group, "n_images": len(manifest)}


# ---------------------------------------------------------------------------
# 2. Course papers: several molecules per paper
# ---------------------------------------------------------------------------

def ground_truth_sets(ground_truth: pd.DataFrame) -> dict[int, dict[str, set[str]]]:
    """Unique canonical ground truth structures per paper, with and without stereo."""
    sets: dict[int, dict[str, set[str]]] = {}
    for paper_id, group in ground_truth.groupby("paper_id"):
        with_stereo = {canonical_smiles(s) for s in group["smiles"]} - {None}
        no_stereo = {canonical_smiles(s, keep_stereo=False) for s in group["smiles"]} - {None}
        sets[int(paper_id)] = {"stereo": with_stereo, "no_stereo": no_stereo}
    return sets


def score_paper(
    paper_predictions: pd.DataFrame, truth: dict[str, set[str]], scorer: TanimotoScorer, near_miss_threshold: float
) -> tuple[dict, list[dict]]:
    """Score one tool on one paper. Returns the metrics row and any near misses."""
    raw_smiles = [s for s in paper_predictions["predicted_smiles"].fillna("").astype(str) if s.strip()]
    valid = [s for s in (canonical_smiles(s) for s in raw_smiles) if s is not None]

    predicted = set(valid)
    predicted_no_stereo = {canonical_smiles(s, keep_stereo=False) for s in valid} - {None}
    exact = set_scores(predicted, truth["stereo"])
    no_stereo = set_scores(predicted_no_stereo, truth["no_stereo"])

    # Near misses: wrong predictions whose closest true molecule is still very similar
    reference = {s: scorer.fingerprint(s) for s in truth["stereo"]}
    reference = {s: fp for s, fp in reference.items() if fp is not None}
    near_misses, best_scores = [], []
    for smiles in predicted:
        score, closest = scorer.best_match(smiles, reference)
        best_scores.append(score)
        if smiles not in truth["stereo"] and score >= near_miss_threshold:
            near_misses.append({"predicted_smiles": smiles, "closest_true_smiles": closest, "tanimoto": round(score, 4)})

    row = {
        "n_ground_truth": len(truth["stereo"]),
        "n_predictions": len(raw_smiles),
        "n_unique_valid": len(predicted),
        "parseable_rate": len(valid) / len(raw_smiles) if raw_smiles else 0.0,
        "tp": exact["tp"],
        "precision": exact["precision"],
        "recall": exact["recall"],
        "f1": exact["f1"],
        "tp_no_stereo": no_stereo["tp"],
        "precision_no_stereo": no_stereo["precision"],
        "recall_no_stereo": no_stereo["recall"],
        "f1_no_stereo": no_stereo["f1"],
        "near_misses": len(near_misses),
        "mean_best_tanimoto": sum(best_scores) / len(best_scores) if best_scores else 0.0,
        "runtime_sec": float(paper_predictions["runtime_sec"].fillna(0).sum()),
    }
    return row, near_misses


def evaluate_course_papers(config: dict, scorer: TanimotoScorer) -> dict | None:
    """Score every course-paper predictions file. Papers with no predictions score zero,
    so every paper keeps equal weight in the average."""
    paths, settings, paper_ids = config["paths"], config["evaluation"], config["papers"]
    results_dir = paths["results_dir"]
    prediction_files = sorted(results_dir.glob("predictions_*.csv"))
    if not prediction_files:
        LOGGER.info("No course-paper predictions found, skipping the course evaluation")
        return None

    truth_sets = ground_truth_sets(load_ground_truth(paths["ground_truth_csv"], paper_ids))
    per_paper_rows, near_miss_rows = [], []
    for path in prediction_files:
        predictions = pd.read_csv(path)
        tool = str(predictions["tool"].dropna().iloc[0]) if predictions["tool"].notna().any() else path.stem
        for paper_id in paper_ids:
            row, near_misses = score_paper(
                predictions[predictions["paper_id"] == paper_id], truth_sets[paper_id], scorer, settings["near_miss_threshold"]
            )
            per_paper_rows.append({"tool": tool, "paper_id": paper_id, **row})
            near_miss_rows += [{"tool": tool, "paper_id": paper_id, **miss} for miss in near_misses]
        LOGGER.info("Scored %s", path.name)

    per_paper = pd.DataFrame(per_paper_rows)
    summary = per_paper.groupby("tool", as_index=False)[COURSE_SUMMARY_METRICS].mean()
    summary.insert(1, "papers", len(paper_ids))
    comparison = per_paper.pivot(index="paper_id", columns="tool", values=["precision", "recall", "f1"])
    comparison.columns = [f"{metric}_{tool}" for metric, tool in comparison.columns]
    comparison = comparison.reset_index()
    near_misses = pd.DataFrame(near_miss_rows, columns=["tool", "paper_id", "predicted_smiles", "closest_true_smiles", "tanimoto"])

    per_paper.to_csv(results_dir / "metrics_per_paper.csv", index=False)
    summary.to_csv(results_dir / "metrics_summary.csv", index=False)
    comparison.to_csv(results_dir / "comparison.csv", index=False)
    near_misses.to_csv(results_dir / "near_misses.csv", index=False)
    f1_columns = ["paper_id"] + [c for c in comparison.columns if c.startswith("f1_")]
    return {
        "summary": summary,
        "per_paper": per_paper,
        "f1_comparison": comparison[f1_columns],
        "unique_structures": {str(p): len(s["stereo"]) for p, s in truth_sets.items()},
    }


# ---------------------------------------------------------------------------
# Readable summary
# ---------------------------------------------------------------------------

def format_markdown_table(table: pd.DataFrame) -> str:
    """Render a small DataFrame as a Markdown table (numbers rounded to 3 places)."""
    def cell(value) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    header = "| " + " | ".join(str(c) for c in table.columns) + " |"
    divider = "|" + "|".join("---" for _ in table.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([header, divider, *body])


def write_summary(results_dir: Path, images: dict | None, course: dict | None) -> None:
    """Write results/summary.md and results/summary.json from whichever evaluations ran."""
    sections = ["# Benchmark Results", ""]
    summary_json: dict = {}

    if images:
        sections += [
            f"## BioVista OCSR images ({images['n_images']} images)",
            "",
            "Share of images whose prediction exactly matches the correct SMILES after RDKit canonicalization.",
            "Published values are from the BioMiner README for all 8,861 images.",
            "",
            format_markdown_table(images["by_group"]),
            "",
        ]
        summary_json["biovista_ocsr"] = images["by_group"].to_dict(orient="records")

    if course:
        sections += [
            "## Course papers (equal-weight average over papers)",
            "",
            "Exact match of canonical SMILES against the course ground truth, scored per paper on unique structures.",
            "",
            format_markdown_table(course["summary"]),
            "",
            "### Side-by-side F1 per paper",
            "",
            format_markdown_table(course["f1_comparison"]),
            "",
        ]
        summary_json["course_papers"] = {
            "unique_structures_per_paper": course["unique_structures"],
            "summary": course["summary"].to_dict(orient="records"),
            "per_paper": course["per_paper"].to_dict(orient="records"),
        }

    biovista_path = results_dir / "biovista_metrics.csv"
    if biovista_path.exists():
        biovista = pd.read_csv(biovista_path).dropna(subset=["published_450_papers"])
        sections += [
            "## BioMiner end to end on the BioVista subset vs. published results",
            "",
            "Published values are the authors' averages over the 450-paper BioVista test set.",
            "",
            format_markdown_table(biovista),
            "",
        ]
        summary_json["biominer_biovista"] = biovista.to_dict(orient="records")

    (results_dir / "summary.md").write_text("\n".join(sections), encoding="utf-8")
    (results_dir / "summary.json").write_text(json.dumps(summary_json, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    paths, settings = config["paths"], config["evaluation"]
    scorer = TanimotoScorer(settings["fingerprint_radius"], settings["fingerprint_bits"])

    images = evaluate_images(paths["results_dir"], paths["teammate_dir"], scorer)
    course = evaluate_course_papers(config, scorer)
    if images is None and course is None:
        raise SystemExit("No predictions found to score. Run the tool steps first.")

    write_summary(paths["results_dir"], images, course)
    LOGGER.info("Wrote results to %s", paths["results_dir"])


if __name__ == "__main__":
    main()
