"""Pipeline step: run BioMiner end to end with an API model.

Two modes:
  course    Runs BioMiner on each course PDF and saves the extracted structures to
            results/predictions_biominer.csv in the shared schema.
  biovista  Runs BioMiner on the BioVista subset with its built-in evaluation and
            saves the metrics next to the authors' published numbers, as a check
            that our installation reproduces their results.

Runs inside BioMiner's own conda environment (BioMiner/main_environment.yml).
Credentials come from environment variables and are written only to a temp file
outside the repo, so they are never committed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import yaml

from src import EXTRA_COLUMNS, PREDICTION_COLUMNS
from src.config import DEFAULT_CONFIG, load_config, load_env_file, require_env, setup_logging
from src.data import find_paper_pdfs

LOGGER = logging.getLogger("run_biominer")
TOOL_NAME = "BioMiner"

# BioMiner's three model servers (PDF parsing, molecule detection, OCSR), from its README
SERVER_SCRIPTS = ["scripts/mineru_server.py", "scripts/moldet_server.py", "scripts/ocsr_server.py"]

# BioMiner config fields for API models, mapped to the environment variables that fill them
API_FIELDS = {
    "api_key": "BIOMINER_API_KEY",
    "base_url": "BIOMINER_BASE_URL",
    "text_mllm_type": "BIOMINER_TEXT_MODEL",
    "vision_mllm_type": "BIOMINER_VISION_MODEL",
}

# Averages the authors report on the 450-paper BioVista test set (BioMiner README).
# Our subset is far smaller, so these are a sanity check, not an exact target.
PUBLISHED_BIOVISTA = {
    "recall_structure_list": 0.5801,
    "precision_structure_list": 0.6151,
    "recall_coreference_structure_list": 0.5113,
    "precision_coreference_structure_list": 0.5350,
    "recall_chiral_structure_list": 0.4540,
    "precision_chiral_structure_list": 0.4671,
    "recall_bioactivity_list": 0.6535,
    "precision_bioactivity_list": 0.6136,
    "recall_together_list": 0.3284,
    "precision_together_list": 0.3188,
}

TABLE_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".xlsx"}


# ---------------------------------------------------------------------------
# Configuration with credentials
# ---------------------------------------------------------------------------

def set_nested_keys(node, values: dict) -> set[str]:
    """Overwrite every key named in `values`, wherever it sits in the YAML tree."""
    found: set[str] = set()
    if isinstance(node, dict):
        for key in list(node):
            if key in values:
                node[key] = values[key]
                found.add(key)
            else:
                found |= set_nested_keys(node[key], values)
    elif isinstance(node, list):
        for item in node:
            found |= set_nested_keys(item, values)
    return found


def write_runtime_config(template_path: Path) -> Path:
    """Copy BioMiner's API config to a temp folder and fill it from environment variables."""
    with open(template_path, "r", encoding="utf-8") as handle:
        biominer_config = yaml.safe_load(handle)

    values = {field: require_env(env_name) for field, env_name in API_FIELDS.items()}
    missing = set(values) - set_nested_keys(biominer_config, values)
    if missing:
        raise KeyError(f"Fields not found in {template_path}: {sorted(missing)}")

    runtime_path = Path(tempfile.mkdtemp(prefix="biominer_config_")) / "runtime_config.yaml"
    with open(runtime_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(biominer_config, handle, sort_keys=False)
    return runtime_path


# ---------------------------------------------------------------------------
# Running BioMiner
# ---------------------------------------------------------------------------

def biominer_env(repo_dir: Path) -> dict:
    """BioMiner's README runs everything with the repo root on PYTHONPATH."""
    return {**os.environ, "PYTHONPATH": str(repo_dir)}


@contextmanager
def biominer_servers(repo_dir: Path, log_dir: Path, warmup_seconds: int):
    """Start BioMiner's model servers in the background and stop them afterwards."""
    processes = []
    try:
        for script in SERVER_SCRIPTS:
            log_file = open(log_dir / f"{Path(script).stem}.log", "w", encoding="utf-8")
            process = subprocess.Popen(
                [sys.executable, script],
                cwd=repo_dir,
                env=biominer_env(repo_dir),
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            processes.append((script, process, log_file))
            LOGGER.info("Started %s (pid %d)", script, process.pid)

        LOGGER.info("Waiting %d seconds for servers to load their models", warmup_seconds)
        time.sleep(warmup_seconds)
        for script, process, _ in processes:
            if process.poll() is not None:
                raise RuntimeError(f"{script} exited early. Check its log in {log_dir}")
        yield
    finally:
        for script, process, log_file in processes:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
            log_file.close()
            LOGGER.info("Stopped %s", script)


def run_biominer_cli(
    repo_dir: Path, config_path: Path, pdf_path: Path, output_dir: Path, log_path: Path, biovista: bool = False
) -> tuple[bool, float, str]:
    """Run BioMiner's command line entry point. Returns (succeeded, seconds, stdout)."""
    command = [
        sys.executable,
        "example_open_source.py",
        f"--config_path={config_path}",
        f"--pdf={pdf_path}",
        f"--output_dir={output_dir}",
    ]
    if biovista:
        command.append("--biovista_evaluate")

    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    completed = subprocess.run(
        command, cwd=repo_dir, env=biominer_env(repo_dir), capture_output=True, text=True
    )
    elapsed = time.perf_counter() - start

    log_path.write_text(f"{completed.stdout}\n--- stderr ---\n{completed.stderr}", encoding="utf-8")
    if completed.returncode != 0:
        LOGGER.error("BioMiner failed on %s (exit %d). See %s", pdf_path.name, completed.returncode, log_path)
    return completed.returncode == 0, elapsed, completed.stdout


# ---------------------------------------------------------------------------
# Collecting outputs
# ---------------------------------------------------------------------------

def find_column(table: pd.DataFrame, name: str) -> str | None:
    """Find a column by name, ignoring case and surrounding spaces."""
    for column in table.columns:
        if str(column).strip().lower() == name:
            return column
    return None


def read_table(path: Path) -> pd.DataFrame | None:
    """Read one BioMiner output table. Returns None for files that are not tables."""
    try:
        if path.suffix == ".csv":
            return pd.read_csv(path)
        if path.suffix == ".tsv":
            return pd.read_csv(path, sep="\t")
        if path.suffix == ".jsonl":
            return pd.read_json(path, lines=True)
        if path.suffix == ".json":
            return pd.read_json(path)
        if path.suffix == ".xlsx":
            return pd.read_excel(path)
    except (ValueError, ImportError, pd.errors.ParserError):
        return None
    return None


def collect_structures(output_dir: Path) -> pd.DataFrame:
    """Gather every structure BioMiner extracted, from any output table with a SMILES column.

    BioMiner reports bioactivity rows (protein, ligand, type, value, unit, smiles).
    The same ligand can appear in several rows, which scoring later collapses into a set.
    """
    frames = []
    for path in sorted(output_dir.rglob("*")):
        if path.suffix.lower() not in TABLE_SUFFIXES:
            continue
        table = read_table(path)
        if table is None or table.empty:
            continue
        smiles_column = find_column(table, "smiles")
        if smiles_column is None:
            continue
        label_column = find_column(table, "ligand")
        frames.append(
            pd.DataFrame(
                {
                    "predicted_smiles": table[smiles_column].fillna("").astype(str),
                    "label": table[label_column].astype(str) if label_column else "",
                    "source": path.relative_to(output_dir).as_posix(),
                }
            )
        )

    if not frames:
        return pd.DataFrame(columns=["predicted_smiles", "label", "source"])
    structures = pd.concat(frames, ignore_index=True)
    structures = structures[structures["predicted_smiles"].str.strip() != ""]
    return structures.drop_duplicates(subset=["predicted_smiles", "label"]).reset_index(drop=True)


def parse_biovista_metrics(stdout: str) -> pd.DataFrame:
    """Parse BioMiner's 'Average <metric>: <value>' evaluation lines."""
    rows = [
        {"metric": name, "value": float(value)}
        for name, value in re.findall(r"Average\s+(\w+):\s*([0-9.]+)", stdout)
    ]
    metrics = pd.DataFrame(rows, columns=["metric", "value"])
    metrics["published_450_papers"] = metrics["metric"].map(PUBLISHED_BIOVISTA)
    return metrics


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_course_papers(config: dict, runtime_config: Path, raw_dir: Path, log_dir: Path) -> None:
    """Run BioMiner on each course paper and save predictions in the shared schema."""
    paths = config["paths"]
    pdfs = find_paper_pdfs(paths["course_papers_dir"], config["papers"])
    rows = []

    for paper_id, pdf_path in pdfs.items():
        LOGGER.info("Running BioMiner on paper %s", paper_id)
        output_dir = raw_dir / f"paper_{paper_id}"
        succeeded, seconds, _ = run_biominer_cli(
            paths["biominer_repo_dir"], runtime_config, pdf_path, output_dir, log_dir / f"biominer_paper_{paper_id}.log"
        )
        structures = collect_structures(output_dir) if succeeded else pd.DataFrame()
        if structures.empty:
            # Keep a row so the failure or empty result is visible in the results
            rows.append(_prediction_row(paper_id, "", "", "", seconds, "failed" if not succeeded else "no structures"))
            continue

        # BioMiner reports time per paper, so it is split evenly across that paper's structures
        per_structure_seconds = seconds / len(structures)
        for index, structure in structures.iterrows():
            rows.append(
                _prediction_row(
                    paper_id,
                    crop_id=f"biominer_{paper_id}_{index}",
                    smiles=structure["predicted_smiles"],
                    label=structure["label"],
                    seconds=per_structure_seconds,
                    error="",
                    source=structure["source"],
                )
            )
        LOGGER.info("Paper %s: %d structures in %.1f s", paper_id, len(structures), seconds)

    predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS + EXTRA_COLUMNS)
    output_path = paths["results_dir"] / "predictions_biominer.csv"
    predictions.to_csv(output_path, index=False)
    LOGGER.info("Saved %d predictions to %s", len(predictions), output_path)


def _prediction_row(paper_id, crop_id, smiles, label, seconds, error, source="") -> dict:
    """One row in the shared schema. BioMiner does not report page numbers or confidence."""
    return {
        "paper_id": paper_id,
        "page": None,
        "crop_id": crop_id,
        "predicted_smiles": smiles,
        "tool": TOOL_NAME,
        "runtime_sec": round(seconds, 3),
        "confidence": None,
        "label": label,
        "source": source,
        "error": error,
    }


def run_biovista_subset(config: dict, runtime_config: Path, raw_dir: Path, log_dir: Path) -> None:
    """Run BioMiner's built-in BioVista evaluation on the subset and save the metrics."""
    paths = config["paths"]
    pdf_dir = paths["biovista_pdfs_dir"]
    n_papers = len(list(pdf_dir.glob("*.pdf")))
    LOGGER.info("Running BioVista evaluation on %d papers in %s", n_papers, pdf_dir)

    succeeded, seconds, stdout = run_biominer_cli(
        paths["biominer_repo_dir"], runtime_config, pdf_dir, raw_dir / "biovista", log_dir / "biominer_biovista.log", biovista=True
    )
    metrics = parse_biovista_metrics(stdout)
    if not succeeded or metrics.empty:
        raise RuntimeError("BioVista evaluation produced no metrics. Check results/logs/biominer_biovista.log")

    results_dir = paths["results_dir"]
    metrics.to_csv(results_dir / "biovista_metrics.csv", index=False)
    summary = {"n_papers": n_papers, "runtime_sec": round(seconds, 1), "metrics": metrics.to_dict(orient="records")}
    (results_dir / "biovista_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Saved %d BioVista metrics to %s", len(metrics), results_dir / "biovista_metrics.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml")
    parser.add_argument("--mode", choices=["course", "biovista"], required=True)
    args = parser.parse_args()

    setup_logging()
    load_env_file()
    config = load_config(args.config)
    paths, settings = config["paths"], config["biominer"]

    results_dir = paths["results_dir"]
    raw_dir = results_dir / "raw" / "biominer"  # raw outputs stay out of git (see .gitignore)
    log_dir = results_dir / "logs"
    for folder in (raw_dir, log_dir):
        folder.mkdir(parents=True, exist_ok=True)

    runtime_config = write_runtime_config(paths["biominer_repo_dir"] / settings["config_template"])
    with biominer_servers(paths["biominer_repo_dir"], log_dir, settings["server_warmup_seconds"]):
        if args.mode == "course":
            run_course_papers(config, runtime_config, raw_dir, log_dir)
        else:
            run_biovista_subset(config, runtime_config, raw_dir, log_dir)


if __name__ == "__main__":
    main()
