"""Run the full benchmark end to end with one command.

    python run_pipeline.py                      # every step
    python run_pipeline.py --steps evaluate     # just one step

Steps, in order:
  ocsr_molscribe     MolScribe on the public BioVista OCSR images (same sample as
                     Jirapa's DECIMER benchmark, so the two tools read identical images)
  course_molscribe   MolScribe on the DECIMER crops of the six course papers
  course_biominer    BioMiner end to end on the six course papers
  biovista_biominer  BioMiner end to end on the BioVista subset, checked against published results
  evaluate           score every tool and write the results tables

Each step runs inside its own environment (set in config/config.yaml), because
the tools have conflicting dependencies. If a step fails, the remaining steps
still run, and evaluation scores whichever predictions exist.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys

from src.config import DEFAULT_CONFIG, REPO_ROOT, load_config, setup_logging

LOGGER = logging.getLogger("pipeline")

# step name -> (environment name in config.yaml, module arguments)
STEPS = {
    "ocsr_molscribe": ("molscribe", ["-m", "src.run_molscribe", "--dataset", "biovista_ocsr"]),
    "course_molscribe": ("molscribe", ["-m", "src.run_molscribe", "--dataset", "course"]),
    "course_biominer": ("biominer", ["-m", "src.run_biominer", "--mode", "course"]),
    "biovista_biominer": ("biominer", ["-m", "src.run_biominer", "--mode", "biovista"]),
    "evaluate": ("scoring", ["-m", "src.evaluate"]),
}


def run_step(step: str, config: dict, config_path: str) -> bool:
    """Launch one step in its environment. Returns True if it finished successfully."""
    environment, module_args = STEPS[step]
    command = list(config["environments"][environment]) + module_args + ["--config", config_path]
    LOGGER.info("Starting step '%s': %s", step, " ".join(command))
    # No step displays plots. Notebooks like Colab set a notebook-only matplotlib
    # backend that the tool environments lack, so steps use the headless one.
    step_env = {**os.environ, "MPLBACKEND": "Agg"}
    completed = subprocess.run(command, cwd=REPO_ROOT, env=step_env)
    if completed.returncode == 0:
        LOGGER.info("Step '%s' finished", step)
        return True
    LOGGER.error("Step '%s' failed with exit code %d", step, completed.returncode)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml")
    parser.add_argument("--steps", nargs="+", choices=list(STEPS), default=list(STEPS), help="Steps to run")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop at the first failed step")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    outcomes = {}
    for step in args.steps:
        outcomes[step] = run_step(step, config, args.config)
        if not outcomes[step] and args.stop_on_error:
            break

    LOGGER.info("Summary: %s", ", ".join(f"{step}={'ok' if ok else 'FAILED'}" for step, ok in outcomes.items()))
    sys.exit(0 if all(outcomes.values()) else 1)


if __name__ == "__main__":
    main()
