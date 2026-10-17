# XN Benchmark: BioMiner and MolScribe

Assignment 1.2 (Individual Benchmarking Code) for AAI 6630, XN project.
Author: Showrish Reddy Sampath

This repo benchmarks two tools that read molecular structures in scientific documents and convert them to SMILES:

- **BioMiner**, run end to end (PDF parsing, molecule detection, recognition, and Markush enumeration), using an API model for its reasoning step
- **MolScribe**, a recognition model that reads one cropped molecule image at a time

It follows the plan in our group's 1.1 setup proposal, where I own MolScribe and BioMiner, Jirapa Bangvak owns DECIMER and segmentation, and Hinal Sachpara owns scoring. There are three evaluations:

| Evaluation | Data | Tools | Purpose |
|---|---|---|---|
| 1. Public OCSR images | BioVista OCSR set, the same 100-image sample (seed 42) as Jirapa's DECIMER benchmark | MolScribe, compared side by side with DECIMER | DECIMER and MolScribe read identical images, so differences come from recognition alone |
| 2. Course papers | The six course PDFs and their ground truth (220 unique structures) | BioMiner end to end, and MolScribe on the DECIMER crops | Performance on the project's real papers |
| 3. BioMiner on BioVista | A subset of BioVista papers, with BioMiner's built-in evaluation | BioMiner | Check that our installation reproduces the published results |

## Pipeline

```
data loading               inference                          metrics                outputs
--------------------       -------------------------------    -------------------    ------------------------------
BioVista OCSR images  -->  MolScribe ---------------------+
DECIMER (Jirapa)      ------------------------------------+-> per-image scoring --> results/biovista_ocsr/
course PDFs           -->  BioMiner (end to end) ---------+
DECIMER crops         -->  MolScribe ---------------------+-> per-paper scoring --> results/*.csv, summary.md
ground truth          ------------------------------------^
BioVista papers       -->  BioMiner + built-in evaluation -------------------------> results/biovista_metrics.*
```

One command runs every step: `python run_pipeline.py`.

## Repository layout

```
xn-ocsr-benchmark-1.2/
├── run_pipeline.py              Entry point. Runs every step end to end, each in its own environment
├── config/
│   └── config.yaml              All paths and settings (nothing is hardcoded in the code)
├── src/
│   ├── __init__.py              Shared results schema (the columns every tool writes)
│   ├── config.py                Loads config, resolves paths, reads credentials from env vars
│   ├── data.py                  Loads the course ground truth, course PDFs, and DECIMER crops
│   ├── biovista_ocsr.py         Loads the BioVista OCSR images and picks the benchmark sample
│   ├── run_molscribe.py         Step: MolScribe on the BioVista images or the course crops
│   ├── run_biominer.py          Steps: BioMiner on the course papers and on the BioVista subset
│   ├── metrics.py               Canonical SMILES, exact match, stereo-free match, Tanimoto
│   └── evaluate.py              Step: scores every tool and writes the results tables
├── scripts/
│   └── setup_external.py        One-time setup: BioMiner, BioVista data, weights, teammate results
├── envs/
│   ├── molscribe.yml            MolScribe environment
│   ├── scoring.yml              Scoring environment (RDKit 2026.03.2)
│   └── lock/                    Exact versions exported after setup
├── notebooks/
│   └── run_on_colab.ipynb       Runs the whole pipeline on a Colab GPU
├── tests/
│   └── test_metrics.py          Unit tests for the scoring logic
├── results/                     All outputs (see results/README.md)
├── requirements.txt             Base environment for run_pipeline.py and the setup script
└── .env.example                 Template for the API credentials
```

## Data

| Dataset | Source | Used for |
|---|---|---|
| BioVista OCSR images (8,861 images with SMILES) | Public, downloaded by `scripts/setup_external.py` from the BioMiner authors' Google Drive | Evaluation 1 |
| DECIMER results on the same BioVista sample | Jirapa Bangvak's public repo (github.com/JirapaB/xn-benchmark), copied by the setup script | Evaluation 1 sample and side-by-side comparison |
| Course papers (IDs 1, 2, 4, 6, 16, 19) and ground truth CSV | Canvas, `Dataset.zip` | Evaluation 2 |
| DECIMER crops of the course papers | Jirapa's DECIMER segmentation run (shared by her, not in her public repo) | Evaluation 2, MolScribe input |
| BioVista labels and papers | Labels downloaded by the setup script. Papers: the open-access BioVista papers in BioMiner's `example/pdfs`, plus any others you add | Evaluation 3 |

Papers are never committed. Several course papers carry no-derivatives licenses, and BioVista does not distribute its papers for copyright reasons.

Place the course data like this (or change the paths in `config/config.yaml`):

```
data/
├── course_papers/
│   ├── 1.pdf  2.pdf  4.pdf  6.pdf  16.pdf  19.pdf
│   └── ground_truth_SMILES_confirmed_1,2,4,6,16,19.csv
└── decimer_crops/               Crop images. File or folder names must include the paper ID
```

`crops.filename_pattern` reads the paper and page from each crop's path. If the crops come with a manifest CSV (`paper_id, page, crop_id, image_path`), set `crops.manifest_csv` instead. BioVista OCSR images are matched to their labels with the BioMiner authors' own rule, from `BioVista/component_infernece/ocsr/molscribe_inference_batch.py` in their repo: an image named `<paper>_..._<page>-<index>` belongs to the label row with that `pdf_name` and `index`. All 8,861 images match.

## Setup

### Option A: Google Colab (recommended)

Open `notebooks/run_on_colab.ipynb` in Colab with a GPU runtime (L4 or A100), set `DATA_DIR` in the settings cell, add the four API secrets, and run the cells in order. The notebook installs conda, creates the three environments, exports their exact versions to `envs/lock/`, runs the tests, runs the pipeline, and packages the results.

### Option B: Local machine with conda and an NVIDIA GPU

```bash
git clone https://github.com/Showrish/xn-ocsr-benchmark-1.2.git
cd xn-ocsr-benchmark-1.2

pip install -r requirements.txt
python scripts/setup_external.py                 # BioMiner, BioVista data, weights, teammate results

sed -i '/^prefix:/d' external/BioMiner/main_environment.yml   # drop the authors' local install path
conda env create -n BioMiner -f external/BioMiner/main_environment.yml
conda env create -f envs/molscribe.yml
conda env create -f envs/scoring.yml
python scripts/setup_external.py --steps mineru  # MinerU config and models, inside the BioMiner env

cp .env.example .env                             # then fill in your API credentials
```

### Why three environments

MolScribe pins older versions of its dependencies, BioMiner ships its own conda environment with MinerU 1.3.1, and scoring needs a specific RDKit version. Keeping them apart avoids version conflicts. `run_pipeline.py` launches each step in the right environment using the commands under `environments:` in `config/config.yaml`.

### Credentials

BioMiner calls an API model for coreference and Markush enumeration, because its own 32B model needs at least two 80 GB GPUs. Set these as environment variables, Colab secrets, or in an untracked `.env` file:

| Variable | Meaning |
|---|---|
| `BIOMINER_API_KEY` | API key for an OpenAI-compatible endpoint |
| `BIOMINER_BASE_URL` | The endpoint's base URL |
| `BIOMINER_TEXT_MODEL` | Model name for text tasks |
| `BIOMINER_VISION_MODEL` | Model name for image tasks |

At run time these are written into a copy of BioMiner's config in a temporary folder outside the repo, so keys are never committed. MolScribe needs no credentials.

## Running

```bash
python run_pipeline.py                            # all steps
python run_pipeline.py --steps ocsr_molscribe     # one step
python run_pipeline.py --steps evaluate           # rescore existing predictions
python run_pipeline.py --stop-on-error            # stop at the first failure
```

| Step | Environment | What it does |
|---|---|---|
| `ocsr_molscribe` | xn-molscribe | Runs MolScribe on the BioVista OCSR sample and saves SMILES, confidence, and time per image |
| `course_molscribe` | xn-molscribe | Runs MolScribe on every DECIMER crop of the course papers |
| `course_biominer` | BioMiner | Starts BioMiner's model servers, runs BioMiner on each course PDF, and collects every extracted structure |
| `biovista_biominer` | BioMiner | Runs BioMiner on the BioVista subset with `--biovista_evaluate` and saves its metrics next to the published ones |
| `evaluate` | xn-scoring | Scores every tool on the BioVista images and the course papers, and writes the results tables |

If a step fails, the others still run, and evaluation scores whichever predictions exist.

Tests for the scoring logic: `conda run -n xn-scoring python -m pytest -q tests`

## Metrics

Scoring follows the Module 1 protocol from our setup proposal: a prediction is correct only if its canonical SMILES (RDKit 2026.03.2) exactly matches the answer. There is no partial credit.

**Evaluation 1, BioVista images (one answer per image)**

| Metric | Definition |
|---|---|
| Exact match | Share of images whose prediction equals the correct SMILES |
| Exact match, no stereo | The same after removing stereochemistry, which separates stereo errors from wrong connectivity |
| Valid rate | Share of predictions that RDKit can parse |
| Mean Tanimoto | Average Morgan fingerprint similarity (radius 2, 2048 bits) between prediction and answer, with invalid predictions counted as 0 |
| Seconds per image | Inference time |

Images are grouped by their correct SMILES into `markush` (contains `*`), `chiral` (contains `@`), and `full` (neither). This approximates BioVista's Full, Chiral, and Markush groups and uses the same rule as Jirapa's DECIMER benchmark, so the groups line up.

**Evaluation 2, course papers (many answers per paper)**

| Metric | Definition |
|---|---|
| Precision, recall, F1 | Per paper, on sets of unique structures, so duplicates on either side count once |
| Equal-weight average | Mean over the six papers. A pooled score would mostly reflect paper 19, which holds 104 of the 220 unique structures |
| Stereo-free scores | The same scores after removing stereochemistry |
| Near misses | Wrong predictions with Tanimoto similarity of at least 0.85 to a true molecule, listed in `near_misses.csv` |
| Parseable rate, runtime | Share of parseable predictions, and seconds per paper |

The ground truth has 222 rows but 220 unique structures. In paper 4, compounds 42 and dAurA383 share a structure, and in paper 19, VMF and compound 1 do.

## Results

The headline tables are in [`results/summary.md`](results/summary.md), and every output file is described in [`results/README.md`](results/README.md).

**Status for this submission:** Evaluation 1 (MolScribe on the BioVista OCSR sample, side by side with DECIMER) was run, and its outputs are in `results/`. Evaluations 2 and 3 are fully implemented but were not run for this submission: BioMiner's reasoning step needs an API key, and the DECIMER crops of the course papers were not available in time. Both run with `python run_pipeline.py` once the credentials and crops are in place.

### Key results: BioVista OCSR images (100 images, seed 42)

Exact match of canonical SMILES. Published values are the BioMiner README's results on all 8,861 images.

| Group | Images | MolScribe | DECIMER | MolScribe, published | DECIMER, published |
|---|---|---|---|---|---|
| All | 100 | 0.460 | 0.300 | 0.455 | 0.298 |
| Full | 38 | 0.789 | 0.526 | 0.703 | 0.545 |
| Chiral | 19 | 0.474 | 0.526 | 0.481 | 0.326 |
| Markush | 43 | 0.163 | 0.000 | 0.156 | 0.000 |

- **The setup reproduces published results.** MolScribe's overall exact match (0.460) is within half a point of its published full-set value (0.455).
- **The comparison is on identical images.** The DECIMER rows reproduce Jirapa's reported numbers exactly, including the group sizes of 38, 19, and 43, which confirms both runs used the same 100 images and the same scoring.
- **MolScribe is ahead overall** (0.46 vs. 0.30), mostly on full structures (0.79 vs. 0.53) and Markush structures (0.16 vs. 0.00). DECIMER is slightly ahead on chiral structures (0.53 vs. 0.47), but that group has only 19 images.
- **Some errors are stereochemistry only.** Ignoring stereochemistry raises MolScribe from 0.46 to 0.50. MolScribe returned a valid SMILES for 94% of images.
- **Timings are not comparable across tools.** MolScribe ran on a Colab L4 GPU (about 0.9 seconds per image) and DECIMER on a CPU in Jirapa's run (about 5.1 seconds per image).

## Limitations

- Evaluation 1 uses 100 of the 8,861 BioVista images, so its numbers carry real sampling error and will not match the published full-set values exactly.
- The Full, Chiral, and Markush groups are an approximation from the SMILES text.
- The DECIMER numbers in the comparison come from Jirapa's run, not from this repo's code.
- Evaluation 3 uses a small subset of BioVista papers, so it is a sanity check on the installation rather than a full reproduction.
- BioMiner runs with an API model instead of its own BioMiner-Instruct model, which needs more GPU memory than Colab provides. It reports results per paper, so its runtime is split evenly across a paper's structures, and it gives no page numbers or confidence scores.

## References

- Gulluoglu, H. S. A., et al. (2026). Practical use of advanced AI frameworks on real-life scientific problems: Three case studies [Preprint]. bioRxiv. https://doi.org/10.64898/2026.06.23.734132
- Qian, Y., Guo, J., Tu, Z., Li, Z., Coley, C. W., & Barzilay, R. (2023). MolScribe: Robust molecular structure recognition with image-to-graph generation. Journal of Chemical Information and Modeling, 63(7), 1925-1934. https://doi.org/10.1021/acs.jcim.2c01480
- Rajan, K., Brinkhaus, H. O., Agea, M. I., Zielesny, A., & Steinbeck, C. (2023). DECIMER.ai: An open platform for automated optical chemical structure identification, segmentation and recognition in scientific publications. Nature Communications, 14(1), 5045. https://doi.org/10.1038/s41467-023-40782-0
- RDKit. (2026). RDKit: Open-source cheminformatics (Version 2026.3.2) [Computer software]. https://www.rdkit.org
- Yan, J., et al. (2026). BioMiner: A multi-modal system for automated mining of protein-ligand bioactivity data from literature (arXiv:2604.21508). arXiv. https://doi.org/10.48550/arXiv.2604.21508
- BioMiner code and BioVista data: https://github.com/jiaxianyan/BioMiner
- MolScribe code: https://github.com/thomas0809/MolScribe
- Bangvak, J. (2026). XN benchmark: DECIMER on the BioVista OCSR images [Computer software]. GitHub. https://github.com/JirapaB/xn-benchmark
