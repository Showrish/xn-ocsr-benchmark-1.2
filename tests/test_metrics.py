"""Unit tests for the scoring logic. Run with: pytest tests/ (inside xn-scoring)."""

import pandas as pd
import pytest

from src.evaluate import score_paper
from src.metrics import TanimotoScorer, canonical_smiles, set_scores


def test_canonical_smiles_matches_different_spellings():
    # Ethanol written two ways should canonicalize to the same string
    assert canonical_smiles("OCC") == canonical_smiles("CCO")


def test_invalid_smiles_returns_none():
    assert canonical_smiles("not_a_smiles(") is None
    assert canonical_smiles("") is None


def test_stereo_removal():
    with_stereo = canonical_smiles("C[C@H](N)O")
    without_stereo = canonical_smiles("C[C@H](N)O", keep_stereo=False)
    assert "@" in with_stereo
    assert "@" not in without_stereo


def test_set_scores():
    scores = set_scores({"A", "B", "C"}, {"A", "B", "D", "E"})
    assert scores["tp"] == 2
    assert scores["precision"] == pytest.approx(2 / 3)
    assert scores["recall"] == pytest.approx(0.5)
    assert scores["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))


def test_empty_predictions_score_zero():
    assert set_scores(set(), {"A"}) == {"tp": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_identical_molecules_have_tanimoto_one():
    scorer = TanimotoScorer()
    reference = {"CCO": scorer.fingerprint("CCO")}
    score, closest = scorer.best_match("OCC", reference)
    assert score == pytest.approx(1.0)
    assert closest == "CCO"


def test_score_paper_counts_duplicates_once():
    truth_smiles = {canonical_smiles("CCO"), canonical_smiles("c1ccccc1")}
    truth = {"stereo": truth_smiles, "no_stereo": truth_smiles}
    predictions = pd.DataFrame(
        {"predicted_smiles": ["OCC", "CCO", "CCN", "bad("], "runtime_sec": [0.1, 0.1, 0.1, 0.1]}
    )
    row, _ = score_paper(predictions, truth, TanimotoScorer(), near_miss_threshold=0.85)
    assert row["tp"] == 1  # ethanol predicted twice still counts once
    assert row["n_unique_valid"] == 2  # ethanol and ethylamine
    assert row["parseable_rate"] == pytest.approx(3 / 4)
    assert row["recall"] == pytest.approx(0.5)


def test_structure_groups_follow_biovista_categories():
    from src.biovista_ocsr import structure_group

    assert structure_group("*CC(=O)O") == "markush"
    assert structure_group("C[C@H](N)O") == "chiral"
    assert structure_group("CCO") == "full"


def test_score_image_exact_invalid_and_stereo():
    from src.evaluate import score_image

    scorer = TanimotoScorer()
    assert score_image("OCC", "CCO", scorer)["exact_match"] == 1
    invalid = score_image("bad(", "CCO", scorer)
    assert invalid["valid"] == 0 and invalid["exact_match"] == 0 and invalid["tanimoto"] == 0.0
    stereo_only = score_image("C[C@@H](N)O", "C[C@H](N)O", scorer)
    assert stereo_only["exact_match"] == 0 and stereo_only["exact_match_no_stereo"] == 1
