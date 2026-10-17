"""Evaluation metrics, following the Module 1 protocol in the 1.1 setup proposal.

A prediction counts as correct only when its canonical SMILES (RDKit) exactly
matches a ground truth molecule from the same paper. Matching is done on sets of
unique structures per paper, so duplicates on either side are not double counted.
"""

from __future__ import annotations

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

# Invalid SMILES are expected from OCSR tools. They are counted, not printed.
RDLogger.DisableLog("rdApp.*")


def to_mol(smiles: str | None):
    """Parse a SMILES string into an RDKit molecule, or return None if it is invalid."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles.strip())


def canonical_smiles(smiles: str | None, keep_stereo: bool = True) -> str | None:
    """Canonical SMILES for exact matching. keep_stereo=False drops stereochemistry."""
    molecule = to_mol(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, isomericSmiles=keep_stereo)


def set_scores(predicted: set[str], truth: set[str]) -> dict[str, float]:
    """Precision, recall, and F1 between a predicted set and a ground truth set."""
    true_positives = len(predicted & truth)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": true_positives, "precision": precision, "recall": recall, "f1": f1}


class TanimotoScorer:
    """Morgan fingerprint similarity, used to flag near misses for manual review."""

    def __init__(self, radius: int = 2, n_bits: int = 2048):
        self.generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    def fingerprint(self, smiles: str):
        """Fingerprint for a SMILES string, or None if it cannot be parsed."""
        molecule = to_mol(smiles)
        return None if molecule is None else self.generator.GetFingerprint(molecule)

    def best_match(self, smiles: str, reference: dict[str, object]) -> tuple[float, str | None]:
        """Highest similarity between one molecule and a set of reference fingerprints."""
        query = self.fingerprint(smiles)
        if query is None or not reference:
            return 0.0, None
        best_smiles, best_score = None, 0.0
        for reference_smiles, reference_fp in reference.items():
            score = DataStructs.TanimotoSimilarity(query, reference_fp)
            if score > best_score:
                best_smiles, best_score = reference_smiles, score
        return best_score, best_smiles
