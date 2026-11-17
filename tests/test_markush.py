"""Tests for the Markush helpers. Run in the xn-scoring environment (needs RDKit)."""

from rdkit import Chem

from src.markush import attachment_points, normalize_attachment_points

# Real DECIMER outputs from our Phase 1 BioVista run, which RDKit rejected
DECIMER_OUTPUTS = {
    "CN1CC(CC1=O)[R]": "CN1CC(CC1=O)[*]",
    "C1=CC=C(C=C1)CN2C3=CC(=CC=C3NC4=C(C=C(C=C4)[R1])C2=O)F": "C1=CC=C(C=C1)CN2C3=CC(=CC=C3NC4=C(C=C(C=C4)[1*])C2=O)F",
}


def test_placeholders_become_valid_attachment_points():
    for raw, expected in DECIMER_OUTPUTS.items():
        fixed = normalize_attachment_points(raw)
        assert fixed == expected
        assert Chem.MolFromSmiles(raw) is None
        assert Chem.MolFromSmiles(fixed) is not None


def test_real_elements_are_untouched():
    assert normalize_attachment_points("[Ru]C[Rh]C[Rb]") == "[Ru]C[Rh]C[Rb]"


def test_attachment_point_count_and_empty_input():
    assert attachment_points("CN1CC(CC1=O)[*]") == 1
    assert attachment_points("CCO") == 0
    assert normalize_attachment_points(None) == ""
