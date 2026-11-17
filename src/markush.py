"""Helpers for the Markush branch (stage 3b of the Phase 2 pipeline).

DECIMER writes a variable group as a placeholder such as [R] or [R1]. RDKit cannot
parse those, and in our Phase 1 benchmark 10 of DECIMER's 12 invalid outputs on
Markush images were placeholders like these. Rewriting them as SMILES attachment
points ([*], [1*]) turns the scaffold into a valid molecule that later stages can use.
"""

from __future__ import annotations

import re

# [R], [R1], [R2] ... but not real elements such as [Ru], [Rh] or [Rb]
R_GROUP = re.compile(r"\[R(\d*)\]")


def normalize_attachment_points(smiles: str) -> str:
    """Replace R-group placeholders with attachment points RDKit can parse."""
    if not isinstance(smiles, str):
        return ""
    return R_GROUP.sub(lambda m: f"[{m.group(1)}*]" if m.group(1) else "[*]", smiles)


def attachment_points(smiles: str) -> int:
    """Number of attachment points in a SMILES string."""
    return smiles.count("*") if isinstance(smiles, str) else 0
