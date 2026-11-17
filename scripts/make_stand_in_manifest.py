"""Build a stand-in gated_manifest.csv from the DECIMER crops.

Lets stage 3 be developed and tested before stage 2's gate is ready, as agreed in
the Assignment 1.4 plan. Every crop is labelled "molecule" because there is no
gate yet. Expects crops named like Jirapa's: paper_19/p004_c01.png.

    python scripts/make_stand_in_manifest.py
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd

CROP_NAME = re.compile(r"paper_(?P<paper>\d+)/p(?P<page>\d+)_c(?P<crop>\d+)\.png$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crops", default="data/decimer_crops", help="Folder with paper_<id>/p<page>_c<crop>.png")
    parser.add_argument("--out", default="data/stage2/gated_manifest.csv", help="Where to write the manifest")
    args = parser.parse_args()

    crops, out = Path(args.crops), Path(args.out)
    rows = []
    for path in sorted(crops.rglob("*.png")):
        match = CROP_NAME.search(path.as_posix())
        if match:
            rows.append({
                "paper_id": int(match["paper"]), "page": int(match["page"]), "crop_id": int(match["crop"]),
                "x": "", "y": "", "width": "", "height": "",
                "crop_path": os.path.relpath(path, out.parent), "label": "molecule", "gate_score": "",
            })
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} crops to {out}")


if __name__ == "__main__":
    main()
