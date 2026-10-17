# Benchmark Results

## BioVista OCSR images (100 images)

Share of images whose prediction exactly matches the correct SMILES after RDKit canonicalization.
Published values are from the BioMiner README for all 8,861 images.

| tool | group | images | exact_match | exact_match_no_stereo | valid_rate | mean_tanimoto | mean_runtime_sec | published_exact_match |
|---|---|---|---|---|---|---|---|---|
| DECIMER | all | 100 | 0.300 | 0.340 | 0.880 | 0.684 | 5.101 | 0.298 |
| MolScribe | all | 100 | 0.460 | 0.500 | 0.940 | 0.747 | 0.882 | 0.455 |
| DECIMER | full | 38 | 0.526 | 0.553 | 1.000 | 0.944 | 6.432 | 0.545 |
| MolScribe | full | 38 | 0.789 | 0.816 | 0.947 | 0.917 | 0.957 | 0.703 |
| DECIMER | chiral | 19 | 0.526 | 0.684 | 1.000 | 0.933 | 7.878 | 0.326 |
| MolScribe | chiral | 19 | 0.474 | 0.579 | 0.789 | 0.711 | 1.458 | 0.481 |
| DECIMER | markush | 43 | 0.000 | 0.000 | 0.721 | 0.345 | 2.699 | 0.000 |
| MolScribe | markush | 43 | 0.163 | 0.186 | 1.000 | 0.613 | 0.562 | 0.156 |
