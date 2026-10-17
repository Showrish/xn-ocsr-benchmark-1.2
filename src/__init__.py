"""Benchmarking pipeline for BioMiner and MolScribe on the XN course papers and BioVista."""

# Shared results schema from the group's 1.1 setup proposal. Every tool writes its
# predictions with these columns so one scoring script can read all of them.
PREDICTION_COLUMNS = ["paper_id", "page", "crop_id", "predicted_smiles", "tool", "runtime_sec"]

# Extra columns kept so no tool output is thrown away (confidence scores, the
# compound label BioMiner linked to a structure, and which raw file a row came from).
EXTRA_COLUMNS = ["confidence", "label", "source", "error"]
