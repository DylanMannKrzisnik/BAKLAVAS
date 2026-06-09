#!/usr/bin/env bash
#
# Build MSI AnnData (.h5ad) for the V11L12-038 slide of the SMA dataset,
# for BOTH matrices:
#   * 9-AA  (metabolites)  -> capture area D1
#   * DHB   (lipids)       -> capture areas A1 and B1
#
# Prereqs:
#   pip install anndata pandas numpy scipy --break-system-packages
#
# Data: download the SMA processed bundle from Mendeley Data (w7nw4km7xd),
#   https://data.mendeley.com/datasets/w7nw4km7xd/1
# unzip it, and point DATA_DIR at the unpacked "data" folder. Inside it the
# MSI CSVs live under .../sma/V11L12-038/V11L12-038_<AREA>/output_data/V11L12-038_<AREA>_MSI/
#
set -euo pipefail

# ---- edit this to your unpacked Mendeley data folder ----
DATA_DIR="${1:-./data}"
OUT_DIR="${2:-./msi_h5ad}"
mkdir -p "$OUT_DIR"

SMA="$DATA_DIR/sma/V11L12-038"

# 9-AA metabolites (capture area D1)
python sma_msi_to_h5ad.py \
  "$SMA/V11L12-038_D1/output_data/V11L12-038_D1_MSI/V11L12-038_Mouse_D1.Visium.9aa.220826_smamsi.csv" \
  --sample-id V11L12-038_D1 --matrix 9-AA --modality metabolites \
  -o "$OUT_DIR/V11L12-038_D1_9AA_metabolites.h5ad"

# DHB lipids (capture area A1)
python sma_msi_to_h5ad.py \
  "$SMA/V11L12-038_A1/output_data/V11L12-038_A1_MSI/V11L12-038_A1.Visium.DHB.220826_smamsi.csv" \
  --sample-id V11L12-038_A1 --matrix DHB --modality lipids \
  -o "$OUT_DIR/V11L12-038_A1_DHB_lipids.h5ad"

# DHB lipids (capture area B1)
python sma_msi_to_h5ad.py \
  "$SMA/V11L12-038_B1/output_data/V11L12-038_B1_MSI/V11L12-038_B1.Visium.DHB.220826_smamsi.csv" \
  --sample-id V11L12-038_B1 --matrix DHB --modality lipids \
  -o "$OUT_DIR/V11L12-038_B1_DHB_lipids.h5ad"

echo "Done. h5ad files written to $OUT_DIR/"