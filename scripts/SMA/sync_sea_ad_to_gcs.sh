#!/usr/bin/env bash
# Run this on the GCP VM where SEA-AD data lives.
#
# Prerequisites on the VM:
#   curl https://sdk.cloud.google.com | bash   # or apt install google-cloud-cli
#   gcloud auth login
#   gcloud config set project YOUR_PROJECT_ID
#
# Usage:
#   export GCS_DATA_BUCKET=your-bucket-name
#   export GCS_DATA_PREFIX=BAKLAVA/data          # optional
#   export LOCAL_SEA_AD_ROOT=/path/to/data/on/vm # root containing Spatial_ATAC_RNA/ or SEA_AD/
#   bash sync_sea_ad_to_gcs.sh
#
# For script 4__project_target_data_SMA.py (human_sead_mtg), upload at least:
#   ${LOCAL_SEA_AD_ROOT}/Spatial_ATAC_RNA/human/sead_mtg/sead_mtg.h5ad
# to:
#   gs://${GCS_DATA_BUCKET}/${GCS_DATA_PREFIX}/Spatial_ATAC_RNA/human/sead_mtg/sead_mtg.h5ad

set -euo pipefail

: "${GCS_DATA_BUCKET:?Set GCS_DATA_BUCKET to your GCS bucket name}"
GCS_DATA_PREFIX="${GCS_DATA_PREFIX:-BAKLAVA/data}"
LOCAL_SEA_AD_ROOT="${LOCAL_SEA_AD_ROOT:?Set LOCAL_SEA_AD_ROOT to the data directory on this VM}"

GCS_ROOT="gs://${GCS_DATA_BUCKET}/${GCS_DATA_PREFIX}"

echo "Syncing ${LOCAL_SEA_AD_ROOT} -> ${GCS_ROOT}/"
gsutil -m rsync -r "${LOCAL_SEA_AD_ROOT}/" "${GCS_ROOT}/"

echo "Done. On the compute server, set in .env:"
echo "  export GCS_DATA_BUCKET=${GCS_DATA_BUCKET}"
echo "  export GCS_DATA_PREFIX=${GCS_DATA_PREFIX}"
echo "  export TARGET_DATASET=human_sead_mtg"
