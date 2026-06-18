#!/usr/bin/env bash
# One-time GCS bucket setup (run locally or on VM with gcloud configured).
#
# Usage:
#   export GCP_PROJECT_ID=your-project-id
#   export GCS_DATA_BUCKET=your-unique-bucket-name   # globally unique
#   export GCP_REGION=us-central1                      # optional
#   bash create_gcs_bucket.sh

set -euo pipefail

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
: "${GCS_DATA_BUCKET:?Set GCS_DATA_BUCKET (globally unique bucket name)}"
GCP_REGION="${GCP_REGION:-us-central1}"

gcloud config set project "${GCP_PROJECT_ID}"

if gsutil ls -b "gs://${GCS_DATA_BUCKET}" >/dev/null 2>&1; then
  echo "Bucket gs://${GCS_DATA_BUCKET} already exists."
else
  gcloud storage buckets create "gs://${GCS_DATA_BUCKET}" \
    --project="${GCP_PROJECT_ID}" \
    --location="${GCP_REGION}" \
    --uniform-bucket-level-access
  echo "Created gs://${GCS_DATA_BUCKET}"
fi

# Service account for this compute server (read-only).
SA_NAME="${GCS_READER_SA_NAME:-baklava-gcs-reader}"
SA_EMAIL="${SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

if ! gcloud iam service-accounts describe "${SA_EMAIL}" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${SA_NAME}" \
    --display-name="BAKLAVA GCS read-only"
fi

gcloud storage buckets add-iam-policy-binding "gs://${GCS_DATA_BUCKET}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectViewer"

KEY_PATH="${GCS_KEY_PATH:-${HOME}/.config/gcp/${SA_NAME}-key.json}"
mkdir -p "$(dirname "${KEY_PATH}")"
if [[ ! -f "${KEY_PATH}" ]]; then
  gcloud iam service-accounts keys create "${KEY_PATH}" \
    --iam-account="${SA_EMAIL}"
fi

echo ""
echo "Bucket ready: gs://${GCS_DATA_BUCKET}"
echo "Reader key:   ${KEY_PATH}"
echo ""
echo "On this server, add to .env:"
echo "  export GCS_DATA_BUCKET=${GCS_DATA_BUCKET}"
echo "  export GCS_DATA_PREFIX=BAKLAVA/data"
echo "  export GOOGLE_APPLICATION_CREDENTIALS=${KEY_PATH}"
echo "  export TARGET_DATASET=human_sead_mtg"
echo ""
echo "On the VM, upload data with:"
echo "  export GCS_DATA_BUCKET=${GCS_DATA_BUCKET}"
echo "  export LOCAL_SEA_AD_ROOT=/path/to/data/on/vm"
echo "  bash scripts/SMA/sync_sea_ad_to_gcs.sh"
