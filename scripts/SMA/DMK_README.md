# SMA RNA-MSI h5mu Export

Concise workflow for creating Python/Muon-ready `.h5mu` files from the SMA aligned RNA-MSI data.

Paths below are relative to the directory containing `BAKLAVA_base`.

## Inputs

There are two alignment workflows, depending on sample type:

### Murine samples (default)

Run or reproduce the alignment workflow in:

`BAKLAVA_base/sma/scripts/MSI_SRT_mPD_LL.Rmd`

This creates:

- `BAKLAVA_base/outputs/SMA/R_objects/se.multi.list`
- `BAKLAVA_base/outputs/SMA/R_objects/knn_spatial_df_filtered_list`

### Human samples (V11T17-102)

Run or reproduce the alignment workflow in:

`BAKLAVA_base/sma/scripts/MSI_SRT_hPDStr_LLMV.Rmd`

This creates:

- `BAKLAVA_base/outputs/SMA/R_objects/hPDStr.multi.list`
- `BAKLAVA_base/outputs/SMA/results/tables/<sample_id>_nearest_neighbors.csv` (one per section)

`se.multi.list` / `hPDStr.multi.list` contain paired RNA and MSI assays. Alignment mapping is stored in `knn_spatial_df_filtered_list` (murine) or the `*_nearest_neighbors.csv` tables (human).

## Step 1: Export Seurat Objects

### Murine samples

```bash
Rscript BAKLAVA_base/BAKLAVAS/scripts/SMA/export_se_multi_to_mtx.R
```

### Human V11T17-102 sections (A1, B1, C1, D1)

Use `hPDStr.multi.list` and the nearest-neighbor CSVs written by `MSI_SRT_hPDStr_LLMV.Rmd`:

```bash
Rscript BAKLAVA_base/BAKLAVAS/scripts/SMA/export_se_multi_to_mtx.R \
  --input-rds hPDStr.multi.list \
  --from-neighbors-csv "BAKLAVA_base/outputs/SMA/results/tables/V11T17-102_*_nearest_neighbors.csv"
```

To export only specific sections, pass `--samples` explicitly:

```bash
Rscript BAKLAVA_base/BAKLAVAS/scripts/SMA/export_se_multi_to_mtx.R \
  --input-rds hPDStr.multi.list \
  --samples V11T17-102_A1,V11T17-102_B1,V11T17-102_C1,V11T17-102_D1 \
  --from-neighbors-csv "BAKLAVA_base/outputs/SMA/results/tables/V11T17-102_*_nearest_neighbors.csv"
```

Both workflows write per-sample Matrix Market files to:

`BAKLAVA_base/outputs/SMA/h5mu_export/<sample_id>/`

Each sample directory contains:

`rna.mtx`, `msi.mtx`, `rna_features.tsv`, `msi_features.tsv`, `barcodes.tsv`, `obs.tsv`

`obs.tsv` includes alignment provenance: `msi_barcode`, `alignment_distance`, and RNA/MSI warped coordinates.

## Step 2: Build h5mu Files

### Build all exported samples

```bash
/Users/dmannk/cisformer/envs/torch_env_py39/bin/python \
  BAKLAVA_base/BAKLAVAS/scripts/SMA/build_mudata_from_mtx.py
```

### Build only the human V11T17-102 sections

```bash
/Users/dmannk/cisformer/envs/torch_env_py39/bin/python \
  BAKLAVA_base/BAKLAVAS/scripts/SMA/build_mudata_from_mtx.py \
  --sample-glob 'V11T17-102*'
```

You can also list sections explicitly:

```bash
/Users/dmannk/cisformer/envs/torch_env_py39/bin/python \
  BAKLAVA_base/BAKLAVAS/scripts/SMA/build_mudata_from_mtx.py \
  --samples V11T17-102_A1,V11T17-102_B1,V11T17-102_C1,V11T17-102_D1
```

This writes one `.h5mu` file per sample:

`BAKLAVA_base/outputs/SMA/h5mu_export/<sample_id>.h5mu`

Each file has two modalities:

`mdata.mod["rna"]`

`mdata.mod["msi"]`

Raw counts are stored in `.X` and `.layers["counts"]`.

## Step 3: Load in Python

```python
import sys
sys.path.insert(0, "BAKLAVA_base/outputs/SMA/scripts")

from load_aligned_mudata import load_sample, list_samples

samples = list_samples()
mdata = load_sample(samples[0])
```

For direct Muon loading:

```python
import mudata as mu

mdata = mu.read_h5mu(
    "BAKLAVA_base/outputs/SMA/h5mu_export/V11L12-038_A1.h5mu"
)
```

## Files

`export_se_multi_to_mtx.R`: Seurat RDS to MTX/TSV export.

`build_mudata_from_mtx.py`: MTX/TSV to `.h5mu`.

`load_aligned_mudata.py`: Convenience loader for generated `.h5mu` files.