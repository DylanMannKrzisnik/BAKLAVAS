# SMA RNA-MSI h5mu Export

Concise workflow for creating Python/Muon-ready `.h5mu` files from the SMA aligned RNA-MSI data.

Paths below are relative to the directory containing `BAKLAVA_base`.

## Inputs

Run or reproduce the alignment workflow in:

`BAKLAVA_base/sma/scripts/MSI_SRT_mPD_LL.Rmd`

This must create these R objects:

`BAKLAVA_base/outputs/SMA/R_objects/se.multi.list`

`BAKLAVA_base/outputs/SMA/R_objects/knn_spatial_df_filtered_list`

`se.multi.list` contains paired RNA and MSI assays. `knn_spatial_df_filtered_list` contains the RNA-MSI alignment mapping.

## Step 1: Export Seurat Objects

```bash
Rscript BAKLAVA_base/outputs/SMA/scripts/export_se_multi_to_mtx.R
```

This writes per-sample Matrix Market files to:

`BAKLAVA_base/outputs/SMA/h5mu_export/<sample_id>/`

Each sample directory contains:

`rna.mtx`, `msi.mtx`, `rna_features.tsv`, `msi_features.tsv`, `barcodes.tsv`, `obs.tsv`

`obs.tsv` includes alignment provenance: `msi_barcode`, `alignment_distance`, and RNA/MSI warped coordinates.

## Step 2: Build h5mu Files

```bash
/Users/dmannk/cisformer/envs/torch_env_py39/bin/python \
  BAKLAVA_base/outputs/SMA/scripts/build_mudata_from_mtx.py
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
