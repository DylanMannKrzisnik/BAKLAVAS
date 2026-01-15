# --- DROP-IN SCRIPT: build AnnData from *_RNA_matrix.csv.gz (in tar) + SpatialData from images/positions/scalefactors (in tar) ---

import os
import re
import io
import json
import gzip
import tarfile
from glob import glob

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import anndata as ad
from tqdm import tqdm

import scanpy as sc
import celltypist

# SpatialData stack
import imageio.v3 as iio
import geopandas as gpd
from shapely.geometry import Point
import spatialdata as sd
from spatialdata.models import Image2DModel, ShapesModel, TableModel
from spatialdata.transformations import Scale


# -------------------------
# CONFIG
# -------------------------
datapath = "/home/mcb/users/dmannk/BAKLAVA_base/data/MouseDev_Spatial_Triomic"

# RNA tar (counts)
rna_tarfilename = "GSE308623.tar"
rna_tarpath = os.path.join(datapath, rna_tarfilename)

# Spatial tar (images/coords/scalefactors)
spatial_tarfilename = "GSE308526.tar"
spatial_tarpath = os.path.join(datapath, spatial_tarfilename)

# Output
rna_h5ad_path = os.path.join(datapath, "rna_adata.h5ad")

# If you only want "developmental" P#S# samples, keep these regexes.
# If you want ALL samples, change developmental_*_pattern to r'_RNA_matrix\.csv\.gz$' etc.
developmental_rna_pattern = r"_P\d+S\d+_RNA_matrix\.csv\.gz$"
developmental_lowres_pattern = r"_P\d+S\d+_tissue_lowres_image\.png\.gz$"
developmental_hires_pattern = r"_P\d+S\d+_tissue_hires_image\.png\.gz$"
developmental_coords_pattern = r"_P\d+S\d+_tissue_positions_list\.csv\.gz$"
developmental_scalefactors_pattern = r"_P\d+S\d+_scalefactors_json\.json\.gz$"

KEEP_ONLY_IN_TISSUE = True  # typical
CELLTYPIST_MODEL = "Developing_Mouse_Brain.pkl"


# -------------------------
# HELPERS
# -------------------------
def sample_id_from_filename(fname: str, suffix: str) -> str:
    """
    Extract sample id robustly from filenames like:
      GSM9247574_00_P0S1_RNA_matrix.csv.gz           -> P0S1
      GSM9247597_09_LPC5S1_H3K27me3_RNA_matrix.csv.gz -> LPC5S1_H3K27me3
      GSM9247601_LPC5_Saggital_RNA_matrix.csv.gz      -> LPC5_Saggital
    """
    base = os.path.basename(fname)
    if not base.endswith(suffix):
        raise ValueError(f"Expected suffix {suffix}, got {base}")
    core = base[: -len(suffix)]
    parts = core.split("_")
    # if the second token is a numeric batch like "00", drop first two tokens; else drop first
    if len(parts) >= 2 and parts[1].isdigit():
        return "_".join(parts[2:])
    return "_".join(parts[1:])


def read_gz_member_bytes(tar: tarfile.TarFile, member: str) -> bytes:
    f = tar.extractfile(member)
    if f is None:
        raise FileNotFoundError(member)
    with gzip.GzipFile(fileobj=f) as gz:
        return gz.read()


def read_png_gz_from_tar(tar: tarfile.TarFile, member: str) -> np.ndarray:
    data = read_gz_member_bytes(tar, member)
    return iio.imread(io.BytesIO(data))


def read_json_gz_from_tar(tar: tarfile.TarFile, member: str) -> dict:
    data = read_gz_member_bytes(tar, member)
    return json.loads(data.decode("utf-8"))


def read_csv_gz_from_tar(tar: tarfile.TarFile, member: str, **kwargs) -> pd.DataFrame:
    f = tar.extractfile(member)
    if f is None:
        raise FileNotFoundError(member)
    with gzip.GzipFile(fileobj=f) as gz:
        return pd.read_csv(gz, **kwargs)


def looks_like_barcodes(idx: pd.Index) -> bool:
    s = pd.Index(idx).astype(str)
    # ACGT-heavy barcodes, sometimes with "-1"
    frac = s.to_series().str.match(r"^[ACGT]+(-\d+)?$").mean()
    return frac > 0.8


# -------------------------
# 1) RNA: build/load AnnData
# -------------------------
if os.path.exists(rna_h5ad_path):
    print("RNA data already processed, loading from file:", rna_h5ad_path)
    rna_adata = ad.read_h5ad(rna_h5ad_path)
else:
    with tarfile.open(rna_tarpath, "r:*") as tar:
        all_members = tar.getnames()
        rna_files = sorted([m for m in all_members if re.search(developmental_rna_pattern, m)])

        if len(rna_files) == 0:
            raise RuntimeError(f"No RNA files matched: {developmental_rna_pattern}")

        rna_adatas = []
        for member in tqdm(rna_files, desc="Reading RNA matrices"):
            sample_name = sample_id_from_filename(member, "_RNA_matrix.csv.gz")

            rna_df = read_csv_gz_from_tar(tar, member, index_col=0)

            # Decide which axis is barcodes by pattern (robust), else fallback
            if looks_like_barcodes(rna_df.index) and not looks_like_barcodes(rna_df.columns):
                pass
            elif looks_like_barcodes(rna_df.columns) and not looks_like_barcodes(rna_df.index):
                rna_df = rna_df.T
            else:
                # fallback to length heuristic if ambiguous
                idx_len_unique = pd.Index(rna_df.index.astype(str).str.len()).nunique()
                col_len_unique = pd.Index(rna_df.columns.astype(str).str.len()).nunique()
                if idx_len_unique == 1 and col_len_unique != 1:
                    pass
                elif col_len_unique == 1 and idx_len_unique != 1:
                    rna_df = rna_df.T
                else:
                    raise ValueError(f"Cannot determine barcode axis for {member}")

            # obs
            barcodes = rna_df.index.astype(str)
            obs = pd.DataFrame({"barcode": barcodes, "sample_name": sample_name}, index=barcodes)
            obs.index = obs["barcode"] + "_" + obs["sample_name"]

            # var
            genes = rna_df.columns.astype(str)
            var = pd.DataFrame({"gene_id": genes}, index=genes)

            adata = ad.AnnData(
                X=csr_matrix(rna_df.values),
                obs=obs,
                var=var,
            )

            rna_adatas.append(adata)

    rna_adata = ad.concat(rna_adatas, axis=0, join="inner", label=None, merge="same")
    rna_adata.write_h5ad(rna_h5ad_path)
    print("Wrote:", rna_h5ad_path)


# -------------------------
# 2) Basic processing + CellTypist
# -------------------------
# (If you prefer to keep raw counts for downstream, store a copy before normalization)
# rna_adata.layers["counts"] = rna_adata.X.copy()

sc.pp.filter_cells(rna_adata, min_genes=100)
sc.pp.filter_genes(rna_adata, min_cells=3)

sc.pp.normalize_total(rna_adata, target_sum=1e4)
sc.pp.log1p(rna_adata)
sc.pp.highly_variable_genes(rna_adata, n_top_genes=2000)

sc.tl.pca(rna_adata, svd_solver="arpack", use_highly_variable=True)
sc.pp.neighbors(rna_adata, use_rep="X_pca")

# CellTypist
pred = celltypist.annotate(rna_adata, model=CELLTYPIST_MODEL, majority_voting=False)
pred_df = pred.predicted_labels.copy()
pred_df["predicted_labels_broad"] = pred_df["predicted_labels"].astype(str).str.split(":").str[0]

# Join safely (no merge index weirdness)
rna_adata.obs = rna_adata.obs.join(pred_df, how="left")

# UMAP (optional plotting)
sc.tl.umap(rna_adata, min_dist=0.1)
# sc.pl.umap(rna_adata, color=["sample_name", "predicted_labels_broad"])


# -------------------------
# 3) Spatial: build per-sample SpatialData from tar members
# -------------------------
with tarfile.open(spatial_tarpath, "r:*") as tar:
    members = tar.getnames()

# index members by sample_id
lowres_map, hires_map, coords_map, scales_map = {}, {}, {}, {}
for m in members:
    if re.search(developmental_lowres_pattern, m):
        sid = sample_id_from_filename(m, "_tissue_lowres_image.png.gz")
        lowres_map[sid] = m
    if re.search(developmental_hires_pattern, m):
        sid = sample_id_from_filename(m, "_tissue_hires_image.png.gz")
        hires_map[sid] = m
    if re.search(developmental_coords_pattern, m):
        sid = sample_id_from_filename(m, "_tissue_positions_list.csv.gz")
        coords_map[sid] = m
    if re.search(developmental_scalefactors_pattern, m):
        sid = sample_id_from_filename(m, "_scalefactors_json.json.gz")
        scales_map[sid] = m

sample_ids = sorted(set(lowres_map) & set(hires_map) & set(coords_map) & set(scales_map))
if len(sample_ids) == 0:
    raise RuntimeError("No complete samples found (need lowres+hires+coords+scalefactors).")

# Only keep samples that exist in RNA
rna_samples = set(rna_adata.obs["sample_name"].unique())
sample_ids = [sid for sid in sample_ids if sid in rna_samples]
if len(sample_ids) == 0:
    raise RuntimeError("No overlap between spatial samples and rna_adata sample_name values.")

print(f"Building SpatialData for {len(sample_ids)} samples")

sdata_by_sample = {}

with tarfile.open(spatial_tarpath, "r:*") as tar:
    for sid in tqdm(sample_ids, desc="SpatialData per sample"):
        # ---- positions ----
        pos = read_csv_gz_from_tar(tar, coords_map[sid], header=None, index_col=0)
        pos.columns = ["in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
        pos.index = pos.index.astype(str)
        pos.index.name = "barcode"

        if KEEP_ONLY_IN_TISSUE:
            pos = pos[pos["in_tissue"] == 1].copy()

        # make obs_names to match RNA: barcode_sample
        pos["sample_name"] = sid
        pos["obs_name"] = pos.index + "_" + sid
        pos = pos.set_index("obs_name", drop=True)

        # ---- subset RNA AnnData to this sample, align to available spots ----
        adata_s = rna_adata[rna_adata.obs["sample_name"] == sid].copy()
        adata_s = adata_s[adata_s.obs_names.isin(pos.index)].copy()

        # join positions into obs
        adata_s.obs = adata_s.obs.join(
            pos[["in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]],
            how="left",
        )

        # obsm spatial: x=col, y=row
        adata_s.obsm["spatial"] = adata_s.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=float)

        # ---- scalefactors ----
        sf = read_json_gz_from_tar(tar, scales_map[sid])
        spot_radius_fullres = float(sf["spot_diameter_fullres"]) / 2.0
        low_s = float(sf["tissue_lowres_scalef"])
        hi_s = float(sf["tissue_hires_scalef"])

        # ---- shapes (circles) ----
        centers = [Point(x, y) for x, y in adata_s.obsm["spatial"]]
        gdf = gpd.GeoDataFrame(
            {"radius": spot_radius_fullres, "geometry": centers},
            index=adata_s.obs_names,
        )
        spots_name = f"spots_{sid}"
        spots = ShapesModel.parse(gdf)

        # ---- images ----
        img_low = read_png_gz_from_tar(tar, lowres_map[sid])
        img_hi = read_png_gz_from_tar(tar, hires_map[sid])

        dims_low = ("y", "x", "c") if img_low.ndim == 3 else ("y", "x")
        dims_hi = ("y", "x", "c") if img_hi.ndim == 3 else ("y", "x")

        # Transform lowres/hires pixel coords into the same "global" (fullres) coordinate system
        low_img = Image2DModel.parse(
            img_low,
            dims=dims_low,
            transformations={"global": Scale([1.0 / low_s, 1.0 / low_s], axes=("y", "x"))},
        )
        hi_img = Image2DModel.parse(
            img_hi,
            dims=dims_hi,
            transformations={"global": Scale([1.0 / hi_s, 1.0 / hi_s], axes=("y", "x"))},
        )

        # ---- link table <-> shapes ----
        adata_s.obs["region"] = spots_name
        adata_s.obs["instance_id"] = adata_s.obs_names

        table = TableModel.parse(
            adata_s,
            region=spots_name,
            region_key="region",
            instance_key="instance_id",
        )

        # ---- assemble SpatialData ----
        sdata = sd.SpatialData(
            images={f"lowres_{sid}": low_img, f"hires_{sid}": hi_img},
            shapes={spots_name: spots},
            tables={f"rna_{sid}": table},
        )

        sdata_by_sample[sid] = sdata

print("Done. SpatialData objects available in dict: sdata_by_sample")
print("Example keys:", list(sdata_by_sample.keys())[:5])
print("Example object:", sdata_by_sample[list(sdata_by_sample.keys())[0]])