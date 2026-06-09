#!/usr/bin/env python3
"""
sma_pair_msi_visium.py
=====================

Build a *paired* AnnData where SMA MALDI-MSI intensities are mapped onto the
matching 10x Visium spots, reproducing the strategy used in the SMA paper
(`marcovito/sma -> scripts/MSI_SRT_mPD_LL.Rmd`): for every Visium spot, take the
nearest MSI pixel (within a distance cutoff) and copy its m/z intensities.

The result is what a MISO-style `sma_msi.h5ad` actually is: MSI features living
on Visium spots, so `obsm["spatial"]` / `uns["spatial"]` come from Visium (with
the H&E image) and `.X` holds the MSI intensities.

IMPORTANT - coordinate alignment
--------------------------------
The MSI grid coordinates and the Visium spot pixel coordinates are NOT in the
same frame out of the box. In the original paper this was solved by a *manual*
rotation/translation/warp step (STUtility `ManualAlignImages`, a GUI). There is
no fully automatic substitute. This script therefore supports two modes:

  (A) --neighbors NN.csv
      Use the authors' exported nearest-neighbour table
      (`results/tables/<sample>_nearest_neighbors.csv`, columns `from`,`to`)
      if you have run their R pipeline. `from` = MSI pixel id ("<x>x<y>"),
      `to` = Visium barcode. This is the exact, paper-faithful mapping.

  (B) --affine "a b c d e f"  (optional 2x3 affine applied to MSI coords)
      Provide your own alignment (e.g. estimated once in napari/QuPato or by
      landmark matching), then the script does kNN matching in the shared frame.
      Without an affine it assumes coordinates are already comparable, which is
      usually only true for quick sanity checks, not real data.

Usage
-----
    # Faithful mapping using the authors' NN table:
    python sma_pair_msi_visium.py \
        --visium-dir .../V11L12-038_D1/output_data/V11L12-038_D1_RNA/outs \
        --msi-csv   .../V11L12-038_Mouse_D1.Visium.9aa.220826_smamsi.csv \
        --neighbors .../V11L12-038_D1_nearest_neighbors.csv \
        --sample-id V11L12-038_D1 -o V11L12-038_D1_paired.h5ad

    # kNN matching after supplying an affine alignment of the MSI grid:
    python sma_pair_msi_visium.py \
        --visium-dir .../outs --msi-csv .../*_smamsi.csv \
        --affine "0 -20 4000 20 0 1000" --max-dist 35 \
        --sample-id V11L12-038_D1 -o V11L12-038_D1_paired.h5ad
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from sma_msi_to_h5ad import load_msi_csv  # reuse the validated reader


def read_visium(visium_dir):
    """Load Space Ranger output as AnnData via scanpy."""
    import scanpy as sc

    adata = sc.read_visium(visium_dir)
    adata.var_names_make_unique()
    return adata


def apply_affine(coords, affine):
    """coords: (n,2); affine: iterable of 6 numbers [a,b,c,d,e,f] mapping
    (x,y) -> (a*x + b*y + c, d*x + e*y + f)."""
    a, b, c, d, e, f = [float(v) for v in affine]
    x, y = coords[:, 0], coords[:, 1]
    return np.column_stack([a * x + b * y + c, d * x + e * y + f])


def pair_by_neighbors_csv(visium, msi_X, msi_ids, mz_values, nn_csv, sample_id):
    """Use authors' nearest-neighbour table (columns `from`=MSI id, `to`=Visium barcode)."""
    import anndata as ad

    nn = pd.read_csv(nn_csv)
    if not {"from", "to"}.issubset(nn.columns):
        raise ValueError("neighbors csv must have 'from' (MSI id) and 'to' (Visium barcode) columns")

    msi_index = {mid: i for i, mid in enumerate(msi_ids)}
    keep = nn[nn["to"].isin(set(visium.obs_names)) & nn["from"].isin(msi_index)].copy()
    keep = keep.drop_duplicates(subset="to")

    spots = keep["to"].to_numpy()
    rows = np.array([msi_index[m] for m in keep["from"]])

    sub = visium[spots].copy()
    msi_layer = msi_X[rows, :]

    paired = ad.AnnData(X=sub.X.copy(), obs=sub.obs.copy(), var=sub.var.copy())
    paired.obsm = {k: v for k, v in sub.obsm.items()}
    paired.uns["spatial"] = sub.uns.get("spatial", {})
    paired.obsm["X_spatial"] = sub.obsm.get("spatial")
    paired.layers = {}
    paired.obsm["msi"] = np.asarray(msi_layer.todense()) if hasattr(msi_layer, "todense") else np.asarray(msi_layer)
    paired.uns["msi_var"] = list(mz_values)
    paired.uns["pairing"] = {"method": "authors_nearest_neighbors_csv", "source": os.path.basename(nn_csv)}
    return paired


def pair_by_knn(visium, msi_coords, msi_X, mz_values, sample_id, max_dist):
    """kNN match: for each Visium spot find nearest MSI pixel within max_dist."""
    import anndata as ad
    from sklearn.neighbors import NearestNeighbors

    spot_xy = visium.obsm["spatial"].astype(float)
    nn = NearestNeighbors(n_neighbors=1).fit(msi_coords)
    dist, idx = nn.kneighbors(spot_xy)
    dist = dist.ravel()
    idx = idx.ravel()

    mask = dist <= max_dist
    if mask.sum() == 0:
        raise RuntimeError(
            "No Visium spot found an MSI pixel within --max-dist. The two "
            "coordinate frames are almost certainly not aligned; supply an "
            "--affine alignment or use --neighbors with the authors' table."
        )

    sub = visium[mask].copy()
    rows = idx[mask]
    msi_layer = msi_X[rows, :]

    paired = ad.AnnData(X=sub.X.copy(), obs=sub.obs.copy(), var=sub.var.copy())
    paired.obsm = {k: v for k, v in sub.obsm.items()}
    paired.uns["spatial"] = sub.uns.get("spatial", {})
    paired.obs["msi_match_dist"] = dist[mask]
    paired.obsm["msi"] = np.asarray(msi_layer.todense()) if hasattr(msi_layer, "todense") else np.asarray(msi_layer)
    paired.uns["msi_var"] = list(mz_values)
    paired.uns["pairing"] = {"method": "knn", "max_dist": max_dist, "matched": int(mask.sum())}
    return paired


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--visium-dir", required=True, help="Space Ranger 'outs' directory")
    p.add_argument("--msi-csv", required=True, help="SMA *_smamsi.csv")
    p.add_argument("--sample-id", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--neighbors", help="authors' <sample>_nearest_neighbors.csv (mode A)")
    p.add_argument("--affine", help="6 numbers 'a b c d e f' to transform MSI coords (mode B)")
    p.add_argument("--max-dist", type=float, default=35.0, help="kNN distance cutoff (mode B)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    coords, intensities, mz_values = load_msi_csv(args.msi_csv)
    msi_ids = [f"{int(round(x))}x{int(round(y))}" for x, y in coords]

    visium = read_visium(args.visium_dir)

    if args.neighbors:
        paired = pair_by_neighbors_csv(
            visium, intensities, msi_ids, mz_values, args.neighbors, args.sample_id
        )
    else:
        msi_coords = coords
        if args.affine:
            msi_coords = apply_affine(coords, args.affine.split())
        paired = pair_by_knn(
            visium, msi_coords, intensities, mz_values, args.sample_id, args.max_dist
        )

    paired.write_h5ad(args.output)
    print(f"[ok] paired object -> {args.output}")
    print(f"     {paired.n_obs} spots | {paired.n_vars} genes | "
          f"{paired.obsm['msi'].shape[1]} m/z features in obsm['msi']")
    print(f"     pairing: {paired.uns['pairing']}")
    return paired


if __name__ == "__main__":
    main()