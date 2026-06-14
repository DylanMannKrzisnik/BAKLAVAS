#!/usr/bin/env python3
"""
sma_msi_to_h5ad.py
==================

Convert a Spatial Multimodal Analysis (SMA) MALDI-MSI table (`*_smamsi.csv`)
into an AnnData `.h5ad` object, analogous in structure to the `sma_msi.h5ad`
demo file distributed with the MISO paper (figshare file 44624971).

Background
----------
The SMA project (Vicari, Mirzazadeh et al., Nat Biotechnol 2024) stores its
MALDI-MSI data as one CSV per tissue section. Based on the authors' own export
code (`marcovito/sma -> scripts/making_csv_files.py`) and the reader in
`MSI_SRT_mPD_LL.Rmd`, every `*_smamsi.csv` has this layout:

    x, y, <mz_1>, <mz_2>, ..., <mz_k>
    0,  0,  0.0 ,  12.3 , ...,  0.0
    0,  1,  ...

  * column 1 ("x") and column 2 ("y") are the *integer grid coordinates*
    of each MSI pixel;
  * every remaining column is one m/z feature, the column header being the
    feature's m/z value ( (mz_low + mz_high) / 2 );
  * each row is one MSI pixel.

This script reproduces the authors' reader and packages the result as AnnData:

  * adata.X                 -> pixels x m/z intensity matrix (float32)
  * adata.obs_names         -> "<x>x<y>"  (same convention as the SMA R code)
  * adata.var_names         -> m/z values as strings (made unique)
  * adata.var["mz"]         -> m/z as float
  * adata.obs["array_x/y"]  -> raw grid coordinates
  * adata.obsm["spatial"]   -> (x, y) float array (squidpy/scanpy convention)
  * adata.uns["spatial"][sample_id] -> small metadata block
  * basic QC in adata.obs: total_intensity, n_features

For the V11L12-038 slide the relevant inputs are:
    9-AA  (metabolites) : V11L12-038_Mouse_D1.Visium.9aa.220826_smamsi.csv   (capture area D1)
    DHB   (lipids)      : V11L12-038_A1.Visium.DHB.220826_smamsi.csv          (capture area A1)
    DHB   (lipids)      : V11L12-038_B1.Visium.DHB.220826_smamsi.csv          (capture area B1)
(There is no DHB run on D1 and no 9-AA run on A1/B1 - they are different
capture areas of the same Visium slide.)

Usage
-----
    python sma_msi_to_h5ad.py INPUT.csv -o OUTPUT.h5ad \
        --sample-id V11L12-038_D1 --matrix 9-AA --modality metabolites

Run `python sma_msi_to_h5ad.py --help` for all options.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


def _find_coord_columns(df):
    """Return the names of the two coordinate columns (x, y), being tolerant
    about case. The SMA export writes them lowercase as the first two columns."""
    lower = {c.lower(): c for c in df.columns}
    if "x" in lower and "y" in lower:
        return lower["x"], lower["y"]
    # fall back to first two columns
    return df.columns[0], df.columns[1]


def load_msi_csv(path):
    """Read a *_smamsi.csv into (coords, intensities, mz_values).

    coords      : (n_pixels, 2) float array of (x, y) grid coordinates
    intensities : (n_pixels, n_features) float32 array
    mz_values   : list[str] of the m/z column headers
    """
    df = pd.read_csv(path)
    if df.shape[1] < 3:
        raise ValueError(
            f"{path!r} has only {df.shape[1]} columns; expected 'x','y' plus "
            "one column per m/z feature."
        )

    xcol, ycol = _find_coord_columns(df)
    coords = df[[xcol, ycol]].to_numpy(dtype=float)

    mz_cols = [c for c in df.columns if c not in (xcol, ycol)]
    intensities = df[mz_cols].to_numpy(dtype=np.float32)
    intensities = np.nan_to_num(intensities, nan=0.0)

    mz_values = [str(c) for c in mz_cols]
    return coords, intensities, mz_values


def build_anndata(
    coords,
    intensities,
    mz_values,
    sample_id,
    matrix=None,
    modality=None,
    sparse=False,
):
    import anndata as ad

    n_pixels = coords.shape[0]

    obs = pd.DataFrame(
        {
            "array_x": coords[:, 0],
            "array_y": coords[:, 1],
            "sample_id": sample_id,
        }
    )
    obs.index = [f"{int(round(x))}x{int(round(y))}" for x, y in coords]

    var = pd.DataFrame(index=mz_values)
    # m/z headers are numeric; keep the float value too for convenience
    var["mz"] = pd.to_numeric(pd.Series(mz_values, index=mz_values), errors="coerce")

    X = intensities
    if sparse:
        from scipy.sparse import csr_matrix

        X = csr_matrix(X)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.var_names_make_unique()

    # squidpy / scanpy spatial convention
    adata.obsm["spatial"] = coords.astype(float)

    # QC metrics
    dense = intensities
    adata.obs["total_intensity"] = dense.sum(axis=1)
    adata.obs["n_features"] = (dense > 0).sum(axis=1)

    # lightweight uns["spatial"] block (no histology image is shipped with the
    # MSI CSV; coordinates here are MSI grid units, not Visium pixels)
    adata.uns["spatial"] = {
        sample_id: {
            "metadata": {
                "source": "SMA MALDI-MSI (*_smamsi.csv)",
                "coordinate_units": "MSI grid (integer pixel index)",
                "matrix": matrix,
                "modality": modality,
            }
        }
    }
    adata.uns["sma_msi"] = {
        "sample_id": sample_id,
        "matrix": matrix,
        "modality": modality,
        "n_pixels": int(n_pixels),
        "n_features": int(len(mz_values)),
    }
    return adata


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="Path to a SMA *_smamsi.csv file")
    p.add_argument("-o", "--output", help="Output .h5ad path (default: alongside input)")
    p.add_argument("--sample-id", help="Sample id, e.g. V11L12-038_D1 (default: inferred)")
    p.add_argument("--matrix", help="MALDI matrix, e.g. 9-AA or DHB (default: inferred from filename)")
    p.add_argument("--modality", help="Data type, e.g. metabolites / lipids (default: inferred)")
    p.add_argument("--sparse", action="store_true", help="Store X as a sparse CSR matrix")
    return p.parse_args(argv)


def _infer_metadata(path, sample_id, matrix, modality):
    base = os.path.basename(path)
    name = base.lower()
    if matrix is None:
        if "9aa" in name or "9-aa" in name:
            matrix = "9-AA"
        elif "dhb" in name:
            matrix = "DHB"
        elif "fmp" in name:
            matrix = "FMP-10"
    if modality is None and matrix is not None:
        modality = {"9-AA": "metabolites", "DHB": "lipids", "FMP-10": "neurotransmitters"}.get(matrix)
    if sample_id is None:
        # filenames look like V11L12-038_A1.Visium.DHB... or V11L12-038_Mouse_D1.Visium.9aa...
        token = base.split(".")[0]
        token = token.replace("_Mouse", "").replace("_mouse", "")
        sample_id = token
    return sample_id, matrix, modality


def main(argv=None):
    args = parse_args(argv)

    if not os.path.exists(args.input):
        sys.exit(f"Input not found: {args.input}")

    sample_id, matrix, modality = _infer_metadata(
        args.input, args.sample_id, args.matrix, args.modality
    )

    coords, intensities, mz_values = load_msi_csv(args.input)
    adata = build_anndata(
        coords, intensities, mz_values, sample_id, matrix, modality, sparse=args.sparse
    )

    out = args.output
    if out is None:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        out = os.path.join(os.path.dirname(os.path.abspath(args.input)), f"{stem}.h5ad")

    adata.write_h5ad(out)
    print(f"[ok] {args.input}")
    print(f"     sample_id={sample_id}  matrix={matrix}  modality={modality}")
    print(f"     {adata.n_obs} pixels x {adata.n_vars} m/z features -> {out}")
    return adata


if __name__ == "__main__":
    main()