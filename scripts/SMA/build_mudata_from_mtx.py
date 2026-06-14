"""Build .h5mu files from Matrix Market exports of se.multi.list."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import anndata as ad
import mudata as mu
import numpy as np
import pandas as pd
from scipy.io import mmread


EXPORT_DIR = Path("/Users/dmannk/BAKLAVA_base/outputs/SMA/h5mu_export")
BAKLAVA_BASE = Path(__file__).resolve().parents[3]
DEFAULT_VISIUM_DATA_ROOT = BAKLAVA_BASE / "data" / "vicari_2023" / "data"


def read_vector(path: Path) -> pd.Series:
    return pd.read_csv(path, header=None, dtype=str)[0].astype(str)


def read_features(path: Path, modality: str) -> pd.DataFrame:
    feature_ids = read_vector(path)
    var = pd.DataFrame(index=pd.Index(feature_ids, name="feature_id"))
    var["feature_id"] = var.index.astype(str)

    if modality == "rna":
        var["gene_symbol"] = var["feature_id"]
    elif modality == "msi":
        var["mz"] = pd.to_numeric(var["feature_id"], errors="coerce")

    return var


def read_obs(sample_dir: Path, barcodes: pd.Series) -> pd.DataFrame:
    sample_id = sample_dir.name
    obs = pd.read_csv(sample_dir / "obs.tsv", sep="\t")
    if "barcode" not in obs.columns:
        raise ValueError(f"obs.tsv in {sample_dir} must contain a barcode column")

    obs["barcode"] = obs["barcode"].astype(str)
    if "msi_barcode" in obs.columns:
        obs["msi_barcode"] = obs["msi_barcode"].astype(str)

    missing = barcodes.loc[~barcodes.isin(obs["barcode"])]
    if len(missing) > 0:
        preview = ", ".join(missing.head(5))
        raise ValueError(f"{sample_dir} is missing {len(missing)} barcodes in obs.tsv: {preview}")

    obs = obs.set_index("barcode").loc[barcodes].copy()
    obs.insert(0, "barcode", obs.index.astype(str))

    if "sample_id" not in obs.columns:
        obs["sample_id"] = sample_id
    obs["sample_id"] = obs["sample_id"].astype(str)

    obs.index = pd.Index([f"{sample_id}:{barcode}" for barcode in obs["barcode"]], name="obs_id")
    return obs


def build_anndata(matrix_path: Path, obs: pd.DataFrame, var: pd.DataFrame) -> ad.AnnData:
    counts = mmread(matrix_path).tocsr()
    expected_shape = (obs.shape[0], var.shape[0])
    if counts.shape != expected_shape:
        raise ValueError(
            f"{matrix_path} has shape {counts.shape}, expected {expected_shape} "
            "from obs/features tables"
        )

    adata = ad.AnnData(X=counts, obs=obs.copy(), var=var)
    adata.layers["counts"] = adata.X.copy()
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()
    return adata


def numeric_obsm(obs: pd.DataFrame, columns: list[str]) -> np.ndarray | None:
    if not set(columns).issubset(obs.columns):
        return None

    values = obs.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy()
    if np.isnan(values).any():
        return None
    return values


def visium_data_root(data_root: Path | None = None) -> Path:
    if data_root is not None:
        return data_root
    datapath = os.environ.get("DATAPATH")
    if datapath:
        return Path(datapath) / "vicari_2023" / "data"
    return DEFAULT_VISIUM_DATA_ROOT


def visium_outs(sample_id: str, data_root: Path | None = None) -> Path:
    """Space Ranger outs directory for an SMA sample (e.g. V11L12-038_A1)."""
    root = visium_data_root(data_root)
    array_id = sample_id.rsplit("_", 1)[0]
    return root / "sma" / array_id / sample_id / "output_data" / f"{sample_id}_RNA" / "outs"


def attach_visium_uns(
    *adatas: ad.AnnData,
    sample_id: str,
    data_root: Path | None = None,
) -> bool:
    """Attach native Visium uns['spatial'] (H&E images + scalefactors) from sc.read_visium.

    obsm['spatial'] on the exported objects remains the aligned/warped coordinates from
    obs.tsv; this only supplies scanpy metadata needed by sc.pl.spatial (spot_size, images).
    """
    outs = visium_outs(sample_id, data_root)
    if not outs.is_dir():
        warnings.warn(f"No Visium outs at {outs}; skipping uns['spatial']", stacklevel=2)
        return False

    try:
        import scanpy as sc
    except ImportError:
        warnings.warn("scanpy not installed; skipping uns['spatial']", stacklevel=2)
        return False

    ref = sc.read_visium(outs)
    ref.var_names_make_unique()
    spatial_uns = ref.uns.get("spatial")
    if not spatial_uns:
        warnings.warn(f"read_visium({outs}) returned no uns['spatial']", stacklevel=2)
        return False

    for adata in adatas:
        adata.uns["spatial"] = spatial_uns
    return True


def add_spatial_metadata(rna: ad.AnnData, msi: ad.AnnData, obs: pd.DataFrame) -> None:
    spatial = numeric_obsm(obs, ["rna_warped_x", "rna_warped_y"])
    if spatial is None:
        spatial = numeric_obsm(obs, ["pixel_x", "pixel_y"])

    if spatial is not None:
        rna.obsm["spatial"] = spatial
        msi.obsm["spatial"] = spatial

    rna_source_spatial = numeric_obsm(obs, ["rna_warped_x", "rna_warped_y"])
    if rna_source_spatial is not None:
        rna.obsm["source_spatial"] = rna_source_spatial

    msi_source_spatial = numeric_obsm(obs, ["msi_warped_x", "msi_warped_y"])
    if msi_source_spatial is not None:
        msi.obsm["source_spatial"] = msi_source_spatial


def load_sample_from_mtx(sample_dir: Path) -> mu.MuData:
    barcodes = read_vector(sample_dir / "barcodes.tsv")
    obs = read_obs(sample_dir, barcodes)

    rna = build_anndata(sample_dir / "rna.mtx", obs, read_features(sample_dir / "rna_features.tsv", "rna"))
    msi = build_anndata(sample_dir / "msi.mtx", obs, read_features(sample_dir / "msi_features.tsv", "msi"))
    add_spatial_metadata(rna, msi, obs)
    attach_visium_uns(rna, msi, sample_id=sample_dir.name)

    mdata = mu.MuData({"rna": rna, "msi": msi})
    mdata.update()
    return mdata


def build_all(export_dir: Path = EXPORT_DIR) -> dict[str, mu.MuData]:
    samples = sorted(p.name for p in export_dir.iterdir() if p.is_dir())
    out: dict[str, mu.MuData] = {}
    for sample_id in samples:
        mdata = load_sample_from_mtx(export_dir / sample_id)
        h5mu_path = export_dir / f"{sample_id}.h5mu"
        mdata.write(h5mu_path)
        out[sample_id] = mdata
        print(
            f"{sample_id}: wrote {h5mu_path} "
            f"({mdata.mod['rna'].n_obs} spots, "
            f"{mdata.mod['rna'].n_vars} genes, "
            f"{mdata.mod['msi'].n_vars} MSI features)"
        )
    return out


if __name__ == "__main__":
    build_all()
