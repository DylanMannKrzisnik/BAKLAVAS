"""Shared, import-safe loaders and reformatting helpers for SMA fusion.

These functions are factored out of ``1__sma_multimodal_modelling.py`` so that the
downstream integration scripts (MultiVI in ``scvi_env``, scGLUE in
``nichecompass_liana``) can reuse the exact same fused-MuData construction without
executing that script's MISTy/bivariate pipeline on import.

Depends only on anndata / mudata / scanpy / numpy / pandas, so it imports cleanly in
both conda environments. ``DATAPATH`` is read lazily (from the environment, populated
by the caller's ``dotenv``) only when an export dir is not supplied.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import anndata as ad
import mudata as mu
import numpy as np
import pandas as pd
import scanpy as sc

from load_aligned_mudata import load_sample


def _default_export_dir() -> Path:
    return Path(os.environ["DATAPATH"]) / "vicari_2023" / "h5mu_export"


# --------------------------------------------------------------------------- #
# Fused-MuData construction (moved verbatim from 1__sma_multimodal_modelling)  #
# --------------------------------------------------------------------------- #
def raw_barcode(obs_name: str) -> str:
    barcode = str(obs_name).split(":", 1)[-1]
    return re.sub(r"(-\d+)[_.]\d+$", r"\1", barcode)


def prep_fusion_modality(adata: ad.AnnData, source: str, sample_id: str) -> ad.AnnData:
    out = adata.copy()
    out.obs["source"] = source
    out.obs["source_sample_id"] = sample_id
    out.obs["raw_barcode"] = [raw_barcode(obs_name) for obs_name in out.obs_names]

    # Keep FMP-10 and DHB rows distinct after concatenation.
    out.obs_names = [f"{source}:{barcode}" for barcode in out.obs["raw_barcode"]]
    return out


def _combine_var_tables(inputs: dict[str, ad.AnnData], var_names: pd.Index) -> pd.DataFrame:
    """Restore var annotations after AnnData concatenation.

    anndata.concat drops var columns unless told how to merge them. Here we keep
    all columns and add the source(s) each feature was observed in.
    """
    frames = []
    for source, adata in inputs.items():
        var = adata.var.copy()
        var["feature_sources"] = source
        frames.append(var)

    var = pd.concat(frames, axis=0, sort=False)
    if var.index.has_duplicates:
        feature_sources = var.groupby(level=0)["feature_sources"].agg(
            lambda values: ";".join(sorted(pd.unique(values.astype(str))))
        )
        var = var.drop(columns="feature_sources").groupby(level=0).first()
        var["feature_sources"] = feature_sources

    return var.reindex(var_names)


def concat_modalities_keep_obs_var(inputs: dict[str, ad.AnnData]) -> ad.AnnData:
    out = ad.concat(
        inputs,
        label="source_batch",
        index_unique=None,
        join="outer",
        merge="same",
        uns_merge="same",
    )
    out.var = _combine_var_tables(inputs, out.var_names)
    return out


def load_fmp10_partner_fused_mudata(
    fmp10_sample_id: str = "V11L12-109_B1",
    partner_sample_id: str = "V11L12-038_B1",
    partner_source: str = "dhb",
    export_dir: Path | None = None,
    write: bool = False,
) -> mu.MuData:
    export_dir = export_dir or _default_export_dir()
    fmp10 = load_sample(fmp10_sample_id, export_dir=export_dir)
    partner = load_sample(partner_sample_id, export_dir=export_dir)

    rna_inputs = {
        "fmp10": prep_fusion_modality(fmp10.mod["rna"], "fmp10", fmp10_sample_id),
        partner_source: prep_fusion_modality(
            partner.mod["rna"], partner_source, partner_sample_id
        ),
    }
    msi_inputs = {
        "fmp10": prep_fusion_modality(fmp10.mod["msi"], "fmp10", fmp10_sample_id),
        partner_source: prep_fusion_modality(
            partner.mod["msi"], partner_source, partner_sample_id
        ),
    }

    mdata = mu.MuData(
        {
            "rna": concat_modalities_keep_obs_var(rna_inputs),
            "msi": concat_modalities_keep_obs_var(msi_inputs),
        }
    )
    mdata.update()

    if write:
        out_path = export_dir / (
            f"{fmp10_sample_id}__{partner_sample_id}.fmp10_{partner_source}_concat.h5mu"
        )
        mdata.write(out_path)

    return mdata


def load_fmp10_dhb_fused_mudata(
    fmp10_sample_id: str = "V11L12-109_B1",
    dhb_sample_id: str = "V11L12-038_B1",
    export_dir: Path | None = None,
    write: bool = False,
) -> mu.MuData:
    return load_fmp10_partner_fused_mudata(
        fmp10_sample_id=fmp10_sample_id,
        partner_sample_id=dhb_sample_id,
        partner_source="dhb",
        export_dir=export_dir,
        write=write,
    )


def load_fmp10_nineaa_fused_mudata(
    fmp10_sample_id: str = "V11L12-109_B1",
    nineaa_sample_id: str = "V11L12-038_D1",
    export_dir: Path | None = None,
    write: bool = False,
) -> mu.MuData:
    return load_fmp10_partner_fused_mudata(
        fmp10_sample_id=fmp10_sample_id,
        partner_sample_id=nineaa_sample_id,
        partner_source="nineaa",
        export_dir=export_dir,
        write=write,
    )


# --------------------------------------------------------------------------- #
# Reformatting helpers for vertical / diagonal integration                    #
# --------------------------------------------------------------------------- #
def split_msi_by_source(fused_mdata: mu.MuData, mod: str = "msi") -> dict[str, ad.AnnData]:
    """Split the fused, outer-joined MSI matrix back into per-source AnnDatas.

    The fused matrix is block-diagonal: each source's spots are zero in the other
    source's columns. We slice rows by ``source_batch``, drop the all-zero (foreign)
    columns, and re-index obs by ``raw_barcode`` so the two sources can be matched on
    the shared Visium array position (the only available pairing key).
    """
    msi = fused_mdata[mod]
    out: dict[str, ad.AnnData] = {}
    for source in msi.obs["source_batch"].unique():
        sub = msi[msi.obs["source_batch"] == source].copy()
        X = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
        keep = np.asarray(np.abs(X).sum(axis=0)).ravel() > 0
        sub = sub[:, keep].copy()
        sub.obs_names = sub.obs["raw_barcode"].astype(str).values
        out[str(source)] = sub
    return out


def tic_pseudocounts(
    adata: ad.AnnData, target_sum: float = 1e4, layer_from: str | None = None
) -> ad.AnnData:
    """Add count-like and log-normalised layers for continuous MSI intensities.

    ``pseudocounts``: TIC-normalised + rounded to integers (for Poisson/NB likelihoods).
    ``lognorm``     : log1p of the TIC-normalised intensities (for ZILN / continuous).
    The input ``.X`` (raw intensity) is left untouched.
    """
    out = adata.copy()
    base = out.layers[layer_from].copy() if layer_from else out.X.copy()
    norm = ad.AnnData(X=base, obs=out.obs.copy(), var=out.var.copy())
    sc.pp.normalize_total(norm, target_sum=target_sum)
    norm_X = norm.X.toarray() if hasattr(norm.X, "toarray") else np.asarray(norm.X)
    out.layers["pseudocounts"] = np.rint(norm_X).astype(np.float32)
    out.layers["lognorm"] = np.log1p(norm_X).astype(np.float32)
    return out


def build_pseudopaired_multiome(
    a: ad.AnnData, b: ad.AnnData
) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData]:
    """Join two per-source AnnDatas on shared ``raw_barcode`` (array position).

    Returns ``(multi, a_only, b_only)`` ready for
    ``scvi.data.organize_multiome_anndatas``: ``multi`` are pseudo-paired spots
    carrying both feature blocks (a's vars then b's vars), ``a_only``/``b_only`` are the
    spots present in only one section. Pairing is by Visium array position across two
    *different* tissue sections (a pseudo-pairing), not the same physical cell.
    """
    shared = a.obs_names.intersection(b.obs_names)
    a_shared = a[shared].copy()
    b_shared = b[shared].copy()
    # Concatenate feature blocks side-by-side (vars), aligned on the shared obs.
    multi = ad.concat([a_shared, b_shared], axis=1, merge="first")
    multi.obs = a_shared.obs.copy()

    a_only = a[a.obs_names.difference(shared)].copy()
    b_only = b[b.obs_names.difference(shared)].copy()
    return multi, a_only, b_only
