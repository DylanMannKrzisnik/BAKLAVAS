# %% [markdown]
# # SMA metabolite vertical integration with MultiVI
#
# Integrates the two disjoint MALDI matrices (FMP-10 vs DHB) at the **feature** level by
# treating them as different modalities in a MultiVI model, rather than as a batch effect.
# Because the two matrices share zero features and zero spots (different sections), the
# only way to tie them together is either (a) a pseudo-pairing on the shared Visium array
# position, or (b) the common RNA modality as an anchor. Both variants are implemented.
#
# **Environment:** `conda activate scvi_env` (scvi-tools 1.3.3)
#
# **Caveat:** MultiVI is architecturally RNA+ATAC. It models only the *expression* block
# quantitatively (Poisson here, on TIC pseudo-counts); the second block is binarized and
# modelled with Bernoulli. For fully-quantitative modelling of every metabolite modality
# use the scGLUE route (ZILN) in `1__sma_multimodal_modelling.py`.

# %%
import os
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from dotenv import load_dotenv

load_dotenv(dotenv_path="/home/mcb/users/dmannk/BAKLAVA_base/BAKLAVA/.env")

from scvi.data import organize_multiome_anndatas
from scvi.model import MULTIVI

import sma_fusion as sf

DATAPATH = Path(os.environ["DATAPATH"])
OUT_DIR = DATAPATH / "vicari_2023" / "integration_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_EPOCHS = int(os.environ.get("SMA_MAX_EPOCHS", "200"))


def _counts_layer(adata: ad.AnnData) -> np.ndarray:
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return np.rint(X).astype(np.float32)


def _binarize(adata: ad.AnnData, layer: str | None = None) -> np.ndarray:
    X = adata.layers[layer] if layer else adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return (X > 0).astype(np.float32)


def _drop_empty(adata: ad.AnnData) -> ad.AnnData:
    """Drop all-zero features (TIC-rounding can zero out low-intensity m/z) and empty spots,
    which otherwise make per-feature factors / library sizes diverge to NaN during training."""
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.filter_cells(adata, min_counts=1)
    return adata


def train_multivi(mvi: ad.AnnData, n_genes: int, n_regions: int, batch_key: str) -> MULTIVI:
    mvi.layers["counts"] = mvi.X.copy()
    MULTIVI.setup_anndata(mvi, batch_key=batch_key, layer="counts")
    model = MULTIVI(mvi, n_genes=n_genes, n_regions=n_regions, gene_likelihood="poisson")
    model.train(max_epochs=MAX_EPOCHS)
    return model


def evaluate(mvi: ad.AnnData, model: MULTIVI, tag: str, color: str) -> ad.AnnData:
    mvi.obsm["X_multivi"] = model.get_latent_representation()
    sc.pp.neighbors(mvi, use_rep="X_multivi")
    sc.tl.umap(mvi)
    colors = [c for c in (color, "modality") if c in mvi.obs and mvi.obs[c].nunique() > 1]
    if colors:
        sc.pl.umap(mvi, color=colors, show=False, save=None)
    # Cross-modal feature imputation: estimate each block for every cell, including the
    # cells where that block was unobserved -> imputes the missing matrix's features.
    mvi.obsm["imputed_expression"] = np.asarray(model.get_normalized_expression())
    mvi.obsm["imputed_accessibility"] = np.asarray(model.get_normalized_accessibility())
    out_path = OUT_DIR / f"multivi_{tag}.h5ad"
    mvi.write(out_path)
    print(f"[{tag}] latent {mvi.obsm['X_multivi'].shape} written to {out_path}")
    return mvi


# %% Load fused data once
fused_mdata = sf.load_fmp10_dhb_fused_mudata()
msi_parts = sf.split_msi_by_source(fused_mdata)
fused_mdata


# %% [markdown]
# ## Variant A — metabolite-only (pseudo-paired on Visium array position)
# expression block = FMP-10 metab (Poisson pseudo-counts); accessibility block = DHB metab
# (binarized). Spots are pseudo-paired across sections by shared array barcode (~1860).
#
# **Paired-only:** including the single-modality (section-unique) spots made MultiVI diverge
# to NaN under the metabolite->Poisson mapping (their missing block has zero library). We
# restrict to the pseudo-paired spots, which still learn the joint latent and support
# cross-modal imputation (impute DHB metab from FMP-10 metab and vice versa).

# %%
def build_variant_a() -> tuple[ad.AnnData, int, int]:
    a = sf.tic_pseudocounts(msi_parts["fmp10"])
    a.X = a.layers["pseudocounts"].copy()                 # Poisson expression block
    b = sf.tic_pseudocounts(msi_parts["dhb"])
    b.X = (b.layers["pseudocounts"] > 0).astype(np.float32)  # Bernoulli accessibility block
    _drop_empty(a)
    _drop_empty(b)
    n_genes, n_regions = a.n_vars, b.n_vars
    multi, _a_only, _b_only = sf.build_pseudopaired_multiome(a, b)
    mvi = organize_multiome_anndatas(multi)               # paired-only (see note above)
    return mvi, n_genes, n_regions


mvi_a, n_genes_a, n_regions_a = build_variant_a()
model_a = train_multivi(mvi_a, n_genes_a, n_regions_a, batch_key="modality")
mvi_a = evaluate(mvi_a, model_a, tag="metabolite_only", color="modality")


# %% [markdown]
# ## Variant B — RNA-anchored
# expression block = RNA (NB-like, on counts) shared across sections; accessibility block =
# the union metabolite matrix (binarized, block-diagonal by section). Every spot carries
# both blocks (true multiome), and `source_batch` is the batch the model integrates over.

# %%
def build_variant_b() -> tuple[ad.AnnData, int, int]:
    rna = fused_mdata["rna"].copy()
    # Anchor on genes observed in BOTH sections (feature_sources lists both -> contains ';').
    shared_genes = rna.var["feature_sources"].astype(str).str.contains(";")
    rna = rna[:, shared_genes].copy()
    rna.X = _counts_layer(rna)                            # Poisson/NB expression block

    msi = fused_mdata["msi"].copy()
    msi.X = _binarize(msi)                                # Bernoulli accessibility block

    rna.var_names = [f"gene:{g}" for g in rna.var_names]  # keep blocks var-name disjoint
    mvi = ad.concat([rna, msi], axis=1, merge="first")
    mvi.obs = rna.obs.copy()
    mvi.obs["modality"] = "paired"
    return mvi, rna.n_vars, msi.n_vars


mvi_b, n_genes_b, n_regions_b = build_variant_b()
model_b = train_multivi(mvi_b, n_genes_b, n_regions_b, batch_key="source_batch")
mvi_b = evaluate(mvi_b, model_b, tag="rna_anchored", color="source_batch")
