#!/usr/bin/env python
#%%
"""Train a TOTALVI model on a single SMA spatial RNA + metabolomics (MSI) sample.

TOTALVI is architecturally RNA + protein. Here we treat the SMA metabolomics
("SM"/MSI) block as the "protein" modality so its continuous intensities are
modelled with TOTALVI's protein likelihood (a better reconstruction distribution
than MultiVI's binarized Bernoulli accessibility block). The MSI intensities are
therefore passed through *without discretization*.

Data source: the already-processed ``joint_adata.h5ad`` written by
``3__spatial_meta_modelling.py`` at
``${OUTPATH}/spatialjepa_models/<sample_id>/joint_adata.h5ad``. That object has
Moran's-I spatially-variable feature selection, normalization layers, var typing
(``var["type"]`` in {"ST", "SM"}), and obs annotations (region/lesion/section/
RNA_clusters/MSI_clusters) already applied. We split it into RNA + MSI modalities,
train TOTALVI, embed, and save.

Env: ``conda activate scvi_env`` (scvi-tools 1.3.3).
"""
import argparse
import os
from pprint import pprint

import scvi
from dotenv import dotenv_values, load_dotenv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import muon as mu
import numpy as np
import scanpy as sc
import scipy.sparse as sp

import warnings
warnings.filterwarnings("ignore")

ENV_FILE_PATH = "/home/mcb/users/dmannk/BAKLAVA_base/BAKLAVA/.env"
DEFAULT_SAMPLE_ID = "V11L12-109_B1"

RNA_PREFIX = "rna:"
MSI_PREFIX = "msi:"

TOTALVI_LATENT_KEY = "X_totalvi"

OUTPATH = None

#%%
def bootstrap_runtime():
    global OUTPATH

    load_dotenv(dotenv_path=ENV_FILE_PATH)
    print("Loaded environment variables from .env or env:", end="\n\n")
    pprint(dotenv_values(ENV_FILE_PATH))

    OUTPATH = os.getenv("OUTPATH")
    if OUTPATH is None:
        raise EnvironmentError(
            "OUTPATH is not set. Export OUTPATH to the base outputs directory, e.g. "
            "'/home/mcb/users/dmannk/BAKLAVA_base/outputs'."
        )


def is_notebook():
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        if shell == "TerminalInteractiveShell":
            return False
        return False
    except Exception:
        return False


def parse_args(notebook: bool = False):
    parser = argparse.ArgumentParser(
        description="Train TOTALVI on one SMA sample (RNA expression + MSI as protein)."
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=DEFAULT_SAMPLE_ID,
        help="SMA sample id whose joint_adata.h5ad (from script 3) to train on.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=400,
        help="Number of epochs to train TOTALVI.",
    )
    parser.add_argument(
        "--n-latent",
        type=int,
        default=20,
        help="Dimensionality of the TOTALVI joint latent space.",
    )
    parser.add_argument(
        "--gene-likelihood",
        type=str,
        choices=["nb", "zinb"],
        default="nb",
        help="RNA expression likelihood for TOTALVI.",
    )
    parser.add_argument(
        "--protein-layer",
        type=str,
        choices=["normalized", "counts"],
        default="normalized",
        help=(
            "joint_adata layer whose SM (MSI) block feeds the protein modality. "
            "'normalized' = TIC-normalized intensities (NB-friendly scale); "
            "'counts' = raw MSI intensities."
        ),
    )
    parser.add_argument(
        "--batch-key",
        type=str,
        default=None,
        help="Optional obs key to integrate over as TOTALVI batch (default None; single sample).",
    )
    if notebook:
        return parser.parse_known_args()[0]
    else:
        return parser.parse_args()


def validate_args(args):
    if args.max_epochs <= 0:
        raise ValueError("--max-epochs must be a positive integer.")
    if args.n_latent <= 0:
        raise ValueError("--n-latent must be a positive integer.")


def _to_csr(mat):
    return mat.tocsr() if sp.issparse(mat) else sp.csr_matrix(np.asarray(mat))


def _strip_prefix(var_names, prefix):
    return [v[len(prefix):] if v.startswith(prefix) else v for v in var_names]


def load_sma_joint_adata(sample_id, protein_layer):
    """Load the processed joint_adata and split it into RNA + MSI (protein) modalities."""
    joint_path = os.path.join(OUTPATH, "spatialjepa_models", sample_id, "joint_adata.h5ad")
    if not os.path.exists(joint_path):
        raise FileNotFoundError(
            "joint_adata.h5ad not found: {}.\nRun 3__spatial_meta_modelling.py for "
            "sample_id='{}' first.".format(joint_path, sample_id)
        )

    joint = sc.read_h5ad(joint_path)
    if "type" not in joint.var.columns:
        raise ValueError("joint_adata.var is missing the 'type' column (expected 'ST'/'SM').")
    if protein_layer not in joint.layers:
        raise ValueError(
            "Requested protein layer '{}' not in joint_adata.layers ({}).".format(
                protein_layer, list(joint.layers.keys())
            )
        )

    st_mask = (joint.var["type"].values == "ST")
    sm_mask = (joint.var["type"].values == "SM")
    print(
        "[INFO] joint_adata {}: {} ST (RNA) + {} SM (MSI) features".format(
            joint.shape, int(st_mask.sum()), int(sm_mask.sum())
        )
    )

    # RNA expression block: raw integer counts (TOTALVI expects counts).
    rna = joint[:, st_mask].copy()
    rna.var_names = _strip_prefix(rna.var_names, RNA_PREFIX)
    rna.layers["counts"] = _to_csr(joint.layers["counts"][:, st_mask])
    rna.X = rna.layers["counts"].copy()

    # Protein block: continuous MSI intensities (NOT discretized).
    protein = joint[:, sm_mask].copy()
    protein.var_names = _strip_prefix(protein.var_names, MSI_PREFIX)
    protein.layers["counts"] = _to_csr(joint.layers[protein_layer][:, sm_mask])
    protein.X = protein.layers["counts"].copy()

    # Carry spatial coords onto both modalities (obs is shared by the [:, mask] subset).
    if "spatial" in joint.obsm:
        rna.obsm["spatial"] = np.asarray(joint.obsm["spatial"])
        protein.obsm["spatial"] = np.asarray(joint.obsm["spatial"])

    return rna, protein, joint


def compute_umap_and_leiden(mdata, color_keys, out_dir, sample_id, point_size=60):
    """Joint-latent UMAP + Leiden, plotted on UMAP and (if present) spatial coords."""
    sc.pp.neighbors(mdata, use_rep=TOTALVI_LATENT_KEY)
    sc.tl.umap(mdata, min_dist=0.2)
    sc.tl.leiden(mdata, resolution=0.25)

    present = [k for k in color_keys if k in mdata.obs.columns] + ["leiden"]

    fig = sc.pl.umap(mdata, color=present, show=False, return_fig=True)
    fig.savefig(os.path.join(out_dir, f"{sample_id}_totalvi_umap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # spatial coords live per-modality; lift onto the top-level MuData for plotting.
    if "spatial" in mdata.mod["rna"].obsm:
        mdata.obsm["spatial"] = mdata.mod["rna"].obsm["spatial"]
        fig = sc.pl.embedding(mdata, color=present, basis="spatial", s=point_size, show=False, return_fig=True)
        fig.savefig(os.path.join(out_dir, f"{sample_id}_totalvi_spatial.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


#%%
def main():
    # setup runtime
    bootstrap_runtime()
    notebook_mode = is_notebook()
    args = parse_args(notebook=notebook_mode)
    validate_args(args)

    # load and split the processed SMA joint adata
    rna, msi, joint = load_sma_joint_adata(args.sample_id, args.protein_layer)
    mdata = mu.MuData({"rna": rna, "msi": msi})
    mdata.update()

    # setup mudata: RNA -> expression, MSI -> protein. We deliberately pass continuous
    # (non-count) MSI intensities to the protein block per the modelling choice; if
    # setup/training rejects non-count protein data, surface it rather than discretizing.
    scvi.model.TOTALVI.setup_mudata(
        mdata,
        rna_layer="counts",
        protein_layer="counts",
        batch_key=args.batch_key,
        modalities={
            "rna_layer": "rna",
            "protein_layer": "msi",
        },
    )

    # setup and train model.
    # empirical_protein_background_prior=False: TOTALVI's empirical background prior fits a
    # GaussianMixture under warnings-as-errors and asserts the protein block is unnormalized
    # counts. Our MSI block is continuous (intensities, deliberately not discretized), which
    # would raise there. Disabling it keeps the continuous protein input and lets the model
    # learn the background prior instead of seeding it from a count-based GMM.
    model = scvi.model.TOTALVI(
        mdata,
        n_latent=args.n_latent,
        gene_likelihood=args.gene_likelihood,
        empirical_protein_background_prior=False,
    )
    model.view_anndata_setup()
    model.train(max_epochs=args.max_epochs)

    # save model
    model_dir = os.path.join(OUTPATH, "totalvi_mouse_sma", args.sample_id)
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir, overwrite=True)
    print(f"Model saved to {model_dir}")

    #model = scvi.model.TOTALVI.load(model_dir, adata=mdata)

    # joint latent embedding
    Z = model.get_latent_representation()
    assert not np.isnan(Z).any(), "NaNs in TOTALVI latent representation"
    mdata.obsm[TOTALVI_LATENT_KEY] = Z
    print(f"[INFO] TOTALVI latent: {Z.shape}")

    # UMAP / Leiden / spatial visualization
    compute_umap_and_leiden(
        mdata,
        color_keys=["RNA_clusters", "MSI_clusters", "region", "lesion"],
        out_dir=model_dir,
        sample_id=args.sample_id,
    )

    # save umap plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sc.pl.umap(mdata.mod["rna"], color=["RNA_clusters"], ax=ax[0], show=False)
    sc.pl.umap(mdata.mod["msi"], color=["MSI_clusters"], ax=ax[1], show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "totalvi_umap_rna_msi_clusters.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # save the embedded mudata
    mudata_path = os.path.join(model_dir, "totalvi_mdata.h5mu")
    mdata.write_h5mu(mudata_path)
    print(f"Mudata saved to {mudata_path}")


if __name__ == "__main__":
    main()

# %%
