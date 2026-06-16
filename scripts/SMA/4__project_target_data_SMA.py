#%% Project Spatial RNA-ATAC target data into the SMA-trained RNA latent space.
#
# The SMA SpatialJEPA model (script 3) is a joint RNA("ST")+metabolomics("SM") VAE.
# Spatial RNA-ATAC shares only the RNA modality with SMA, so we push the target's RNA
# through the SMA-trained RNA encoder (encoder_ST) and read the RNA-only latent
# H["st"]["q_mu"]. The SM block is filled with zeros (it does not affect H["st"]["q_mu"]).
#
# ST inputs are RAW counts: the model was trained on raw HVG counts (script 3's final
# normalize_total uses target_sum_ST=None, which skips ST), and encode() applies
# log(X_ST + 1) internally. So the target ST block is raw counts reindexed to the SMA
# gene set (zero-filled for genes absent from the target).

from dotenv import load_dotenv
load_dotenv(dotenv_path="/home/mcb/users/dmannk/BAKLAVA_base/BAKLAVA/.env")

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spatialmeta as smt
from anndata import AnnData

sys.path.insert(0, os.path.join(os.getenv("BAKLAVA_ROOT"), "scripts", "SMA"))
from spatialjepa_model import load_spatialjepa_model

OUTPUT_DIR = Path(os.getenv("OUTPATH"))
SAMPLE_ID = "V11L12-038_D1"  # SMA sample whose trained encoders we project with
MODEL_DIR = OUTPUT_DIR / "spatialjepa_models" / SAMPLE_ID
TARGET_RNA_PATH = Path(
    "/home/mcb/users/dmannk/BAKLAVA_base/data/Spatial_ATAC_RNA/mouse/"
    "spatial_omics/spatial_atac_rna_seq_mouse_brain.h5ad"
)
PROJ_DIR = OUTPUT_DIR / "spatialjepa_projection" / SAMPLE_ID
PROJ_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RNA_PREFIX = "rna:"

#%% Load the trained feature space (template) and the target RNA.

template = sc.read_h5ad(MODEL_DIR / "joint_adata.h5ad")
st_mask = (template.var["type"].values == "ST")
sm_mask = (template.var["type"].values == "SM")
st_cols = np.where(st_mask)[0]
# SMA RNA gene symbols, in the exact column order encoder_ST expects.
sma_genes = pd.Index(
    [v[len(RNA_PREFIX):] if v.startswith(RNA_PREFIX) else v
     for v in template.var_names[st_mask]]
)
print(f"[INFO] Template: {template.shape} "
      f"({int(st_mask.sum())} ST + {int(sm_mask.sum())} SM features)")

target = sc.read_h5ad(TARGET_RNA_PATH)
target.obsm['spatial'] = np.array([1, -1]) * target.obsm['spatial']
target.var_names_make_unique()
target_counts = target.layers["counts"] if "counts" in target.layers else target.X
print(f"[INFO] Target RNA: {target.shape}")

#%% Align RNA by exact gene-symbol intersection, reindex into SMA order, zero-fill missing.
# (Same matching convention as DataAligner.find_gene_overlap; we keep all SMA genes and
#  zero-fill because the encoder has a fixed input dimension.)

idx = target.var_names.get_indexer(sma_genes)          # -1 where SMA gene absent in target
present = idx >= 0
n_overlap = int(present.sum())
print(f"[INFO] Gene overlap: {n_overlap}/{len(sma_genes)} SMA ST genes found in target "
      f"({100 * n_overlap / len(sma_genes):.1f}%).")
missing_examples = list(sma_genes[~present][:10])
print(f"[INFO] Example missing genes (zero-filled): {missing_examples}")

n_obs = target.n_obs
X_st = np.zeros((n_obs, len(sma_genes)), dtype=np.float32)
src = target_counts[:, idx[present]]
X_st[:, present] = src.toarray() if sp.issparse(src) else np.asarray(src)

#%% Assemble the target joint AnnData matching the template var space exactly.

X = np.zeros((n_obs, template.n_vars), dtype=np.float32)
X[:, st_cols] = X_st                                   # SM columns stay zero

target_joint = smt.util._classes.AnnDataJointSMST(
    AnnData(X=X, obs=target.obs.copy(), var=template.var.copy())
)
target_joint.obsm["spatial"] = np.asarray(target.obsm["spatial"])  # teacher graph needs this
target_joint.layers["counts"] = X.copy()
print(f"[INFO] Target joint adata: {target_joint.shape}; "
      f"batches: {target_joint.obs['batch'].value_counts().to_dict() if 'batch' in target_joint.obs else 'n/a'}")

#%% Load both SMA-trained models on the target template and project (RNA-only latent).

MODELS = {
    "student": MODEL_DIR / "student.pt",   # graph-free (identity self-loop graph)
    "teacher": MODEL_DIR / "teacher.pt",   # full 6-NN spatial graph, rebuilt on target spots
}

X_full = torch.as_tensor(np.asarray(target_joint.X), dtype=torch.float32, device=DEVICE)
for name, ckpt_path in MODELS.items():
    model = load_spatialjepa_model(ckpt_path, target_joint, device=DEVICE)
    with torch.no_grad():
        H = model.encode(X_full)            # full batch -> GCN uses full target graph directly
    Z = H["st"]["q_mu"].detach().cpu().numpy()
    assert not np.isnan(Z).any(), f"{name}: NaNs in RNA-only embedding"
    target_joint.obsm[f"{name}_X_emb_st"] = Z
    print(f"[INFO] {name}: RNA-only embedding {Z.shape}")
    del model
    torch.cuda.empty_cache()

#%% Downstream UMAP / Leiden per model (sanity check that the SMA RNA encoder transfers).

def embed_and_plot(name: str) -> None:
    emb_key = f"{name}_X_emb_st"
    neigh_key = f"{name}_neighbors"
    umap_key = f"{name}_umap"
    cluster_key = f"{name}_leiden"

    sc.pp.neighbors(target_joint, use_rep=emb_key, n_neighbors=15, key_added=neigh_key)
    sc.tl.umap(target_joint, min_dist=0.3, neighbors_key=neigh_key)
    target_joint.obsm[umap_key] = target_joint.obsm["X_umap"].copy()
    sc.tl.leiden(target_joint, key_added=cluster_key, neighbors_key=neigh_key)

    color = [cluster_key]
    for c in ("RNA_clusters", "ATAC_clusters", "batch"):
        if c in target_joint.obs:
            color.append(c)
    fig = sc.pl.embedding(
        target_joint, basis=umap_key, color=color, ncols=2, wspace=0.3,
        show=False, return_fig=True,
    )
    fig.savefig(PROJ_DIR / f"{name}_umap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] {name}: wrote {PROJ_DIR / f'{name}_umap.png'}")

    fig_spatial = sc.pl.embedding(
        target_joint,
        basis="spatial",
        color=color[:-1],
        show=False,
        return_fig=True,
        s=60,
    )
    spatial_path = PROJ_DIR / f"{name}_spatial_{c}.png"
    fig_spatial.savefig(spatial_path, dpi=150, bbox_inches="tight")
    plt.close(fig_spatial)
    print(f"[INFO] {name}: wrote {spatial_path}")

for name in MODELS:
    embed_and_plot(name)

#%% Student vs teacher agreement (distillation-transfer sanity check).

Zs, Zt = target_joint.obsm["student_X_emb_st"], target_joint.obsm["teacher_X_emb_st"]
per_dim_r = [float(np.corrcoef(Zs[:, d], Zt[:, d])[0, 1]) for d in range(Zs.shape[1])]
print("[INFO] Student-vs-teacher per-latent-dim Pearson r:")
print("       " + ", ".join(f"{r:.3f}" for r in per_dim_r))
print(f"[INFO] Mean |r| across dims: {np.nanmean(np.abs(per_dim_r)):.3f}")

#%% Save the embedded target.

out_path = PROJ_DIR / "target_spatial_atac_rna_embedded.h5ad"
target_joint.write_h5ad(out_path)
print(f"[INFO] Saved embedded target to {out_path}")
