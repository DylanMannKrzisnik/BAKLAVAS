#%% Project Spatial RNA-ATAC target data into the SMA-trained RNA latent/MSI space.
#
# The SMA SpatialJEPA model (script 3) is a joint RNA("ST")+metabolomics("SM") VAE.
# Spatial RNA-ATAC shares only the RNA modality with SMA, so we push the target RNA
# through the trained student and teacher. The student joint latent H["q_mu"] is
# used for the post-hoc PLS MSI decoder; the teacher RNA-only latent H["st"]["q_mu"]
# is used for the trained neural SM decoder. The SM block is filled with zeros in
# the model input.
#
# ST inputs are RAW counts: the model was trained on raw HVG counts (script 3's final
# normalize_total uses target_sum_ST=None, which skips ST), and encode() applies
# log(X_ST + 1) internally. So the target ST block is raw counts reindexed to the SMA
# gene set (zero-filled for genes absent from the target).

from dotenv import load_dotenv
load_dotenv(dotenv_path="/home/mcb/users/dmannk/BAKLAVA_base/BAKLAVA/.env")

import os
import sys
import json
from pathlib import Path

import joblib
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

SAMPLE_ID = "V11L12-109_B1"  # SMA sample/run whose trained student we project with
DECODER_MODE = "auto"         # "auto" -> PLS if present, otherwise neural fallback
BATCH_SIZE = 1024
MIN_ST_OVERLAP_FRACTION = 0.10
MSI_OBSM_KEY = "student_msi"
TEACHER_MSI_OBSM_KEY = "teacher_msi"
DOPAMINE_FEATURE = "msi:Dopamine"

SAMPLE_IDS = [SAMPLE_ID] if isinstance(SAMPLE_ID, str) else list(SAMPLE_ID)
PREFIX = SAMPLE_IDS[0].split("_")[0]

species = "human" if PREFIX == "V11T17-102" else "mouse"
target_rna_path_env = os.getenv("TARGET_RNA_PATH")
if target_rna_path_env:
    TARGET_RNA_PATH = Path(target_rna_path_env)
elif species == "mouse":
    TARGET_RNA_PATH = Path(os.getenv("DATAPATH")) / "aligned_data" / "source_rna_aligned_SCT.h5ad"
else:
    raise ValueError("Set TARGET_RNA_PATH for human target RNA projection.")

OUTPUT_DIR = Path(os.getenv("OUTPATH"))
MODEL_DIR = OUTPUT_DIR / "spatialjepa_models" / SAMPLE_ID
PROJ_DIR = OUTPUT_DIR / "spatialjepa_projection" / SAMPLE_ID
PROJ_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RNA_PREFIX = "rna:"


def bare_feature_symbol(name) -> str:
    """Return a feature name without a leading modality prefix."""
    text = str(name).strip()
    if ":" in text:
        text = text.split(":", 1)[1]
    return text


def feature_token(name) -> str:
    """Case-insensitive gene matching token shared by source and target schemas."""
    return bare_feature_symbol(name).upper()


def choose_decoder_mode(mode: str, pls_decoder_path: Path) -> str:
    mode = mode.lower()
    valid_modes = {"auto", "pls", "neural"}
    if mode not in valid_modes:
        raise ValueError(f"DECODER_MODE must be one of {sorted(valid_modes)}; got {mode!r}")
    if mode == "pls":
        if not pls_decoder_path.exists():
            raise FileNotFoundError(f"Requested PLS decoder is missing: {pls_decoder_path}")
        return "pls"
    if mode == "neural":
        return "neural"
    if pls_decoder_path.exists():
        return "pls"
    print(f"[WARN] PLS decoder missing at {pls_decoder_path}; using neural MSI decoder.")
    return "neural"


def matrix_rows_to_numpy(matrix, indices):
    rows = matrix[indices]
    if sp.issparse(rows):
        rows = rows.toarray()
    return np.asarray(rows, dtype=np.float32)


def batch_index_for_model(model, indices):
    batch_codes = getattr(model, "batch_codes", None)
    if batch_codes is None:
        return None

    idx = np.asarray(indices, dtype=int)
    codes = []
    for code in batch_codes:
        if torch.is_tensor(code):
            code_values = code.detach().cpu().numpy()
        else:
            code_values = np.asarray(code)
        codes.append(
            torch.as_tensor(code_values[idx], dtype=torch.long, device=model.device)
            .unsqueeze(1)
        )
    return torch.hstack(codes)


@torch.no_grad()
def encode_model(model, batch_size: int, *, decode_neural: bool = False):
    model.eval()
    z_joint, z_st, neural_msi = [], [], []

    for batch_idx in model.as_dataloader(batch_size=batch_size, shuffle=False):
        indices = batch_idx[0].detach().cpu().numpy()
        X_batch = torch.as_tensor(
            matrix_rows_to_numpy(model.X, indices),
            dtype=torch.float32,
            device=model.device,
        )
        H = model.encode(X_batch)
        z_joint.append(H["q_mu"].detach().cpu().numpy())
        z_st_batch = H["st"]["q_mu"]
        z_st.append(z_st_batch.detach().cpu().numpy())

        if decode_neural:
            z_decode = z_st_batch
            batch_index = batch_index_for_model(model, indices)
            if batch_index is not None:
                z_decode = torch.hstack([z_decode, batch_index.to(z_decode.device)])
            decoded = model.px_sm_scale_decoder(model.decoder_sm(z_decode.to(model.device)))
            neural_msi.append(decoded.detach().cpu().numpy())

    decoded_msi = np.vstack(neural_msi).astype(np.float32) if decode_neural else None
    return np.vstack(z_joint).astype(np.float32), np.vstack(z_st).astype(np.float32), decoded_msi


def decode_msi_pls(embedding: np.ndarray, pls_decoder_path: Path):
    payload = joblib.load(pls_decoder_path)
    pls = payload["model"]
    target_features = [str(f) for f in payload["target_features"]]
    expected_dim = getattr(pls, "n_features_in_", None)
    if expected_dim is not None and embedding.shape[1] != expected_dim:
        raise ValueError(
            f"PLS decoder expects {expected_dim} latent dimensions, "
            f"but student_X_emb has {embedding.shape[1]}."
        )

    msi_matrix = np.asarray(pls.predict(embedding), dtype=np.float32)
    if msi_matrix.ndim == 1:
        msi_matrix = msi_matrix[:, None]
    if msi_matrix.shape[1] != len(target_features):
        raise ValueError(
            f"PLS output has {msi_matrix.shape[1]} columns but "
            f"{len(target_features)} target features were recorded."
        )

    metadata = {
        "mode": "pls",
        "decoder_path": str(pls_decoder_path),
        "embedding_key": payload.get("embedding_key", "student_X_emb"),
        "target_transform": payload.get("target_transform", {}),
        "run_id": payload.get("run_id", SAMPLE_ID),
    }
    return msi_matrix, target_features, metadata


def attach_msi_output(
    adata,
    *,
    obsm_key: str,
    msi_matrix: np.ndarray,
    msi_feature_names: list[str],
    decoder_metadata: dict,
    gene_overlap: dict,
):
    if msi_matrix.shape != (adata.n_obs, len(msi_feature_names)):
        raise ValueError(
            f"{obsm_key} shape {msi_matrix.shape} does not match "
            f"(n_obs={adata.n_obs}, n_features={len(msi_feature_names)})."
        )

    adata.obsm[obsm_key] = msi_matrix
    adata.uns[f"{obsm_key}_feature_names"] = np.asarray(msi_feature_names, dtype=str)
    decoder_metadata["gene_overlap"] = gene_overlap
    adata.uns[f"{obsm_key}_decoder"] = decoder_metadata
    print(
        f"[INFO] {obsm_key}: decoded MSI matrix {msi_matrix.shape} "
        f"via {decoder_metadata['mode']}"
    )

    if DOPAMINE_FEATURE not in msi_feature_names:
        print(f"[WARN] {DOPAMINE_FEATURE} not found in {obsm_key} features.")
        return None

    dopamine_idx = msi_feature_names.index(DOPAMINE_FEATURE)
    dopamine_obs_key = f"{obsm_key}_Dopamine"
    adata.obs[dopamine_obs_key] = msi_matrix[:, dopamine_idx]
    print(f"[INFO] Wrote obs['{dopamine_obs_key}'] from {DOPAMINE_FEATURE}")
    return dopamine_obs_key


def write_msi_cache(
    *,
    obsm_key: str,
    msi_matrix: np.ndarray,
    msi_feature_names: list[str],
    dopamine_obs_key,
    decoder_metadata: dict,
):
    npz_path = PROJ_DIR / f"{obsm_key}_imputed.npz"
    np.savez_compressed(
        npz_path,
        **{
            obsm_key: msi_matrix,
            "msi_feature_names": np.asarray(msi_feature_names, dtype=str),
            "obs_names": np.asarray(target_joint.obs_names.tolist(), dtype=str),
        },
    )

    json_path = PROJ_DIR / f"{obsm_key}_imputed.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "sample_id": SAMPLE_ID,
                "target_rna_path": str(TARGET_RNA_PATH),
                "msi_obsm_key": obsm_key,
                "msi_feature_names": msi_feature_names,
                "dopamine_feature": DOPAMINE_FEATURE,
                "dopamine_obs_key": dopamine_obs_key,
                "decoder": decoder_metadata,
            },
            handle,
            indent=2,
        )
    print(f"[INFO] Wrote MSI cache to {npz_path} and {json_path}")

#%% Load the trained feature space (template) and the target RNA.

template = sc.read_h5ad(MODEL_DIR / "joint_adata.h5ad")
st_mask = (template.var["type"].values == "ST")
sm_mask = (template.var["type"].values == "SM")
st_cols = np.where(st_mask)[0]
# SMA RNA gene symbols, in the exact column order encoder_ST expects.
sma_genes = pd.Index([bare_feature_symbol(v) for v in template.var_names[st_mask]])
sma_tokens = pd.Index([feature_token(v) for v in template.var_names[st_mask]])
print(f"[INFO] Template: {template.shape} "
      f"({int(st_mask.sum())} ST + {int(sm_mask.sum())} SM features)")

target = sc.read_h5ad(TARGET_RNA_PATH)
target.obsm['spatial'] = np.array([1, -1]) * target.obsm['spatial']
target.var_names_make_unique()
target_counts = target.layers["counts"] if "counts" in target.layers else target.X
print(f"[INFO] Target RNA: {target.shape}")

#%% Align RNA by case-insensitive bare gene symbol, reindex into SMA order, zero-fill missing.
# We keep all SMA ST genes because the encoder has a fixed input dimension.

target_token_to_idx = {}
duplicate_target_tokens = set()
for i, token in enumerate(feature_token(v) for v in target.var_names):
    if token and token not in target_token_to_idx:
        target_token_to_idx[token] = i
    elif token:
        duplicate_target_tokens.add(token)

idx = np.asarray([target_token_to_idx.get(token, -1) for token in sma_tokens], dtype=int)
present = idx >= 0
n_overlap = int(present.sum())
overlap_fraction = n_overlap / max(len(sma_genes), 1)
print(f"[INFO] Gene overlap: {n_overlap}/{len(sma_genes)} SMA ST genes found in target "
      f"({100 * overlap_fraction:.1f}%).")
if duplicate_target_tokens:
    print(f"[INFO] Duplicate target gene tokens ignored after first occurrence: "
          f"{sorted(duplicate_target_tokens)[:10]}")
missing_examples = list(sma_genes[~present][:10])
print(f"[INFO] Example missing genes (zero-filled): {missing_examples}")
if overlap_fraction < MIN_ST_OVERLAP_FRACTION:
    raise ValueError(
        f"Gene overlap {overlap_fraction:.3f} is below "
        f"MIN_ST_OVERLAP_FRACTION={MIN_ST_OVERLAP_FRACTION:.3f}."
    )

gene_overlap = {
    "n_overlap": n_overlap,
    "n_reference_st": int(len(sma_genes)),
    "overlap_fraction": float(overlap_fraction),
    "min_overlap_fraction": float(MIN_ST_OVERLAP_FRACTION),
    "missing_examples": [str(v) for v in missing_examples],
    "duplicate_target_token_examples": sorted(duplicate_target_tokens)[:10],
}

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

#%% Load the SMA-trained student, encode target RNA, and impute MSI with PLS.

student_ckpt_path = MODEL_DIR / "student.pt"
pls_decoder_path = MODEL_DIR / "student_pls_decoder.joblib"
decoder_mode = choose_decoder_mode(DECODER_MODE, pls_decoder_path)

student_model = load_spatialjepa_model(student_ckpt_path, target_joint, device=DEVICE)
Z_joint, Z_st, neural_msi = encode_model(
    student_model,
    BATCH_SIZE,
    decode_neural=(decoder_mode == "neural"),
)
assert not np.isnan(Z_joint).any(), "student: NaNs in joint embedding"
assert not np.isnan(Z_st).any(), "student: NaNs in RNA-only embedding"
target_joint.obsm["student_X_emb"] = Z_joint
target_joint.obsm["student_X_emb_st"] = Z_st
print(f"[INFO] student: joint embedding {Z_joint.shape}; RNA-only embedding {Z_st.shape}")

if decoder_mode == "pls":
    msi_matrix, msi_feature_names, decoder_metadata = decode_msi_pls(
        target_joint.obsm["student_X_emb"],
        pls_decoder_path,
    )
else:
    if neural_msi is None:
        raise RuntimeError("Internal error: neural MSI decode was requested but not computed.")
    msi_matrix = neural_msi
    msi_feature_names = [str(v) for v in template.var_names[sm_mask]]
    decoder_metadata = {
        "mode": "neural",
        "decoder_path": str(student_ckpt_path),
        "embedding_key": "student_X_emb_st",
        "target_transform": {"output_space": "student_neural_sm_scale"},
        "run_id": SAMPLE_ID,
    }

student_dopamine_obs_key = attach_msi_output(
    target_joint,
    obsm_key=MSI_OBSM_KEY,
    msi_matrix=msi_matrix,
    msi_feature_names=msi_feature_names,
    decoder_metadata=decoder_metadata,
    gene_overlap=gene_overlap,
)

del student_model
torch.cuda.empty_cache()

write_msi_cache(
    obsm_key=MSI_OBSM_KEY,
    msi_matrix=msi_matrix,
    msi_feature_names=msi_feature_names,
    dopamine_obs_key=student_dopamine_obs_key,
    decoder_metadata=decoder_metadata,
)

#%% Load the SMA-trained teacher, encode target RNA, and impute MSI with its neural decoder.

teacher_ckpt_path = MODEL_DIR / "teacher.pt"
teacher_model = load_spatialjepa_model(teacher_ckpt_path, target_joint, device=DEVICE)
Z_teacher_joint, Z_teacher_st, teacher_msi = encode_model(
    teacher_model,
    BATCH_SIZE,
    decode_neural=True,
)
assert not np.isnan(Z_teacher_joint).any(), "teacher: NaNs in joint embedding"
assert not np.isnan(Z_teacher_st).any(), "teacher: NaNs in RNA-only embedding"
target_joint.obsm["teacher_X_emb"] = Z_teacher_joint
target_joint.obsm["teacher_X_emb_st"] = Z_teacher_st
print(
    f"[INFO] teacher: joint embedding {Z_teacher_joint.shape}; "
    f"RNA-only embedding {Z_teacher_st.shape}"
)

teacher_msi_feature_names = [str(v) for v in template.var_names[sm_mask]]
teacher_decoder_metadata = {
    "mode": "neural",
    "decoder_path": str(teacher_ckpt_path),
    "embedding_key": "teacher_X_emb_st",
    "target_transform": {"output_space": "teacher_neural_sm_scale"},
    "run_id": SAMPLE_ID,
}
teacher_dopamine_obs_key = attach_msi_output(
    target_joint,
    obsm_key=TEACHER_MSI_OBSM_KEY,
    msi_matrix=teacher_msi,
    msi_feature_names=teacher_msi_feature_names,
    decoder_metadata=teacher_decoder_metadata,
    gene_overlap=gene_overlap,
)

del teacher_model
torch.cuda.empty_cache()

write_msi_cache(
    obsm_key=TEACHER_MSI_OBSM_KEY,
    msi_matrix=teacher_msi,
    msi_feature_names=teacher_msi_feature_names,
    dopamine_obs_key=teacher_dopamine_obs_key,
    decoder_metadata=teacher_decoder_metadata,
)

#%% Downstream UMAP / Leiden per model (sanity check that the SMA RNA encoder transfers).

def embed_and_plot(name: str, dopamine_obs_key=None) -> None:
    emb_key = f"{name}_X_emb_st"
    neigh_key = f"{name}_neighbors"
    umap_key = f"{name}_umap"
    cluster_key = f"{name}_leiden"

    sc.pp.neighbors(target_joint, use_rep=emb_key, n_neighbors=15, key_added=neigh_key)
    sc.tl.umap(target_joint, min_dist=0.3, neighbors_key=neigh_key)
    target_joint.obsm[umap_key] = target_joint.obsm["X_umap"].copy()
    sc.tl.leiden(target_joint, key_added=cluster_key, neighbors_key=neigh_key)

    color = [cluster_key]
    for c in ("RNA_clusters", "ATAC_clusters", "batch", dopamine_obs_key):
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
        color=[c for c in color if c != "batch"],
        show=False,
        return_fig=True,
        s=60,
    )
    spatial_path = PROJ_DIR / f"{name}_spatial.png"
    fig_spatial.savefig(spatial_path, dpi=150, bbox_inches="tight")
    plt.close(fig_spatial)
    print(f"[INFO] {name}: wrote {spatial_path}")

embed_and_plot("student", student_dopamine_obs_key)
embed_and_plot("teacher", teacher_dopamine_obs_key)

#%% Format MSI matrices as Anndata objects.
student_msi_adata = AnnData(
    X=msi_matrix,
    obs=target_joint.obs.copy(),
    var=pd.DataFrame(index=msi_feature_names),
    obsm={
        "ST_student_umap": np.asarray(target_joint.obsm["student_umap"]),
        "spatial": np.asarray(target.obsm["spatial"])
        }
)
teacher_msi_adata = AnnData(
    X=teacher_msi,
    obs=target_joint.obs.copy(),
    var=pd.DataFrame(index=teacher_msi_feature_names),
    obsm={
        "ST_teacher_umap": np.asarray(target_joint.obsm["teacher_umap"]),
        "spatial": np.asarray(target.obsm["spatial"])
    }
)

## UMAP of the MSI matrices, in ST coordiantes
sc.pl.embedding(student_msi_adata, basis="ST_student_umap", color='msi:Dopamine')
sc.pl.embedding(teacher_msi_adata, basis="ST_teacher_umap", color='msi:Dopamine')

## spatial MSI plots

# Define the thesis figures directory, resolved relative to this script
THESIS_FIG_DIR = (Path(__file__).parent / "../../THESIS_base/overleaf-cibb-2026/figures").resolve()
THESIS_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Student MSI spatial plot
student_fig = sc.pl.embedding(
    student_msi_adata,
    basis="spatial",
    color=['msi:Dopamine', 'RNA_clusters', 'ATAC_clusters'],
    size=60,
    ncols=3,
    show=False,
    return_fig=True
)
student_fig_path = THESIS_FIG_DIR / "p22_student_imputed_msi_spatial.png"
student_fig.savefig(student_fig_path, dpi=150, bbox_inches="tight")
plt.close(student_fig)
print(f"[INFO] Saved student MSI spatial figure: {student_fig_path}")

# Teacher MSI spatial plot
teacher_fig = sc.pl.embedding(
    teacher_msi_adata,
    basis="spatial",
    color=['msi:Dopamine', 'RNA_clusters', 'ATAC_clusters'],
    size=60,
    ncols=3,
    show=False,
    return_fig=True
)
teacher_fig_path = THESIS_FIG_DIR / "p22_teacher_imputed_msi_spatial.png"
teacher_fig.savefig(teacher_fig_path, dpi=150, bbox_inches="tight")
plt.close(teacher_fig)
print(f"[INFO] Saved teacher MSI spatial figure: {teacher_fig_path}")

## train UMAP on the MSI matrices
sc.pp.pca(student_msi_adata, n_comps=50)
sc.pp.neighbors(student_msi_adata)
sc.tl.umap(student_msi_adata)

sc.pp.pca(teacher_msi_adata, n_comps=50)
sc.pp.neighbors(teacher_msi_adata)
sc.tl.umap(teacher_msi_adata)

## plot UMAP of the MSI matrices, in MSI coordinates
sc.pl.umap(student_msi_adata, color='msi:Dopamine')
sc.pl.umap(teacher_msi_adata, color='msi:Dopamine')

#%% Student vs teacher agreement (distillation-transfer sanity check).

Zs, Zt = target_joint.obsm["student_X_emb_st"], target_joint.obsm["teacher_X_emb_st"]
per_dim_r = [float(np.corrcoef(Zs[:, d], Zt[:, d])[0, 1]) for d in range(Zs.shape[1])]
print("[INFO] Student-vs-teacher per-latent-dim Pearson r:")
print("       " + ", ".join(f"{r:.3f}" for r in per_dim_r))
print(f"[INFO] Mean |r| across dims: {np.nanmean(np.abs(per_dim_r)):.3f}")

#%% Save the embedded target with MSI imputation.

out_path = PROJ_DIR / "target_spatial_atac_rna_embedded.h5ad"
target_joint.write_h5ad(out_path)
print(f"[INFO] Saved embedded target to {out_path}")

# %%
