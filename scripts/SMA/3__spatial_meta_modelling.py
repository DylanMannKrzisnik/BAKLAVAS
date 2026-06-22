#%% load libraries
from dotenv import load_dotenv
load_dotenv(dotenv_path="/home/mcb/users/dmannk/BAKLAVA_base/BAKLAVA/.env")

import subprocess
from pathlib import Path
import os
import numpy as np
import spatialmeta as smt
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import muon as mu

import sys
sys.path.insert(0, os.path.join(os.getenv("BAKLAVA_ROOT"), "scripts", "SMA"))
from load_aligned_mudata import load_sample

BAKLAVA_BASE = Path(os.getenv("BAKLAVA_BASE_DIR"))
DATA_DIR = BAKLAVA_BASE / "data" / "spatialmeta_tutorial"
JOINT_RAW_PATH = DATA_DIR / "Y7_T_adata_joint_raw.h5ad"
JOINT_HVF_PATH = DATA_DIR / "Y7_T_adata_joint_hvf2800.h5ad"
JOINT_SPATIALMETA_PATH = DATA_DIR / "Y7_T_adata_joint_hvf2800_spatialmeta.h5ad"
ZENODO_JOINT_RAW_URL = (
    "https://zenodo.org/records/14986870/files/adata_joint_Y7_T_raw.h5ad?download=1"
)

FIG_DIR = Path("/home/mcb/users/dmannk/BAKLAVA_base/outputs/sea_ad_lipid_gps")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Optional target-aware ST panel restriction. When SMA_TARGET_PANEL points at a ranked
# gene CSV (see Lipid_GP/build_target_gene_ranking.py), the model's ST features are
# restricted to the top SMA_N_TARGET_TOP genes of that ranking before spatial-variability
# selection, so the trained encoder only keeps genes the transfer target also measures.
TARGET_PANEL_PATH = Path("/home/mcb/users/dmannk/BAKLAVA_base/outputs/target_gene_rankings/target_gene_ranking_CaH_Xenium_final.2026-01-07_protein_coding.csv")
N_TARGET_TOP = 2000

def load_joint_adata(path: Path = JOINT_RAW_PATH):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading tutorial data to {path}")
        subprocess.run(
            ["curl", "-L", "-o", str(path), ZENODO_JOINT_RAW_URL],
            check=True,
        )
    return smt.util._classes.AnnDataJointSMST(sc.read_h5ad(path))

def _assert_identical(a, b, path="uns['spatial']"):
    assert type(a) is type(b), f"{path}: type mismatch ({type(a)} vs {type(b)})"
    if isinstance(a, dict):
        assert a.keys() == b.keys(), f"{path}: keys differ"
        for key in a:
            _assert_identical(a[key], b[key], f"{path}[{key!r}]")
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b, err_msg=path)
    else:
        assert a == b, f"{path}: values differ"


import sys
sys.path.insert(0, os.path.join(os.getenv("BAKLAVA_ROOT"), "scripts", "SMA"))
from spatialjepa_model import (
    build_spatial_edge_index,
    build_identity_edge_index,
    replace_fc_encoder_with_gcn,
    patch_model_encode_for_gcn,
    spatialJEPA_model,
    save_spatialjepa_model,
    copy_decoder_weights,
)
from feature_panel import load_target_panel, restrict_st_to_target_panel

def _flatten_axes(plot_output):
    if plot_output is None:
        return []
    if isinstance(plot_output, dict):
        return [ax for item in plot_output.values() for ax in _flatten_axes(item)]
    if isinstance(plot_output, np.ndarray):
        return [ax for item in plot_output.flat for ax in _flatten_axes(item)]
    if isinstance(plot_output, (list, tuple)):
        return [ax for item in plot_output for ax in _flatten_axes(item)]
    return [plot_output]


def _section_slug(section_id: str) -> str:
    """V11T17-102_A1 -> A1 (short section label for figure filenames)."""
    return section_id.rsplit("_", 1)[-1] if "_" in section_id else section_id


def _save_plot(plot_output, filename: str) -> None:
    axes = _flatten_axes(plot_output)
    figures = []
    for ax in axes:
        fig = ax.figure
        if fig not in figures:
            figures.append(fig)
    if not figures:
        figures = [plt.gcf()]

    path = FIG_DIR / filename
    for i, fig in enumerate(figures):
        fig_path = path if len(figures) == 1 else path.with_name(f"{path.stem}_{i + 1}{path.suffix}")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] wrote {fig_path}")


def _save_labeled_plots(labeled_outputs: list[tuple[str, object]], filename: str) -> None:
    """Save one figure per (section_label, plot_output) pair, e.g. *_A1.png, *_C1.png."""
    path = FIG_DIR / filename
    stem, suffix = path.stem, path.suffix
    for label, plot_output in labeled_outputs:
        axes = _flatten_axes(plot_output)
        fig = axes[0].figure if axes else plt.gcf()
        fig_path = path.with_name(f"{stem}_{label}{suffix}")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] wrote {fig_path}")

class SpatialJEPA_trainer:
    """Train a full-graph teacher while distilling its batch embeddings to a student."""

    DISTILL_TARGETS = (
        (("q_mu",), 1.0),
        (("st", "q_mu"), 0.5),
        (("sm", "q_mu"), 0.5),
    )

    def __init__(
        self,
        teacher_model,
        student_model,
        n_per_batch=128,
        distill_targets=DISTILL_TARGETS,
        student_lr=1e-3,
        student_weight_decay=1e-4,
        student_optimizer=None,
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.n_per_batch = n_per_batch
        self.distill_targets = distill_targets
        self.student_optimizer = student_optimizer or optim.Adam(
            self.student_model.parameters(),
            lr=student_lr,
            weight_decay=student_weight_decay,
        )

        self.batches_per_epoch = len(
            self.teacher_model.as_dataloader(batch_size=n_per_batch, shuffle=True)
        )
        self.epoch_H = []
        self.batch_H = []
        self.current_epoch_H = []
        self.student_distill_loss = []
        self.epoch_student_distill_loss = []

    def get_expression_batch(self, model, indices):
        """Fetch a model input batch by original observation indices."""
        from scipy.sparse import issparse

        indices = torch.as_tensor(indices, dtype=torch.long).detach().cpu()
        indices_np = indices.numpy()
        if issparse(model.X):
            X_batch = model.X[indices_np].toarray()
        else:
            X_batch = model.X[indices_np]
        return torch.as_tensor(np.asarray(X_batch), dtype=torch.float32, device=model.device)

    def encode_indices(self, model, indices):
        """Encode the same observations used by the teacher batch."""
        indices = torch.as_tensor(indices, dtype=torch.long).detach().cpu()
        had_current_indices = hasattr(model, "_current_batch_indices")
        old_current_indices = getattr(model, "_current_batch_indices", None)
        model._current_batch_indices = indices
        try:
            X_batch = self.get_expression_batch(model, indices)
            return model.encode(X_batch)
        finally:
            if had_current_indices:
                model._current_batch_indices = old_current_indices
            else:
                delattr(model, "_current_batch_indices")

    @staticmethod
    def get_nested(mapping, path):
        value = mapping
        for key in path:
            value = value[key]
        return value

    @staticmethod
    def detach_H(H, keys=("q_mu",), nested_keys=("q_mu",), device=None):
        def detach_tensor(x):
            x = x.detach()
            return x if device is None else x.to(device)

        out = {}
        for k, v in H.items():
            if k in ("st", "sm"):
                out[k] = {
                    sk: detach_tensor(sv)
                    for sk, sv in v.items()
                    if sk in nested_keys and torch.is_tensor(sv)
                }
            elif k in keys and torch.is_tensor(v):
                out[k] = detach_tensor(v)
        return out

    def h_distillation_loss(self, H_student, H_teacher):
        loss = None
        for path, weight in self.distill_targets:
            student_value = self.get_nested(H_student, path)
            teacher_value = self.get_nested(H_teacher, path).to(student_value.device)
            term = weight * torch.nn.functional.mse_loss(student_value, teacher_value)
            loss = term if loss is None else loss + term
        return loss

    def jepa_distillation_step(self, H_teacher, indices):
        self.student_model.train()
        H_student = self.encode_indices(self.student_model, indices)
        distil_loss = self.h_distillation_loss(H_student, H_teacher)
        self.student_optimizer.zero_grad(set_to_none=True)
        distil_loss.backward()
        self.student_optimizer.step()
        return float(distil_loss.detach().cpu())

    def capture_teacher_outputs(self, outputs):
        H, Rs, L = outputs
        indices = getattr(self.teacher_model, "_current_batch_indices", None)
        if indices is None:
            raise RuntimeError(
                "Teacher batch indices were not available during distillation. "
                "Make sure patch_model_encode_for_gcn(teacher_model) has been applied."
            )

        indices = indices.detach().cpu().clone()
        H_teacher = self.detach_H(H)
        distil_loss = self.jepa_distillation_step(H_teacher, indices)

        batch_record = dict(
            indices=indices,
            H=self.detach_H(H, device="cpu"),
            student_distill_loss=distil_loss,
        )
        self.batch_H.append(batch_record)
        self.current_epoch_H.append(batch_record)
        self.student_distill_loss.append(distil_loss)

        if len(self.current_epoch_H) == self.batches_per_epoch:
            self.epoch_H.append(self.current_epoch_H.copy())
            self.epoch_student_distill_loss.append(
                float(np.mean([
                    record["student_distill_loss"]
                    for record in self.current_epoch_H
                ]))
            )
            self.current_epoch_H.clear()

    def fit_teacher(self, **fit_kwargs):
        teacher_forward = self.teacher_model.forward

        def teacher_forward_with_distillation(*args, **kwargs):
            outputs = teacher_forward(*args, **kwargs)
            self.capture_teacher_outputs(outputs)
            return outputs

        self.teacher_model.forward = teacher_forward_with_distillation
        try:
            loss_dict = self.teacher_model.fit(
                n_per_batch=self.n_per_batch,
                **fit_kwargs,
            )
        finally:
            self.teacher_model.forward = teacher_forward

        loss_dict["epoch_student_distill_loss_list"] = self.epoch_student_distill_loss
        return loss_dict


#%% load data

## sample notes
# V11T17-102_B1 probably has a large fold that increases dopamine and related genes

# sample_ids may be a single sample ID (str) for vertical-only integration, or a
# list of sample IDs that share a prefix (e.g. all "V11T17-102_*") to additionally
# enable horizontal (multi-section) integration of the same sample/donor.
#sample_ids = "V11T17-102_D1"
sample_ids = ["V11T17-102_A1", "V11T17-102_C1", "V11T17-102_D1"]

SAMPLE_IDS = [sample_ids] if isinstance(sample_ids, str) else list(sample_ids)
SECTION_KEY = "section"
MULTI = len(SAMPLE_IDS) > 1

# All sections must come from the same sample/donor (shared prefix)
PREFIX = SAMPLE_IDS[0].split("_")[0]
assert all(s.split("_")[0] == PREFIX for s in SAMPLE_IDS), (
    f"All sample IDs must share a prefix for horizontal integration; got {SAMPLE_IDS}"
)
RUN_ID = SAMPLE_IDS[0] if not MULTI else PREFIX
species = "human" if PREFIX == "V11T17-102" else "mouse"

METADATA_PATH = Path(os.path.join(os.getenv("BAKLAVA_BASE_DIR"), "data", "vicari_2023", "mendeley_sma", "metadata.csv"))
metadata = pd.read_csv(METADATA_PATH)
sample_metadata = metadata.loc[metadata["Sample.ID"].isin(SAMPLE_IDS)]
print(sample_metadata.loc[~sample_metadata['Data.Type'].eq('RNA'), ['Sample.ID', 'sample', 'Matrix', 'Data.Type']].set_index('Sample.ID'))

H5MU_EXPORT_DIR = Path(os.path.join(os.getenv("BAKLAVA_BASE_DIR"), "data", "vicari_2023", "h5mu_export"))


def assemble_section(sample_id):
    """Load one section and return a per-section ST+SM AnnData tagged with its section id."""
    # Keep only observations/cells/spots shared across modalities
    joint_mudata = load_sample(sample_id, export_dir=H5MU_EXPORT_DIR)
    mu.pp.intersect_obs(joint_mudata)

    rna = joint_mudata.mod["rna"].copy()   # change to your key, e.g. "ST"
    msi = joint_mudata.mod["msi"].copy()   # change to your key, e.g. "SM"

    # Make feature names unique and modality-prefixed
    annotations = msi.var["annotation"].astype("string")
    has_annotation = annotations.notna() & annotations.str.strip().ne("")
    feature_ids = msi.var["feature_id"].astype("string") if "feature_id" in msi.var.columns else msi.var.index.astype("string")
    msi_var_names = annotations.where(has_annotation, feature_ids)
    msi.var_names = ["msi:" + str(v) for v in msi_var_names]
    rna.var_names = ["rna:" + str(v) for v in rna.var_names]

    if not msi.var_names.is_unique:
        msi = msi[:, ~msi.var_names.duplicated()]
        print(f"[{sample_id}] Removed duplicate MSI feature names!")
    assert msi.var_names.is_unique, "MSI feature (var) names are not unique!"

    # Concatenate features (modalities) into one AnnData
    adata = sc.concat(
        {"ST": rna, "SM": msi},
        axis=1,
        join="inner",
        label="type",
        merge="same",
    )
    # carry spot coordinates explicitly (per-section pixel coords)
    adata.obsm["spatial"] = rna.obsm["spatial"]

    rna_spatial = joint_mudata.mod["rna"].uns["spatial"]
    msi_spatial = joint_mudata.mod["msi"].uns["spatial"]
    _assert_identical(rna_spatial, msi_spatial)

    # Re-key the spatial dict under the section id so keys stay unique across sections
    spatial_uns = dict(rna_spatial)
    if len(spatial_uns) == 1:
        spatial_uns = {sample_id: next(iter(spatial_uns.values()))}
    adata.uns["spatial"] = spatial_uns

    adata.var = adata.var.merge(msi.var[['annotation']], left_on='feature_id', right_index=True, how='left')
    adata.obs[SECTION_KEY] = sample_id
    return adata


sections = [assemble_section(s) for s in SAMPLE_IDS]
if MULTI:
    # concat spots across sections; intersect features, keep shared var columns
    adata = sc.concat(sections, axis=0, join="inner", merge="same")
    merged_spatial = {}
    for sec in sections:
        merged_spatial.update(sec.uns["spatial"])
    adata.uns["spatial"] = merged_spatial
else:
    adata = sections[0]
adata.obs[SECTION_KEY] = adata.obs[SECTION_KEY].astype("category")

joint_adata = smt.util._classes.AnnDataJointSMST(adata)

# %% identify spatially highly variable genes and metabolites

if species == "human":
    joint_adata = smt.pp.removeHSP_MT_RPL_DNAJ(joint_adata) # remove HSP, MT, RPL, DNAJ features in human
else:
    joint_adata = smt.pp.removeHsp_mt_Rpl_Dnaj(joint_adata) # remove Hsp, mt, Rpl, Dnaj features in mouse

joint_adata.layers["counts"] = joint_adata.X.copy()

smt.pp.normalize_total_joint_adata_sm_st(
    joint_adata,
    target_sum_SM=1e4,
    target_sum_ST=1e4
)

joint_adata.layers["normalized"] = joint_adata.X.copy()
joint_adata.raw = joint_adata

# Restrict ST features to a ranked target panel (if provided) before Moran's-I selection,
# so the candidate ST pool only contains genes the transfer target measures. Whitelisted
# target genes sit at the top of the ranking, so they are always in the candidate pool.
joint_adata.uns["target_gene_panel_source"] = (
    Path(TARGET_PANEL_PATH).name if TARGET_PANEL_PATH else "nan"
)
if TARGET_PANEL_PATH:
    joint_adata, panel_diag = restrict_st_to_target_panel(
        joint_adata,
        load_target_panel(TARGET_PANEL_PATH),
        n_target_top=N_TARGET_TOP,
    )
    print(f"[panel] restricted ST to target panel: {panel_diag}")

# When integrating multiple sections, the batch_key branch also removes features whose abundance differs strongly *between* sections (assumed technical batch effects). That
# filter cannot tell a batch effect from genuine cross-section biology: e.g. Dopamine
# (and anything co-depleted with it) varies across sections by lesion extent (log2FC ~= 19.7 between A1 and C1) and gets dropped at the default min_logfc=3. Use a very lenient threshold so only the most extreme present/absent artifacts (log2FC > 25) are removed, retaining the dopamine-correlated biological axis.
smt.pp.spatial_variable_joint_adata_sm_st(joint_adata,
                                         n_top_genes = 2000,
                                         n_top_metabolites = 800,
                                         add_key = "highly_variable_moranI",
                                         batch_key = SECTION_KEY if MULTI else None,
                                         min_frac = 0.8,
                                         min_logfc = 25)

joint_adata = joint_adata[:,joint_adata.var.highly_variable_moranI]

# Drop spots with zero RNA library over the retained features. The ZINB ST decoder scales px_rna_scale by lib_size = X_ST.sum(1); a zero-library spot forces logits = log(mu/theta) = log(0) = -inf -> NaN loss on the first forward (independent of learning rate). C1 is the only human sample with none of these.
st_mask = joint_adata.var["type"].eq("ST").to_numpy()
st_lib = np.asarray(joint_adata[:, st_mask].X.sum(1)).ravel()
n_empty_rna = int((st_lib == 0).sum())
if n_empty_rna:
    print(f"Dropping {n_empty_rna} spots with zero RNA library over HVF genes")
    joint_adata = joint_adata[st_lib > 0].copy()

#%%

#joint_adata = sc.read_h5ad(JOINT_HVF_PATH)
joint_adata.X = joint_adata.layers["counts"]

smt.pp.normalize_total_joint_adata_sm_st( # again, now on the spatially variable features
    joint_adata,
    target_sum_SM=1e3,
    target_sum_ST=None
)

sm_mask = joint_adata.var["type"].eq("SM").to_numpy()
joint_adata.X[:, sm_mask] = np.log1p(joint_adata.X[:, sm_mask])

# Optional: preserves zeros, unlike zero_center=True
sm = joint_adata[:, sm_mask].copy()
sc.pp.scale(sm, zero_center=False, max_value=10)
joint_adata.X[:, sm_mask] = sm.X

#%% differential expression/abundance analysis, comparing intact vs lesioned striatum
if species == "human":
    # create artificial lesion mask, using a per-section Dopamine threshold
    x_dopamine = joint_adata[:, 'msi:Dopamine'].X.toarray().flatten()
    thresh_dopamine = {
        "V11T17-102_A1": 1.5,
        "V11T17-102_B1": 0.6,
        "V11T17-102_C1": 1.675,
        "V11T17-102_D1": 1.,
    }
    thresh_vec = joint_adata.obs[SECTION_KEY].map(thresh_dopamine).to_numpy(dtype=float)
    for s in SAMPLE_IDS:
        m = (joint_adata.obs[SECTION_KEY] == s).to_numpy()
        plt.figure(figsize=[3,2]); plt.hist(x_dopamine[m], bins=50); plt.axvline(thresh_dopamine[s], color='r'); plt.xlabel(f'Dopamine ({s})')
        _save_plot(None, f"{s}_dopamine_threshold.png")

    joint_adata.obs['lesion'] = pd.Series(x_dopamine < thresh_vec, index=joint_adata.obs_names).map({True: 'lesioned', False: 'intact'})
    striatum_adata = joint_adata.copy()

elif species == "mouse":
    striatum_mask = joint_adata.obs["region"].eq("striatum")
    striatum_adata = joint_adata[striatum_mask].copy()

striatum_adata.X = striatum_adata.layers["normalized"]

# reference="intact" reports lesioned vs intact; reference="lesioned" reports intact vs lesioned
sc.tl.rank_genes_groups(
    striatum_adata, groupby="lesion", reference="intact", method="wilcoxon", key_added="lesioned_vs_intact"
)
sc.tl.rank_genes_groups(
    striatum_adata, groupby="lesion", reference="lesioned", method="wilcoxon", key_added="intact_vs_lesioned"
)

_save_plot(
    sc.pl.rank_genes_groups(striatum_adata, key="lesioned_vs_intact", n_genes=10, show=False),
    f"{RUN_ID}_rank_genes_lesioned_vs_intact.png",
)
_save_plot(
    sc.pl.rank_genes_groups(striatum_adata, key="intact_vs_lesioned", n_genes=10, show=False),
    f"{RUN_ID}_rank_genes_intact_vs_lesioned.png",
)

#%% plot marker features

MARKER_GENES = [
    "Pcp4", "Tac1",          # V11L12-109_B1
    "Psap", "Sort1", "Snca", # V11L12-038_D1
    "Gba2", "Ugcg",          # V11L12-038_B1
    "Snca", "Pink1", "Park7",
    "mt-Nd2", "mt-Nd4", "mt-Nd1", "Ndufb9", "Ndufa13", "Ndufa4", "Ndufa3", # Complex I
]
if species == "human":
    MARKER_GENES = [g.upper() for g in MARKER_GENES]
    
MARKER_MSI = [
    "Dopamine",
    "(3'-sulfo)Galbeta-Cer(d18:1/24:0(2OH))",
    "307.07157500000005",
    "Glucosylceramide (d18:1/24:0)",
    "PI(12:0/22:2(13Z,16Z)), PI(14:1(9Z)/20:1(11Z)), PI(15:1(9Z)/19:1(9Z)), PI(17:2(9Z,12Z)/17:0), PI(20:1(11Z)/14:1(9Z)), PI(18:2(9Z,12Z)/16:0), PI(17:1(9Z)/17:1(9Z)), PI(16:1(9Z)/18:1(9Z)), PI(14:0/20:2(11Z,14Z)), PI(17:0/17:2(9Z,12Z)), PI(19:1(9Z)/15:1(9Z)), PI(20:2(11Z,14Z)/14:0), PI(22:2(13Z,16Z)/12:0), PI(18:1(9Z)/16:1(9Z)), PI(16:0/18:2(9Z,12Z))",
    #"5-Hydroxydantrolene",
]
MARKER_FEATURES = \
    ["rna:" + g for g in MARKER_GENES if "rna:" + g in joint_adata.var_names] + \
    ["msi:" + m for m in MARKER_MSI if "msi:" + m in joint_adata.var_names]

def spatial_plot(joint_adata, mask=None, filename=None):
    name_mapper = {
        "msi:PI(12:0/22:2(13Z,16Z)), PI(14:1(9Z)/20:1(11Z)), PI(15:1(9Z)/19:1(9Z)), PI(17:2(9Z,12Z)/17:0), PI(20:1(11Z)/14:1(9Z)), PI(18:2(9Z,12Z)/16:0), PI(17:1(9Z)/17:1(9Z)), PI(16:1(9Z)/18:1(9Z)), PI(14:0/20:2(11Z,14Z)), PI(17:0/17:2(9Z,12Z)), PI(19:1(9Z)/15:1(9Z)), PI(20:2(11Z,14Z)/14:0), PI(22:2(13Z,16Z)/12:0), PI(18:1(9Z)/16:1(9Z)), PI(16:0/18:2(9Z,12Z))":
        "msi:Phosphatidylinositols (PI)",
        "msi:(3'-sulfo)Galbeta-Cer(d18:1/24:0(2OH))":
        "Sulfated Galactosylceramide",
    }
    spatial_axes = sc.pl.spatial(
        joint_adata[mask] if mask is not None else joint_adata,
        img_key="hires" if species == "mouse" else None,
        color_map="vlag",
        color=MARKER_FEATURES + [key for key in ["lesion", "region"] if key in joint_adata.obs.keys()],
        layer="normalized",
        size=0.125 if species == "human" else 0.075,
        wspace=0.005,
        show=False,
    )
    if not isinstance(spatial_axes, list):
        spatial_axes = [spatial_axes]
    for ax in spatial_axes:
        title = ax.get_title()
        if title in name_mapper:
            ax.set_title(name_mapper[title])
    if filename is not None:
        _save_plot(spatial_axes, filename)

for sample_id in SAMPLE_IDS:
    section_adata = joint_adata[joint_adata.obs[SECTION_KEY].eq(sample_id)].copy()
    # keep the {library_id: {...}} wrapper so sc.pl.spatial sees a single library
    section_adata.uns['spatial'] = {sample_id: section_adata.uns['spatial'][sample_id]}
    spatial_plot(section_adata, filename=f"{sample_id}_marker_spatial.png")
    if species == "mouse":
        section_striatum_mask = section_adata.obs["region"].eq("striatum")
        spatial_plot(section_adata, section_striatum_mask, filename=f"{sample_id}_striatum_marker_spatial.png")

# %%

# instantiate models
# For horizontal integration (MULTI) pass the section as a batch key: this enables decoder batch conditioning + the MMD alignment loss, and a block-diagonal spatial graph (no cross-section edges) for the full-graph teacher.
batch_keys = [SECTION_KEY] if MULTI else None
section_key = SECTION_KEY if MULTI else None
teacher_model = spatialJEPA_model(joint_adata, graph_conv=True, full_graph=True, batch_keys=batch_keys, section_key=section_key)
student_model = spatialJEPA_model(joint_adata, graph_conv=True, full_graph=False, batch_keys=batch_keys, section_key=section_key)
#nonspatial_model = spatialJEPA_model(joint_adata, graph_conv=False, full_graph=False)
#nonspatial_model.fit(max_epoch=250, lr=1e-5, mode="single")

# instantiate trainer for SpatialJEPA
n_per_batch = 128
spatialjepa_trainer = SpatialJEPA_trainer(
    teacher_model,
    student_model,
    n_per_batch=n_per_batch,
)
# train models; mode="multi" upweights the MMD loss for horizontal integration
loss_dict = spatialjepa_trainer.fit_teacher(
    max_epoch=1000,
    lr=1e-5,
    mode="multi" if MULTI else "single",
)
# extract outputs
epoch_H = spatialjepa_trainer.epoch_H
batch_H = spatialjepa_trainer.batch_H
student_distill_loss = spatialjepa_trainer.student_distill_loss
epoch_student_distill_loss = spatialjepa_trainer.epoch_student_distill_loss

# CAPTION: Training loss curves for SpatialMETA (ConditionalVAESTSM) over 200 epochs. Each panel shows one tracked loss term (ST/SM reconstruction, correlation branches, KL, MMD). Use to assess convergence and balance between transcriptomics and metabolomics objectives.

fig,axes=plt.subplots(3,4,figsize=(26,10))
axes=axes.flatten()
for ax,(k,v) in zip(axes, loss_dict.items()):
    ax.plot(v)
    ax.set_title(k)
_save_plot(fig.axes, f"{RUN_ID}_training_losses.png")

# %%

DOMAIN_MODELS = {
    "teacher": teacher_model,
    "student": student_model,
}

CONTRIBUTION_CMAP = smt.pl.make_colormap(["#2ec4b6", "#ffffff", "#ff9f1c"])

def domain_key(domain: str, suffix: str) -> str:
    return f"{domain}_{suffix}"

def process_latent_embedding(domain: str) -> None:
    embedding_model = DOMAIN_MODELS[domain]

    Z = embedding_model.get_latent_embedding()
    X = embedding_model.get_normalized_expression()
    C = embedding_model.get_modality_contribution()

    joint_adata.layers[domain_key(domain, "reconstruction")] = X
    joint_adata.obsm[domain_key(domain, "X_emb")] = Z
    joint_adata.obs[domain_key(domain, "contribution_st")] = C
    joint_adata.obs[domain_key(domain, "contribution_sm")] = 1 - C

    neighbors_key = domain_key(domain, "neighbors")
    umap_key = domain_key(domain, "umap")

    sc.pp.neighbors(
        joint_adata,
        use_rep=domain_key(domain, "X_emb"),
        n_neighbors=15,
        key_added=neighbors_key,
    )
    sc.tl.umap(
        joint_adata,
        min_dist=1,
        spread=1,
        neighbors_key=neighbors_key,
    )
    joint_adata.obsm[umap_key] = joint_adata.obsm["X_umap"].copy()
    sc.tl.leiden(
        joint_adata,
        key_added=domain_key(domain, "VAE_clusters_latent10"),
        neighbors_key=neighbors_key,
    )


def _spatial_by_section(adata, **kwargs):
    """sc.pl.spatial once per section: with multiple sections, uns['spatial'] holds
    one library per section, so plotting the pooled object raises 'multiple libraries'.
    Subset to each section's spots and narrow uns to its single library. A single
    library (single-section run) falls through to one plain call.

    Returns a list of (section_label, plot_output) tuples for labeled figure saves.
    """
    libs = list(adata.uns["spatial"].keys()) if "spatial" in adata.uns else []
    if len(libs) <= 1:
        lib = libs[0] if libs else (SAMPLE_IDS[0] if SAMPLE_IDS else "all")
        return [(_section_slug(lib), sc.pl.spatial(adata, **kwargs))]
    outputs = []
    for lib in libs:
        sub = adata[adata.obs[SECTION_KEY].eq(lib)].copy()
        sub.uns["spatial"] = {lib: adata.uns["spatial"][lib]}
        outputs.append((_section_slug(lib), sc.pl.spatial(sub, **kwargs)))
    return outputs


def plot_domain_results(domain: str, plot_marker: str) -> None:
    cluster_col = domain_key(domain, "VAE_clusters_latent10")

    fig = sc.pl.embedding(
        joint_adata,
        color=[cluster_col, "lesion", plot_marker] + (["region"] if species == "mouse" else []),
        ncols=2 if species == "mouse" else 3,
        size=100,
        wspace=0.3,
        color_map="Reds",
        basis=domain_key(domain, "umap"),
        show=False,
        return_fig=True,
    )
    _save_plot(fig.axes, f"{RUN_ID}_{domain}_umap.png")
    _save_labeled_plots(
        _spatial_by_section(
            joint_adata,
            img_key="hires" if species == "mouse" else None,
            color=[cluster_col, "lesion", plot_marker] + (["region"] if species == "mouse" else []),
            size=0.125 if species == "human" else 0.075,
            show=False,
            ncols=2 if species == "mouse" else 3,
        ),
        f"{RUN_ID}_{domain}_spatial_clusters.png",
    )
    _save_labeled_plots(
        _spatial_by_section(
            joint_adata,
            img_key="hires" if species == "mouse" else None,
            color_map="vlag",
            color=MARKER_FEATURES,
            layer=domain_key(domain, "reconstruction"),
            size=0.125 if species == "human" else 0.075,
            wspace=0.005,
            show=False,
        ),
        f"{RUN_ID}_{domain}_spatial_reconstruction.png",
    )
    _save_labeled_plots(
        _spatial_by_section(
            joint_adata,
            img_key="hires" if species == "mouse" else None,
            color_map=CONTRIBUTION_CMAP,
            color=[
                domain_key(domain, "contribution_st"),
                domain_key(domain, "contribution_sm"),
            ],
            layer="normalized",
            wspace=0.005,
            show=False,
            alpha_img=0.1,
            size=0.125 if species == "human" else 0.075,
        ),
        f"{RUN_ID}_{domain}_spatial_modality_contribution.png",
    )

    obs_filter_df = pd.concat([
        joint_adata.obs[[cluster_col, domain_key(domain, "contribution_st")]]
        .rename(columns={domain_key(domain, "contribution_st"): "contribution"})
        .assign(type="st"),
        joint_adata.obs[[cluster_col, domain_key(domain, "contribution_sm")]]
        .rename(columns={domain_key(domain, "contribution_sm"): "contribution"})
        .assign(type="sm"),
    ])
    fig, ax = smt.pl.create_fig(figsize=(12, 4))
    sns.violinplot(
        data=obs_filter_df,
        x=cluster_col,
        y="contribution",
        hue="type",
        split=True,
        inner="quart",
        palette=["#2ec4b6", "#FFCC70"],
        scale="width",
        bw=0.2,
        cut=0,
    )
    plt.xticks(rotation=90)
    _save_plot(ax, f"{RUN_ID}_{domain}_cluster_modality_contribution.png")


for domain in DOMAIN_MODELS:
    process_latent_embedding(domain)

# %%
# CAPTION: UMAP of the joint ST+SM latent embedding (10-dim VAE, Leiden clusters). Colors: VAE clusters, tissue region, lesion status, and Dopamine (MSI). Shows how anatomy and pathology align with the integrated representation.
plot_marker = MARKER_FEATURES[-1]
print("Plotting marker:", plot_marker)

for domain in DOMAIN_MODELS:
    plot_domain_results(domain, plot_marker)

if MULTI:
    cluster_crosstab = pd.crosstab(joint_adata.obs['teacher_VAE_clusters_latent10'], joint_adata.obs['section'])
    ax = cluster_crosstab.plot(kind="bar", stacked=True, figsize=(10, 5))
    plt.title("Cluster distribution by section")
    plt.ylabel("Count")
    _save_plot(ax, f"{RUN_ID}_teacher_cluster_distribution_by_section.png")

#%%
OUTPUT_DIR = Path(os.getenv("OUTPATH"))
MODEL_DIR = OUTPUT_DIR / "spatialjepa_models" / RUN_ID
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# The student's decoder is never trained (distillation only touches the encoder/latent
# heads), so copy the teacher's trained, graph-free decoder weights before saving.
copied, skipped = copy_decoder_weights(teacher_model, student_model)
print(f"[student] copied decoder weights: {copied}; skipped (graph): {skipped}")

save_spatialjepa_model(teacher_model, MODEL_DIR / "teacher.pt", full_graph=True)
save_spatialjepa_model(student_model, MODEL_DIR / "student.pt", full_graph=False)

# model.initialize_dataset() reads from adata.X at construction time
joint_adata.write_h5ad(MODEL_DIR / "joint_adata.h5ad")

# %%
