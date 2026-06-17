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
)

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
#joint_adata = load_joint_adata()

sample_id = "V11T17-102_C1"
METADATA_PATH = Path(os.path.join(os.getenv("BAKLAVA_BASE_DIR"), "data", "vicari_2023", "mendeley_sma", "metadata.csv"))
metadata = pd.read_csv(METADATA_PATH)
sample_metadata = metadata.loc[metadata["Sample.ID"].eq(sample_id)]
print(sample_metadata.loc[~sample_metadata['Data.Type'].eq('RNA'), ['Sample.ID', 'sample', 'Matrix', 'Data.Type']].set_index('Sample.ID'))

# Keep only observations/cells/spots shared across modalities
joint_mudata = load_sample(sample_id, export_dir=Path(os.path.join(os.getenv("BAKLAVA_BASE_DIR"), "data", "vicari_2023", "h5mu_export")))
mu.pp.intersect_obs(joint_mudata)

rna = joint_mudata.mod["rna"]   # change to your key, e.g. "ST"
msi = joint_mudata.mod["msi"]   # change to your key, e.g. "SM"

# Make feature names unique and modality-prefixed
rna = rna.copy()
msi = msi.copy()

annotations = msi.var["annotation"].astype("string")
has_annotation = annotations.notna() & annotations.str.strip().ne("")
feature_ids = msi.var["feature_id"].astype("string") if "feature_id" in msi.var.columns else msi.var.index.astype("string")
msi_var_names = annotations.where(has_annotation, feature_ids)
msi.var_names = ["msi:" + str(v) for v in msi_var_names]
rna.var_names = ["rna:" + str(v) for v in rna.var_names]

if not msi.var_names.is_unique:
    msi = msi[:, ~msi.var_names.duplicated()]
    print("Removed duplicate MSI feature names!")

assert msi.var_names.is_unique, "MSI feature (var) names are not unique!"

# Concatenate features into one AnnData
adata = sc.concat(
    {"ST": rna, "SM": msi},
    axis=1,
    join="inner",
    label="type",
    merge="same",
)

joint_adata = smt.util._classes.AnnDataJointSMST(adata)

rna_spatial = joint_mudata.mod["rna"].uns["spatial"]
msi_spatial = joint_mudata.mod["msi"].uns["spatial"]
_assert_identical(rna_spatial, msi_spatial)

joint_adata.uns["spatial"] = rna_spatial
joint_adata.var = joint_adata.var.merge(joint_mudata.mod["msi"].var[['annotation']], left_on='feature_id', right_index=True, how='left')

# %% identify spatially highly variable genes and metabolites

if sample_id.startswith("V11T17-102"):
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

smt.pp.spatial_variable_joint_adata_sm_st(joint_adata,
                                         n_top_genes = 2000,
                                         n_top_metabolites = 800,
                                         add_key = "highly_variable_moranI")

joint_adata = joint_adata[:,joint_adata.var.highly_variable_moranI]
#DATA_DIR.mkdir(parents=True, exist_ok=True)
#joint_adata.write_h5ad(JOINT_HVF_PATH)

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

sc.pl.rank_genes_groups(striatum_adata, key="lesioned_vs_intact", n_genes=10)
sc.pl.rank_genes_groups(striatum_adata, key="intact_vs_lesioned", n_genes=10)

# %%

# instantiate models
teacher_model = spatialJEPA_model(joint_adata, graph_conv=True, full_graph=True)
student_model = spatialJEPA_model(joint_adata, graph_conv=True, full_graph=False)
#nonspatial_model = spatialJEPA_model(joint_adata, graph_conv=False, full_graph=False)

# instantiate trainer for SpatialJEPA
n_per_batch = 128
spatialjepa_trainer = SpatialJEPA_trainer(
    teacher_model,
    student_model,
    n_per_batch=n_per_batch,
)
# train models
loss_dict = spatialjepa_trainer.fit_teacher(
    max_epoch=250,
    lr=1e-5,
    mode="single",
)
# extract outputs
epoch_H = spatialjepa_trainer.epoch_H
batch_H = spatialjepa_trainer.batch_H
student_distill_loss = spatialjepa_trainer.student_distill_loss
epoch_student_distill_loss = spatialjepa_trainer.epoch_student_distill_loss

# %%
# CAPTION: Training loss curves for SpatialMETA (ConditionalVAESTSM) over 200 epochs. Each panel shows one tracked loss term (ST/SM reconstruction, correlation branches, KL, MMD). Use to assess convergence and balance between transcriptomics and metabolomics objectives.

fig,axes=plt.subplots(3,3,figsize=(20,10))
axes=axes.flatten()
for ax,(k,v) in zip(axes, loss_dict.items()):
    ax.plot(v)
    ax.set_title(k)

DOMAIN_MODELS = {
    "teacher": teacher_model,
    "student": student_model,
}
MARKER_GENES = [
    "Pcp4", "Tac1",          # V11L12-109_B1
    "Psap", "Sort1", "Snca", # V11L12-038_D1
    "Gba2", "Ugcg",          # V11L12-038_B1
    "Snca", "Pink1", "Park7",
    "mt-Nd2", "mt-Nd4", "mt-Nd1", "Ndufb9", "Ndufa13", "Ndufa4", "Ndufa3", # Complex I
]
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

def spatial_plot(joint_adata, mask=None):
    name_mapper = {
        "msi:PI(12:0/22:2(13Z,16Z)), PI(14:1(9Z)/20:1(11Z)), PI(15:1(9Z)/19:1(9Z)), PI(17:2(9Z,12Z)/17:0), PI(20:1(11Z)/14:1(9Z)), PI(18:2(9Z,12Z)/16:0), PI(17:1(9Z)/17:1(9Z)), PI(16:1(9Z)/18:1(9Z)), PI(14:0/20:2(11Z,14Z)), PI(17:0/17:2(9Z,12Z)), PI(19:1(9Z)/15:1(9Z)), PI(20:2(11Z,14Z)/14:0), PI(22:2(13Z,16Z)/12:0), PI(18:1(9Z)/16:1(9Z)), PI(16:0/18:2(9Z,12Z))":
        "msi:Phosphatidylinositols (PI)",
        "msi:(3'-sulfo)Galbeta-Cer(d18:1/24:0(2OH))":
        "Sulfated Galactosylceramide",
    }
    spatial_axes = sc.pl.spatial(
        joint_adata[mask] if mask is not None else joint_adata,
        img_key="hires",
        color_map="vlag",
        color=MARKER_FEATURES + ["lesion", "region"],
        layer="normalized",
        size=0.075,
        wspace=0.005,
        show=False,
    )
    if not isinstance(spatial_axes, list):
        spatial_axes = [spatial_axes]
    for ax in spatial_axes:
        title = ax.get_title()
        if title in name_mapper:
            ax.set_title(name_mapper[title])

spatial_plot(joint_adata)
spatial_plot(joint_adata, striatum_mask)

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


def plot_domain_results(domain: str, plot_marker: str) -> None:
    cluster_col = domain_key(domain, "VAE_clusters_latent10")

    sc.pl.embedding(
        joint_adata,
        color=[cluster_col, "region", "lesion", plot_marker],
        ncols=2,
        size=100,
        wspace=0.3,
        color_map="Reds",
        basis=domain_key(domain, "umap"),
    )
    sc.pl.spatial(
        joint_adata,
        img_key="hires",
        color=[cluster_col, "region", "lesion", plot_marker],
        size=0.075,
        show=False,
        ncols=2,
    )
    sc.pl.spatial(
        joint_adata,
        img_key="hires",
        color_map="vlag",
        color=MARKER_FEATURES,
        layer=domain_key(domain, "reconstruction"),
        size=0.075,
        wspace=0.005,
        show=False,
    )
    sc.pl.spatial(
        joint_adata,
        img_key="hires",
        color_map=CONTRIBUTION_CMAP,
        color=[
            domain_key(domain, "contribution_st"),
            domain_key(domain, "contribution_sm"),
        ],
        layer="normalized",
        wspace=0.005,
        show=False,
        alpha_img=0.1,
        size=0.075,
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
    plt.show()


for domain in DOMAIN_MODELS:
    process_latent_embedding(domain)

# %%
# CAPTION: Observed (normalized) spatial expression of striatal markers Pcp4 and Tac1 (RNA) and Dopamine (MSI) before model reconstruction.

if sample_id == "V11L12-038_B1":
    dopamine_striatum_lipids = [
        # --- Sphingolipids (GBA1 Pathway & Vesicular Dynamics) ---
        "Glucosylceramide (d18:1/24:0)",
        "Lactosylceramide (d18:1/12:0)",
        #"N-(2-hydroxydocosanoyl)-1-O-beta-D-glucosyl-15-methylhexadecasphing-4-enine",
        #"N-(2-hydroxytetracosanoyl)-1-O-beta-D-glucosyl-15-methylhexadecasphing-4-enine, N-(2-hydroxytricosanoyl)-D-galactosylsphingosine",
        #"beta-D-glucosyl-(1<->1')-N-tetracosanoyl-14-methylhexadecasphingosine, N-tetracosanoyl-1-O-beta-D-glucosyl-15-methylhexadecasphing-4-enine, N-tricosanoyl-D-galactosylsphingosine, beta-D-glucosyl-N-(tricosanoyl)sphingosine, beta-D-galactosyl-N-(tricosanoyl)sphingosine",
        
        # --- Sphingomyelins (Lipid Rafts & Dopamine Receptor Anchoring) ---
        "SM C16:1",
        "SM(d18:1/24:1(15Z))",
        
        # --- Polyunsaturated Phosphatidylcholines (Vulnerable to Dopamine Oxidative Stress) ---
        #"PC(14:0/20:4(5Z,8Z,11Z,14Z)), PC(14:0/20:4(8Z,11Z,14Z,17Z)), PC(18:4(6Z,9Z,12Z,15Z)/16:0), PE(22:4(7Z,10Z,13Z,16Z)/15:0), PE(15:0/22:4(7Z,10Z,13Z,16Z)), PC(20:4(8Z,11Z,14Z,17Z)/14:0), PC(20:3(5Z,8Z,11Z)/14:1(9Z)), PC(14:1(9Z)/20:3(8Z,11Z,14Z)), PC(14:1(9Z)/20:3(5Z,8Z,11Z)), PC(20:3(8Z,11Z,14Z)/14:1(9Z)), PC(16:1(9Z)/18:3(9Z,12Z,15Z)), PC(16:1(9Z)/18:3(6Z,9Z,12Z)), PC(18:3(6Z,9Z,12Z)/16:1(9Z)), PC(20:4(5Z,8Z,11Z,14Z)/14:0), PC(18:3(9Z,12Z,15Z)/16:1(9Z))",
        #"PC(16:0/18:4(6Z,9Z,12Z,15Z)), PC(14:0/20:4(5Z,8Z,11Z,14Z)), PC(14:0/20:4(8Z,11Z,14Z,17Z)), PC(18:4(6Z,9Z,12Z,15Z)/16:0), PE(22:4(7Z,10Z,13Z,16Z)/15:0), PE(15:0/22:4(7Z,10Z,13Z,16Z)), PC(20:4(8Z,11Z,14Z,17Z)/14:0), PC(20:3(5Z,8Z,11Z)/14:1(9Z)), PC(14:1(9Z)/20:3(8Z,11Z,14Z)), PC(14:1(9Z)/20:3(5Z,8Z,11Z)), PC(20:3(8Z,11Z,14Z)/14:1(9Z)), PC(16:1(9Z)/18:3(9Z,12Z,15Z)), PC(16:1(9Z)/18:3(6Z,9Z,12Z)), PC(18:3(6Z,9Z,12Z)/16:1(9Z)), PC(20:4(5Z,8Z,11Z,14Z)/14:0), PC(18:3(9Z,12Z,15Z)/16:1(9Z))",
        #"PC(18:2(9Z,12Z)/22:6(4Z,7Z,10Z,13Z,16Z,19Z)), PC(22:4(7Z,10Z,13Z,16Z)/18:4(6Z,9Z,12Z,15Z)), PC(22:6(4Z,7Z,10Z,13Z,16Z,19Z)/18:2(9Z,12Z)), PC(20:3(8Z,11Z,14Z)/20:5(5Z,8Z,11Z,14Z,17Z)), PC(20:5(5Z,8Z,11Z,14Z,17Z)/20:3(8Z,11Z,14Z)), PC(18:4(6Z,9Z,12Z,15Z)/22:4(7Z,10Z,13Z,16Z)), PC(20:4(5Z,8Z,11Z,14Z)/20:4(5Z,8Z,11Z,14Z))",
        #"PC(22:6(4Z,7Z,10Z,13Z,16Z,19Z)/22:6(4Z,7Z,10Z,13Z,16Z,19Z))",
        
        # --- Polyunsaturated Phosphatidylethanolamines & Plasmalogens (Vesicular Fusion) ---
        #"PE(18:4(6Z,9Z,12Z,15Z)/20:1(11Z)), PE(18:0/20:5(5Z,8Z,11Z,14Z,17Z)), PE(18:3(9Z,12Z,15Z)/20:2(11Z,14Z)), PE(20:4(5Z,8Z,11Z,14Z)/18:1(9Z)), PE(18:2(9Z,12Z)/20:3(8Z,11Z,14Z)), PE(20:2(11Z,14Z)/18:3(9Z,12Z,15Z)), PE(18:3(6Z,9Z,12Z)/20:2(11Z,14Z)), PE(16:1(9Z)/22:4(7Z,10Z,13Z,16Z)), PE(20:2(11Z,14Z)/18:3(6Z,9Z,12Z)), PE(18:1(9Z)/20:4(5Z,8Z,11Z,14Z)), PE(20:3(8Z,11Z,14Z)/18:2(9Z,12Z)), PE(22:4(7Z,10Z,13Z,16Z)/16:1(9Z)), PE(20:5(5Z,8Z,11Z,14Z,17Z)/18:0), PC(15:0/20:5(5Z,8Z,11Z,14Z,17Z)), PE(20:1(11Z)/18:4(6Z,9Z,12Z,15Z))",
        #"PE(P-18:0/22:6(4Z,7Z,10Z,13Z,16Z,19Z))"
    ]
    dopamine_striatum_lipids = ['msi:' + l for l in dopamine_striatum_lipids]

    sc.pl.spatial(
        joint_adata,
        img_key="hires",
        color_map="vlag",
        color=dopamine_striatum_lipids,
        layer="normalized",
        size=0.075,
        wspace=0.005,
        show=False,
    )

# CAPTION: UMAP of the joint ST+SM latent embedding (10-dim VAE, Leiden clusters). Colors: VAE clusters, tissue region, lesion status, and Dopamine (MSI). Shows how anatomy and pathology align with the integrated representation.

for domain in DOMAIN_MODELS:
    plot_domain_results(domain, MARKER_FEATURES[-3])

#%%
OUTPUT_DIR = Path(os.getenv("OUTPATH"))
MODEL_DIR = OUTPUT_DIR / "spatialjepa_models" / sample_id
MODEL_DIR.mkdir(parents=True, exist_ok=True)

save_spatialjepa_model(teacher_model, MODEL_DIR / "teacher.pt", full_graph=True)
save_spatialjepa_model(student_model, MODEL_DIR / "student.pt", full_graph=False)

# model.initialize_dataset() reads from adata.X at construction time
joint_adata.write_h5ad(MODEL_DIR / "joint_adata.h5ad")
