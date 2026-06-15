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


def build_spatial_edge_index(
    coords: np.ndarray,
    n_neighbors: int = 6,
    device: str = "cpu",
):
    """kNN graph on spot coordinates for PyG GCNConv (spots = nodes)."""
    import torch
    from sklearn.neighbors import kneighbors_graph
    from torch_geometric.utils import from_scipy_sparse_matrix

    adj = kneighbors_graph(
        coords,
        n_neighbors=n_neighbors,
        mode="connectivity",
        include_self=False,
    )
    adj = adj.maximum(adj.T)
    edge_index, _ = from_scipy_sparse_matrix(adj.tocoo())
    return edge_index.to(device)


def build_identity_edge_index(
    num_nodes: int,
    device: str = "cpu",
):
    """Self-loop-only graph: each node only sends messages to itself."""
    import torch

    node_idx = torch.arange(num_nodes, dtype=torch.long, device=device)
    return torch.stack((node_idx, node_idx), dim=0)


def replace_fc_encoder_with_gcn(sae, edge_index, num_nodes: int, layer_idx: int = 0) -> None:
    """Replace one SAE encoder FCLayer with GCNConv + the same BN/ReLU/Dropout tail."""
    import torch.nn as nn
    from torch_geometric.nn import GCNConv

    enc_idx = layer_idx * 2
    fc_layer = sae.layers[enc_idx]
    if fc_layer is None:
        raise ValueError(f"encoder layer {layer_idx} is missing")

    post_modules = [
        module for module in fc_layer._fclayer if not isinstance(module, nn.Linear)
    ]

    class GCNEncoderLayer(nn.Module):
        """GCNConv drop-in replacement for SpatialMETA's encoder FCLayer."""

        def __init__(self, fc, edge_index_, num_nodes, post):
            super().__init__()
            self.in_dim = fc.in_dim
            self.out_dim = fc.out_dim
            self.device = fc.device
            self.num_nodes = num_nodes
            self.register_buffer("edge_index", edge_index_)
            self.gcn = GCNConv(self.in_dim, self.out_dim, bias=True)
            self.post = post
            self._batch_node_idx = None

        def set_batch_node_idx(self, node_idx):
            self._batch_node_idx = node_idx

        def _gcn_edge_index(self, x):
            if x.size(0) == self.num_nodes:
                return self.edge_index
            if self._batch_node_idx is None:
                raise RuntimeError(
                    "GCN mini-batch requires node indices; use n_per_batch=n_obs "
                    "or patch_model_encode_for_gcn()."
                )
            from torch_geometric.utils import subgraph

            node_idx = self._batch_node_idx.to(x.device)
            sub_edge_index, _ = subgraph(
                node_idx,
                self.edge_index,
                relabel_nodes=True,
                num_nodes=self.num_nodes,
            )
            return sub_edge_index

        def forward(self, x, cat_list=None):
            if cat_list is not None:
                raise NotImplementedError(
                    "GCN encoder replacement does not support categorical inputs."
                )
            return self.post(self.gcn(x, self._gcn_edge_index(x)))

        def to(self, device):
            super().to(device)
            self.device = device
            return self

    sae.layers[enc_idx] = GCNEncoderLayer(
        fc_layer,
        edge_index,
        num_nodes=num_nodes,
        post=nn.Sequential(*post_modules),
    )


def patch_model_encode_for_gcn(model) -> None:
    """Track mini-batch node indices so GCN encoder layers can build subgraphs."""
    import torch

    class IndexTrackingLoader:
        def __init__(self, loader):
            self.loader = loader

        def __iter__(self):
            try:
                for batch in self.loader:
                    model._current_batch_indices = batch[0].detach().cpu()
                    yield batch
            finally:
                model._current_batch_indices = None

        def __len__(self):
            return len(self.loader)

    def _set_gcn_batch_indices(node_idx):
        if node_idx is not None:
            node_idx = torch.as_tensor(node_idx, dtype=torch.long)
        for encoder in (model.encoder_ST, model.encoder_SM):
            layer = encoder.layers[0]
            if hasattr(layer, "set_batch_node_idx"):
                layer.set_batch_node_idx(node_idx)

    _orig_as_dataloader = model.as_dataloader

    def as_dataloader_with_index_tracking(*args, **kwargs):
        return IndexTrackingLoader(_orig_as_dataloader(*args, **kwargs))

    model.as_dataloader = as_dataloader_with_index_tracking

    _orig_encode = model.encode

    def encode_with_gcn_indices(X, *args, **kwargs):
        node_idx = None
        if X.size(0) != model._n_record:
            node_idx = getattr(model, "_current_batch_indices", None)
            if node_idx is None:
                raise RuntimeError(
                    "GCN mini-batch encode requires original node indices. "
                    "Call through model.as_dataloader() or use n_per_batch=model._n_record."
                )
            if len(node_idx) != X.size(0):
                raise RuntimeError(
                    f"GCN batch index length ({len(node_idx)}) does not match "
                    f"input batch size ({X.size(0)})."
                )
        _set_gcn_batch_indices(node_idx)
        try:
            return _orig_encode(X, *args, **kwargs)
        finally:
            _set_gcn_batch_indices(None)

    model.encode = encode_with_gcn_indices


#%% load data
#joint_adata = load_joint_adata()

import muon as mu
import sys
sys.path.insert(0, os.path.join(os.getenv("BAKLAVA_ROOT"), "scripts", "SMA"))
from load_aligned_mudata import load_sample

sample_id = "V11L12-109_B1"
joint_mudata = load_sample(sample_id, export_dir=Path(os.path.join(os.getenv("BAKLAVA_BASE_DIR"), "data", "vicari_2023", "h5mu_export")))

# Keep only observations/cells/spots shared across modalities
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

joint_adata = smt.pp.removeHSP_MT_RPL_DNAJ(joint_adata) # remove HSP, MT, RPL, DNAJ features
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

smt.pp.normalize_total_joint_adata_sm_st( # again?
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

# %%
model = smt.model.ConditionalVAESTSM(
    joint_adata,
    device='cuda:0',
    reconstruction_method_sm='g',
    reconstruction_method_st='zinb',
)

graph_conv = True
full_graph = False

if graph_conv:
    if "spatial" not in joint_adata.obsm:
        joint_adata.obsm["spatial"] = joint_mudata.mod["rna"].obsm["spatial"].copy()

    if full_graph:
        edge_index = build_spatial_edge_index(
            joint_adata.obsm["spatial"],
            n_neighbors=6,
            device=str(model.device),
        )
    else:
        edge_index = build_identity_edge_index(
            joint_adata.n_obs,
            device=str(model.device),
        )

    # SAE.layers = ModuleList([FCLayer(in->128), None]) for encode_only stacks=[128]
    num_nodes = joint_adata.n_obs
    replace_fc_encoder_with_gcn(model.encoder_ST, edge_index, num_nodes, layer_idx=0)
    replace_fc_encoder_with_gcn(model.encoder_SM, edge_index, num_nodes, layer_idx=0)
    patch_model_encode_for_gcn(model)
    model.to(model.device)

loss_dict = model.fit(
    max_epoch=250,
    lr=1e-3,
    mode='single',
    n_per_batch=128, #num_nodes if graph_conv else 128,
)

# %%
# CAPTION: Training loss curves for SpatialMETA (ConditionalVAESTSM) over 200 epochs. Each panel shows one tracked loss term (ST/SM reconstruction, correlation branches, KL, MMD). Use to assess convergence and balance between transcriptomics and metabolomics objectives.

fig,axes=plt.subplots(3,3,figsize=(20,10))
axes=axes.flatten()
for ax,(k,v) in zip(axes, loss_dict.items()):
    ax.plot(v)
    ax.set_title(k)

Z = model.get_latent_embedding()
X = model.get_normalized_expression()
C = model.get_modality_contribution()

joint_adata.layers['reconstruction'] = X
joint_adata.obsm['X_emb']=Z
joint_adata.obs['contribution_st']=C
joint_adata.obs['contribution_sm']=1-C

sc.pp.neighbors(
    joint_adata,
    use_rep="X_emb",
    n_neighbors=15
)
sc.tl.umap(
    joint_adata,
    min_dist=1,
    spread=1
)
sc.tl.leiden(
    joint_adata,
    key_added="VAE_clusters_latent10"
)

# %%
# CAPTION: UMAP of the joint ST+SM latent embedding (10-dim VAE, Leiden clusters). Colors: VAE clusters, tissue region, lesion status, and Dopamine (MSI). Shows how anatomy and pathology align with the integrated representation.

sc.pl.umap(
    joint_adata,
    color=["VAE_clusters_latent10", "region", "lesion", "msi:Dopamine"],
    ncols=2,
    size=100,
    wspace=0.3,
    color_map="Reds"
)

# CAPTION: Spatial maps of VAE clusters, region, lesion, and Dopamine on the H&E image. Same variables as UMAP, projected onto aligned Visium coordinates.

sc.pl.spatial(
    joint_adata,
    img_key="hires",
    color=["VAE_clusters_latent10", "region", "lesion", "msi:Dopamine"],
    size=0.075,
    show=False,
    ncols=2,
)

# CAPTION: Observed (normalized) spatial expression of striatal markers Pcp4 and Tac1 (RNA) and Dopamine (MSI) before model reconstruction.

sc.pl.spatial(joint_adata,
              img_key="hires",
              color_map = "vlag",
              color=["rna:Pcp4", "rna:Tac1", "msi:Dopamine"],
              layer="normalized",
              size=0.075,
              wspace=0.005,
              show=False)

# CAPTION: SpatialMETA model reconstruction of Pcp4, Tac1, and Dopamine. Compare with normalized layer to judge reconstruction fidelity and spatial detail retention.

sc.pl.spatial(joint_adata,
              img_key="hires",
              color_map = "vlag",
              color=["rna:Pcp4", "rna:Tac1", "msi:Dopamine"],
              layer="reconstruction",
              size=0.075,
              wspace=0.005,
              show=False)

# CAPTION: Per-spot modality contribution to the joint embedding (teal = ST/transcriptomics, orange = SM/metabolomics). Highlights regions driven more by RNA or MSI signal.

sc.pl.spatial(
    joint_adata,
    img_key="hires",
    color_map = smt.pl.make_colormap(['#2ec4b6','#ffffff','#ff9f1c' ]),
    color=['contribution_st','contribution_sm'],
    layer="normalized",
    wspace=0.005,
    show=False,
    alpha_img=0.1,
    size=0.075
)

obs_df = joint_adata.obs
obs_filter_df = pd.concat([
    obs_df[['VAE_clusters_latent10', 'contribution_st']].rename(columns={'contribution_st': 'contribution'}).assign(type='st'),
    obs_df[['VAE_clusters_latent10', 'contribution_sm']].rename(columns={'contribution_sm': 'contribution'}).assign(type='sm')
])

# CAPTION: Split violin plots of ST vs SM modality contribution within each VAE cluster. Shows whether clusters are RNA-dominated, MSI-dominated, or mixed.

fig,ax = smt.pl.create_fig(
    figsize = (12,4)
)
sns.violinplot(
    data=obs_filter_df,
    x="VAE_clusters_latent10",
    y="contribution",
    hue="type",
    split=True,
    inner="quart",
    palette=['#2ec4b6', '#FFCC70'],
    scale='width',  # Make violins the same width
    bw=0.2,         # Adjust smoothness (lower value = fatter violins)
    cut=0           # Limit the violin to data range
)

plt.xticks(rotation=90)
plt.show()

#%%
#DATA_DIR.mkdir(parents=True, exist_ok=True)
#joint_adata.write_h5ad(JOINT_SPATIALMETA_PATH)
