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

# %% remove HSP, MT, RPL, DNAJ features

joint_adata = smt.pp.removeHSP_MT_RPL_DNAJ(joint_adata)
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

model = smt.model.ConditionalVAESTSM(
    joint_adata,
    device='cuda:0',
    reconstruction_method_sm='g',
    reconstruction_method_st='zinb',
)

# %%

loss_dict = model.fit(
    max_epoch=64,
    lr=1e-3,
    mode='single'
)

# %%

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
# To resume after model training, uncomment:
# joint_adata = sc.read_h5ad(JOINT_SPATIALMETA_PATH)
# %%
sc.pl.umap(
    joint_adata,
    color=["VAE_clusters_latent10", "region", "lesion", "msi:Dopamine"],
    ncols=2,
    size=100,
    wspace=0.3,
    color_map="Reds"
)

sc.pl.spatial(
    joint_adata,
    img_key="hires",
    color=["VAE_clusters_latent10", "region", "lesion", "msi:Dopamine"],
    size=0.075,
    show=False,
    ncols=2,
)

sc.pl.spatial(joint_adata,
              img_key="hires",
              color_map = "vlag",
              color=["rna:Pcp4", "rna:Tac1", "msi:Dopamine"],
              layer="normalized",
              size=0.075,
              wspace=0.005,
              show=False)

sc.pl.spatial(joint_adata,
              img_key="hires",
              color_map = "vlag",
              color=["rna:Pcp4", "rna:Tac1", "msi:Dopamine"],
              layer="reconstruction",
              size=0.075,
              wspace=0.005,
              show=False)

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
