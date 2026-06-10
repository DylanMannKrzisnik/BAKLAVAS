#!/usr/bin/env python
#%%
import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple

# Python 3.7 compatibility for muon/mudata (they use typing.Literal in newer versions)
if sys.version_info < (3, 8):
    import typing
    from typing_extensions import Literal

    typing.Literal = Literal

import scvi

from dotenv import dotenv_values, load_dotenv
ENV_FILE_PATH = "/home/mcb/users/dmannk/BAKLAVA_base/BAKLAVA/.env"
DEFAULT_SOURCE_LABEL_KEY = "RNA_clusters"
DEFAULT_TARGET_LABEL_KEY = "arc_gex_kmeans_5_clusters_Cluster"
SPLIT_RATIO_TRAIN = 0.5
SPLIT_RATIO_VAL = 0.3
SPLIT_RATIO_TEST = 0.2
SPLIT_ARTIFACT_PATH = os.path.join("splits", "domain_splits.json")
SPLIT_SCHEMA_VERSION = 1

BASE_PATH = None
REPO_ROOT = None
MultiGATE = None
MultiGATETrainer = None


def bootstrap_runtime():
    global BASE_PATH, REPO_ROOT, MultiGATE, MultiGATETrainer

    load_dotenv(dotenv_path=ENV_FILE_PATH)
    print("Loaded environment variables from .env or env:", end="\n\n")
    pprint(dotenv_values(ENV_FILE_PATH))

    datapath = os.getenv("DATAPATH")
    if datapath is None:
        raise EnvironmentError(
            "DATAPATH is not set. Export DATAPATH to the base data directory, e.g. "
            "'/home/mcb/users/dmannk/BAKLAVA_base/data'."
        )

    baklava_base_dir = os.getenv("BAKLAVA_BASE_DIR")
    if baklava_base_dir is None:
        raise EnvironmentError(
            "BAKLAVA_BASE_DIR is not set. Export BAKLAVA_BASE_DIR to the base repo directory, e.g. "
            "'/home/mcb/users/dmannk/BAKLAVA_base'."
        )

    REPO_ROOT = os.path.join(baklava_base_dir, "MultiGATE")
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    baklava_repo_root = os.path.join(baklava_base_dir, "BAKLAVA")
    if os.path.isdir(baklava_repo_root) and baklava_repo_root not in sys.path:
        sys.path.insert(0, baklava_repo_root)

    BASE_PATH = os.path.join(datapath, "aligned_data")

    import MultiGATE
    import scvi


import matplotlib.pyplot as plt
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from anndata import AnnData
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import tempfile
import pickle

import warnings
warnings.filterwarnings("ignore")

@dataclass
class DomainData:
    rna: AnnData
    atac: AnnData
    label_key: Optional[str] = None


@dataclass
class DomainSplitBundle:
    full: DomainData
    train: DomainData
    val: DomainData
    test: DomainData
    eval: DomainData
    split_indices: Dict[str, np.ndarray] = field(default_factory=dict)
    split_obs_names: Dict[str, np.ndarray] = field(default_factory=dict)
    split_seed: Optional[int] = None

    @property
    def rna(self):
        return self.train.rna

    @property
    def atac(self):
        return self.train.atac

    @property
    def label_key(self):
        return self.train.label_key


@dataclass
class DataBundle:
    source: DomainSplitBundle
    target: DomainSplitBundle
    split_metadata: Dict[str, Any] = field(default_factory=dict)
    # Optional NicheCompass prior pathways used to build fixed rho pathway masks.
    combined_gp_dict: Optional[Dict[str, Any]] = None


@dataclass
class GraphInputBundle:
    graph_tf: Tuple[Any, Any, Any]
    gp_tf: Tuple[Any, Any, Any]
    x1: pd.DataFrame
    x2: pd.DataFrame

    def __repr__(self) -> str:
        def _sparse_tf_summary(tf: Tuple[Any, Any, Any]) -> str:
            if tf is None or len(tf) < 3:
                return "invalid"
            shape = tf[2]
            n_edges = int(getattr(tf[0], "shape", [0])[0]) if tf[0] is not None else 0
            return "edges={}, shape={}".format(n_edges, shape)

        return (
            "GraphInputBundle(graph_tf=({}), gp_tf=({}), x1.shape={}, x2.shape={})".format(
                _sparse_tf_summary(self.graph_tf),
                _sparse_tf_summary(self.gp_tf),
                tuple(self.x1.shape),
                tuple(self.x2.shape),
            )
        )


@dataclass
class GraphBundle:
    source: GraphInputBundle
    target: GraphInputBundle
    bp_width: int = 400
    graph_type: str = "ATAC"
    protein_value: float = 0.001

    def __repr__(self) -> str:
        return (
            "GraphBundle(bp_width={}, graph_type={}, protein_value={}, source={}, target={})".format(
                self.bp_width,
                self.graph_type,
                self.protein_value,
                self.source,
                self.target,
            )
        )


@dataclass
class PathwayDecoderMaskBundle:
    pathway_names: np.ndarray
    source_pathway_names: np.ndarray
    target_pathway_names: np.ndarray
    rho_rna_mask: np.ndarray
    rho_atac_mask: np.ndarray
    n_zero_source_pathways: int = 0
    n_zero_target_pathways: int = 0


@dataclass
class Stage1CacheConfig:
    use_cache: bool
    run_name: Optional[str] = None
    run_id: Optional[str] = None
    run_params: Dict[str, Any] = field(default_factory=dict)
    dual_source_kd: bool = False
    student_graph_type: str = "identity"
    vgp_anchor_mode: str = "feature"
    skip_gp_attention: bool = True


@dataclass
class Stage1TrainerBundle:
    teacher: Any
    student: Optional[Any]
    nonspatial: Optional[Any]
    source_inputs_tensors: Optional[Tuple[Any, Any, Any, Any, Any]]
    source_student_inputs_tensors: Optional[Tuple[Any, Any, Any, Any, Any]]
    source_student_graph_tf: Optional[Tuple[Any, Any, Any]]
    primary: Any
    primary_model_name: str
    primary_source_graph_tf: Tuple[Any, Any, Any]


def parse_args(notebook: bool = False):
    parser = argparse.ArgumentParser(description="Train MultiGATE on source and run live zero-shot eval on target.")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Random seed used to generate deterministic 70/20/10 train/val/test splits for both domains.",
    )
    parser.add_argument(
        "--source-split-train-eval",
        action="store_true",
        default=False,
        help=(
            "If set, use source train/eval split subsets (train vs val+test). "
            "By default, source train/eval both use the full source dataset while target remains split."
        ),
    )
    parser.add_argument(
        "--stage1-epochs",
        type=int,
        default=500,
        help="Number of epochs to train the model for stage 1.",
    )
    parser.add_argument(
        "--top-n-genes",
        type=int,
        default=0,
        help="Number of top genes to keep for filtering. If 0, use SCT genes.",
    )
    parser.add_argument(
        "--top-n-peaks",
        type=int,
        default=0,
        help="Number of top in-cis peaks to keep for filtering. If 0, use peaks linked to SCT genes.",
    )
    parser.add_argument(
        "--log-mudata-umaps",
        action="store_true",
        default=False,
        help="If set, log source/target MuData UMAP artifacts in addition to concat AnnData UMAPs.",
    )
    parser.add_argument(
        "--source-label-key",
        type=str,
        default=None,
        help="Optional source label key in .obs for scib metrics. Falls back to pseudo labels if missing.",
    )
    parser.add_argument(
        "--target-label-key",
        type=str,
        default=None,
        help="Optional target label key in .obs for scib metrics. Falls back to pseudo labels if missing.",
    )
    parser.add_argument(
        "--scib-n-jobs",
        type=int,
        default=1,
        help="Number of jobs for scib-metrics neighbor search.",
    )
    parser.add_argument(
        "--spatial-graph-type",
        type=str,
        choices=["spatial", "knn", "identity", "tangram"],
        default="identity",
        help="Type of graph to use for MultiGATE.",
    )
    if notebook:
        return parser.parse_known_args()[0]
    else:
        return parser.parse_args()


def _to_dense_df(adata):
    if isinstance(adata.X, np.ndarray):
        matrix = adata.X
    else:
        matrix = adata.X.toarray()
    return pd.DataFrame(matrix, index=adata.obs.index, columns=adata.var.index)


def prepare_graph_data(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return (indices, adj.data, adj.shape)


def build_graph_inputs(adata_vars1, adata_vars2, bp_width=450, graph_type="ATAC", protein_value=0.001):

    x1 = _to_dense_df(adata_vars1)
    x2 = _to_dense_df(adata_vars2)

    cells = np.array(x1.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))

    genes = np.array(x1.columns)
    peaks = np.array(x2.columns)
    genes_id_tran = dict(zip(genes, range(genes.shape[0])))
    peaks_id_tran = dict(zip(peaks, range(peaks.shape[0])))

    if "Spatial_Net" not in adata_vars1.uns:
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    spatial_net = adata_vars1.uns["Spatial_Net"]
    graph_df = spatial_net.copy()
    graph_df["Cell1"] = graph_df["Cell1"].map(cells_id_tran)
    graph_df["Cell2"] = graph_df["Cell2"].map(cells_id_tran)
    graph_df = graph_df.dropna(subset=["Cell1", "Cell2"])
    graph_df[["Cell1", "Cell2"]] = graph_df[["Cell1", "Cell2"]].astype(int)

    graph = sp.coo_matrix(
        (np.ones(graph_df.shape[0]), (graph_df["Cell1"], graph_df["Cell2"])),
        shape=(adata_vars1.n_obs, adata_vars1.n_obs),
    )
    graph_tf = prepare_graph_data(graph)

    if "gene_peak_Net" not in adata_vars1.uns:
        raise ValueError("gene_peak_Net is not existed! Run Cal_gene_peak_Net first!")

    gene_peak_net = adata_vars1.uns["gene_peak_Net"]
    if graph_type == "protein":
        gene_peak_net = gene_peak_net.copy()
        gene_peak_net.columns = ["Gene", "Peak"]

    gp_df = gene_peak_net.copy()
    gp_df["Gene"] = gp_df["Gene"].map(genes_id_tran)
    gp_df["Peak"] = gp_df["Peak"].map(peaks_id_tran)
    gp_df = gp_df.dropna(subset=["Gene", "Peak"]).copy()
    if gp_df.empty:
        raise ValueError("gene_peak_Net does not overlap with selected RNA/ATAC features.")

    gp_df["Gene"] = gp_df["Gene"].astype(int)
    gp_df["Peak"] = gp_df["Peak"].astype(int) + adata_vars1.n_vars

    if graph_type in ["ATAC", "ATAC_RNA"]:
        dist = gp_df["Distance"].astype(float)
        gp_bp_width = bp_width if graph_type == "ATAC" else 2000
        weights = np.concatenate(
            (
                ((dist + gp_bp_width) / gp_bp_width) ** (-0.75),
                ((dist + gp_bp_width) / gp_bp_width) ** (-0.75),
            ),
            axis=0,
        )
    else:
        weights = np.ones(gp_df.shape[0] * 2) * protein_value

    gp_graph = sp.coo_matrix(
        (
            weights,
            (
                np.concatenate((gp_df["Gene"], gp_df["Peak"]), axis=0),
                np.concatenate((gp_df["Peak"], gp_df["Gene"]), axis=0),
            ),
        ),
        shape=(adata_vars1.n_vars + adata_vars2.n_vars, adata_vars1.n_vars + adata_vars2.n_vars),
    )
    gp_graph_tf = prepare_graph_data(gp_graph)

    return graph_tf, gp_graph_tf, x1, x2


def build_knn_graph_as_spatial_net(adata, n_neighbors=15):
    # Build a generic kNN cell graph for non-spatial data and store it in the
    # format expected by MultiGATE.forward_MultiGATE (adata.uns['Spatial_Net']).
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    conn = adata.obsp["connectivities"].tocoo()
    mask = conn.row != conn.col
    adata.uns["Spatial_Net"] = pd.DataFrame(
        {
            "Cell1": adata.obs_names[conn.row[mask]].to_numpy(),
            "Cell2": adata.obs_names[conn.col[mask]].to_numpy(),
            "Distance": np.zeros(int(mask.sum()), dtype=float),
        }
    )


def build_graph_tf_from_spatial_net(spatial_net, obs_names):
    if not {"Cell1", "Cell2"}.issubset(set(spatial_net.columns)):
        raise ValueError("Spatial_Net must contain columns {'Cell1', 'Cell2'}.")

    cells = np.asarray(obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))

    graph_df = spatial_net.copy()
    graph_df["Cell1"] = graph_df["Cell1"].map(cells_id_tran)
    graph_df["Cell2"] = graph_df["Cell2"].map(cells_id_tran)
    graph_df = graph_df.dropna(subset=["Cell1", "Cell2"])

    if graph_df.empty:
        graph = sp.coo_matrix((len(cells), len(cells)))
        return prepare_graph_data(graph)

    graph_df[["Cell1", "Cell2"]] = graph_df[["Cell1", "Cell2"]].astype(int)
    graph = sp.coo_matrix(
        (np.ones(graph_df.shape[0]), (graph_df["Cell1"], graph_df["Cell2"])),
        shape=(len(cells), len(cells)),
    )
    return prepare_graph_data(graph)


def build_source_student_graph_tf(source_rna, spatial_graph_type, knn_neighbors=15):
    if spatial_graph_type == "spatial":
        if "Spatial_Net" not in source_rna.uns:
            raise ValueError("Source Spatial_Net is missing for stage-1 dual-source KD.")
        return build_graph_tf_from_spatial_net(source_rna.uns["Spatial_Net"], source_rna.obs_names)

    if spatial_graph_type == "knn":
        # Build kNN edges on source cells without mutating the main source AnnData.
        tmp_adata = AnnData(X=source_rna.X, obs=source_rna.obs.copy())
        build_knn_graph_as_spatial_net(tmp_adata, n_neighbors=knn_neighbors)
        return build_graph_tf_from_spatial_net(tmp_adata.uns["Spatial_Net"], source_rna.obs_names)

    if spatial_graph_type == "identity":
        identity_net = pd.DataFrame(columns=["Cell1", "Cell2", "Distance"])
        return build_graph_tf_from_spatial_net(identity_net, source_rna.obs_names)

    raise ValueError(
        "Unsupported --spatial-graph-type '{}' for stage-1 source student graph.".format(spatial_graph_type)
    )

# Legacy helper kept for compatibility with older co-embed runs that do not
# have split artifacts.
def pair_and_subsample_target(target_rna, target_atac, subsample_n, seed):
    shared_obs = target_rna.obs_names.intersection(target_atac.obs_names)
    if len(shared_obs) == 0:
        raise ValueError("Target RNA/ATAC share zero cells after preprocessing.")

    target_rna = target_rna[shared_obs].copy()
    target_atac = target_atac[shared_obs].copy()

    if target_rna.n_obs > subsample_n:
        rng = np.random.RandomState(seed)
        selected = np.array(target_rna.obs_names)[
            rng.choice(target_rna.n_obs, size=subsample_n, replace=False)
        ]
        target_rna = target_rna[selected].copy()
        target_atac = target_atac[selected].copy()

    return target_rna, target_atac


def pair_modalities(rna, atac, domain_name):
    shared_obs = rna.obs_names.intersection(atac.obs_names)
    if len(shared_obs) == 0:
        raise ValueError("{} RNA/ATAC share zero cells after preprocessing.".format(domain_name))
    return rna[shared_obs].copy(), atac[shared_obs].copy()


def _build_split_indices(n_obs, seed, domain_name):
    if n_obs <= 0:
        raise ValueError("{} has zero observations and cannot be split.".format(domain_name))

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_obs)
    train_n = int(np.floor(SPLIT_RATIO_TRAIN * n_obs))
    val_n = int(np.floor(SPLIT_RATIO_VAL * n_obs))
    test_n = int(n_obs - train_n - val_n)
    if train_n <= 0 or val_n <= 0 or test_n <= 0:
        raise ValueError(
            "{} split would create an empty subset with n_obs={} (train={}, val={}, test={}).".format(
                domain_name,
                n_obs,
                train_n,
                val_n,
                test_n,
            )
        )

    idx_train = perm[:train_n]
    idx_val = perm[train_n:train_n + val_n]
    idx_test = perm[train_n + val_n:]
    idx_eval = np.concatenate([idx_val, idx_test], axis=0)
    return {
        "train": idx_train,
        "val": idx_val,
        "test": idx_test,
        "eval": idx_eval,
    }


def _subset_domain(rna, atac, obs_names, label_key):
    sub_rna = rna[obs_names].copy()
    sub_atac = atac[obs_names].copy()
    sub_rna.uns["label_key"] = label_key
    return DomainData(rna=sub_rna, atac=sub_atac, label_key=label_key)


def build_domain_split_bundle(rna, atac, label_key, split_seed, domain_name):
    split_indices = _build_split_indices(rna.n_obs, split_seed, domain_name)
    split_obs_names = {
        split_name: np.asarray(rna.obs_names)[indices]
        for split_name, indices in split_indices.items()
    }
    full_domain = _subset_domain(rna, atac, rna.obs_names, label_key)
    train_domain = _subset_domain(rna, atac, split_obs_names["train"], label_key)
    val_domain = _subset_domain(rna, atac, split_obs_names["val"], label_key)
    test_domain = _subset_domain(rna, atac, split_obs_names["test"], label_key)
    eval_domain = _subset_domain(rna, atac, split_obs_names["eval"], label_key)

    return DomainSplitBundle(
        full=full_domain,
        train=train_domain,
        val=val_domain,
        test=test_domain,
        eval=eval_domain,
        split_indices=split_indices,
        split_obs_names=split_obs_names,
        split_seed=split_seed,
    )


def configure_source_train_eval_bundle(source_split_bundle, use_source_split_train_eval):
    if use_source_split_train_eval:
        return source_split_bundle

    full_obs_names = np.asarray(source_split_bundle.full.rna.obs_names)
    full_indices = np.arange(source_split_bundle.full.rna.n_obs)
    source_split_bundle.train = source_split_bundle.full
    source_split_bundle.eval = source_split_bundle.full
    source_split_bundle.split_obs_names["train"] = full_obs_names
    source_split_bundle.split_obs_names["eval"] = full_obs_names
    source_split_bundle.split_indices["train"] = full_indices
    source_split_bundle.split_indices["eval"] = full_indices
    return source_split_bundle


def build_split_metadata(source_split_bundle, target_split_bundle):
    def _domain_payload(bundle):
        return {
            "n_obs": int(bundle.full.rna.n_obs),
            "seed": int(bundle.split_seed),
            "splits": {
                split_name: {
                    "indices": [int(v) for v in bundle.split_indices[split_name].tolist()],
                    "obs_names": [str(v) for v in bundle.split_obs_names[split_name].tolist()],
                    "n_obs": int(len(bundle.split_indices[split_name])),
                }
                for split_name in ("train", "val", "test", "eval")
            },
        }

    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "evaluation_split": "val_plus_test",
        "ratios": {
            "train": SPLIT_RATIO_TRAIN,
            "val": SPLIT_RATIO_VAL,
            "test": SPLIT_RATIO_TEST,
        },
        "domains": {
            "source": _domain_payload(source_split_bundle),
            "target": _domain_payload(target_split_bundle),
        },
    }


def apply_hvg_and_gp_filtering(
    source_rna,
    source_atac,
    target_rna,
    target_atac,
    gp_net,
    top_n_genes,
    top_n_peaks,
    rank_type="fused",
):
    """Shared source/target feature filtering logic used by training and co-embed scripts."""
    assert rank_type in ["fused", "source", "target"], "rank_type must be 'fused' or 'source' or 'target'"

    gp_net_genes = gp_net["Gene"].unique()
    gp_net_peaks = gp_net["Peak"].unique()

    source_rna = source_rna[:, source_rna.var_names.isin(gp_net_genes)].copy()
    source_atac = source_atac[:, source_atac.var_names.isin(gp_net_peaks)].copy()
    target_rna = target_rna[:, target_rna.var_names.isin(gp_net_genes)].copy()
    target_atac = target_atac[:, target_atac.var_names.isin(gp_net_peaks)].copy()

    source_rna.var["highly_variable"] = False
    source_atac.var["highly_variable"] = False
    target_rna.var["highly_variable"] = False
    target_atac.var["highly_variable"] = False

    source_rna.var["highly_variable_rank"] = source_rna.var["dispersions_norm"].rank(ascending=False)
    target_rna.var["highly_variable_rank"] = target_rna.var["dispersions_norm"].rank(ascending=False)
    source_atac.var["highly_variable_rank"] = source_atac.var["dispersions_norm"].rank(ascending=False)
    target_atac.var["highly_variable_rank"] = target_atac.var["dispersions_norm"].rank(ascending=False)

    # Compute combined rank. Note that may have more than n_top_genes/peaks due to rank ties.
    def _rank_fused(source_adata, target_adata, local_rank_type):
        if local_rank_type == "fused":
            return pd.concat(
                [
                    source_adata.var["highly_variable_rank"],
                    target_adata.var["highly_variable_rank"],
                ],
                axis=1,
            ).mean(axis=1).rank(ascending=True, method="min")
        elif local_rank_type == "source":
            return source_adata.var["highly_variable_rank"]
        elif local_rank_type == "target":
            return target_adata.var["highly_variable_rank"]

    # Compute combined rank for genes.
    rna_combined_rank = _rank_fused(source_rna, target_rna, rank_type)
    gene_filt = rna_combined_rank.le(top_n_genes)
    source_rna.var.loc[gene_filt, "highly_variable"] = True
    target_rna.var.loc[gene_filt, "highly_variable"] = True

    # Filter peaks in-cis with filtered genes.
    peak_filt = gp_net.loc[
        gp_net["Gene"].isin(gene_filt.loc[gene_filt].index),
        "Peak",
    ].unique()

    # Compute combined rank for peaks.
    atac_combined_rank = _rank_fused(source_atac, target_atac, rank_type)
    atac_combined_rank_filt = atac_combined_rank.loc[source_atac.var_names.isin(peak_filt)]
    atac_combined_rank_filt = atac_combined_rank_filt.rank(ascending=True, method="min")
    peak_filt = atac_combined_rank_filt.le(top_n_peaks)
    peak_filt = peak_filt.loc[peak_filt].index

    source_atac.var.loc[source_atac.var_names.isin(peak_filt), "highly_variable"] = True
    target_atac.var.loc[target_atac.var_names.isin(peak_filt), "highly_variable"] = True

    # Re-introduce gp-net based on filtered genes and peaks.
    gp_net = gp_net[
        gp_net["Gene"].isin(gene_filt.loc[gene_filt].index)
        & gp_net["Peak"].isin(peak_filt)
    ]
    source_rna.uns["gene_peak_Net"] = gp_net.copy()
    target_rna.uns["gene_peak_Net"] = gp_net.copy()

    return source_rna, source_atac, target_rna, target_atac, gp_net

def get_sct_genes_and_gp_filtering(
    source_rna,
    source_atac,
    target_rna,
    target_atac,
    gp_net
    ):

    gp_net_genes = gp_net["Gene"].unique()
    gp_net_peaks = gp_net["Peak"].unique()

    source_rna = source_rna[:, source_rna.var_names.isin(gp_net_genes)].copy()
    source_atac = source_atac[:, source_atac.var_names.isin(gp_net_peaks)].copy()
    target_rna = target_rna[:, target_rna.var_names.isin(gp_net_genes)].copy()
    target_atac = target_atac[:, target_atac.var_names.isin(gp_net_peaks)].copy()

    source_rna.var["highly_variable"] = False
    source_atac.var["highly_variable"] = False
    target_rna.var["highly_variable"] = False
    target_atac.var["highly_variable"] = False

    where_overlap_sct_genes = source_rna.var['SCT_gene'] & target_rna.var['SCT_gene']
    overlap_sct_genes = where_overlap_sct_genes.index[where_overlap_sct_genes]

    source_rna.var.loc[overlap_sct_genes, "highly_variable"] = True
    target_rna.var.loc[overlap_sct_genes, "highly_variable"] = True

    peak_filt = gp_net.loc[
        gp_net["Gene"].isin(overlap_sct_genes),
        "Peak",
    ].unique()
    source_atac.var.loc[source_atac.var_names.isin(peak_filt), "highly_variable"] = True
    target_atac.var.loc[target_atac.var_names.isin(peak_filt), "highly_variable"] = True

    gp_net = gp_net[
        gp_net["Gene"].isin(overlap_sct_genes)
        & gp_net["Peak"].isin(peak_filt)
    ]
    source_rna.uns["gene_peak_Net"] = gp_net.copy()
    target_rna.uns["gene_peak_Net"] = gp_net.copy()

    return source_rna, source_atac, target_rna, target_atac, gp_net


def prepare_target_for_spatial_graph_type(
    target_rna,
    target_atac,
    source_rna,
    source_atac,
    spatial_graph_type,
    gtf_path,
):
    """Shared target graph preparation used by training and co-embed scripts."""
    if spatial_graph_type == "spatial":
        MultiGATE.Cal_Spatial_Net(target_rna, rad_cutoff=40)
        MultiGATE.Stats_Spatial_Net(target_rna)
        MultiGATE.Cal_Spatial_Net(target_atac, rad_cutoff=40)
        MultiGATE.Stats_Spatial_Net(target_atac)
        target_rna = target_rna[:, target_rna.var["highly_variable"]].copy()
        target_atac = target_atac[:, target_atac.var["highly_variable"]].copy()
        MultiGATE.Cal_gene_peak_Net_new(target_rna, target_atac, 150000, file=gtf_path)
        target_rna.uns["gene_peak_Net"] = target_atac.uns["gene_peak_Net"]

    elif spatial_graph_type == "tangram":
        target_rna = target_rna[:, target_rna.var_names.isin(source_rna.var_names)].copy()
        target_atac = target_atac[:, target_atac.var_names.isin(source_atac.var_names)].copy()
        tangram_net = pd.read_csv(os.path.join(os.getenv("OUTPATH"), "tangram", "tangram_spatial_net_affinity.csv"))
        target_rna.uns["Spatial_Net"] = tangram_net.copy()
        target_atac.uns["Spatial_Net"] = tangram_net.copy()
        target_rna.uns["gene_peak_Net"] = source_rna.uns["gene_peak_Net"].copy()
        target_atac.uns["gene_peak_Net"] = source_atac.uns["gene_peak_Net"].copy()

    elif spatial_graph_type == "knn":
        target_rna = target_rna[:, target_rna.var["highly_variable"]].copy()
        target_atac = target_atac[:, target_atac.var["highly_variable"]].copy()
        target_rna.uns["gene_peak_Net"] = source_rna.uns["gene_peak_Net"]
        target_atac.uns["gene_peak_Net"] = source_rna.uns["gene_peak_Net"]
        build_knn_graph_as_spatial_net(target_rna, n_neighbors=15)
        target_atac.uns["Spatial_Net"] = target_rna.uns["Spatial_Net"].copy()
        MultiGATE.Stats_Spatial_Net(target_rna)
        MultiGATE.Stats_Spatial_Net(target_atac)

    elif spatial_graph_type == "identity":
        target_rna = target_rna[:, target_rna.var["highly_variable"]].copy()
        target_atac = target_atac[:, target_atac.var["highly_variable"]].copy()
        target_rna.uns["gene_peak_Net"] = source_rna.uns["gene_peak_Net"]
        target_atac.uns["gene_peak_Net"] = source_rna.uns["gene_peak_Net"]
        target_rna.uns["Spatial_Net"] = pd.DataFrame(columns=["Cell1", "Cell2", "Distance"])
        target_atac.uns["Spatial_Net"] = target_rna.uns["Spatial_Net"].copy()
    else:
        raise ValueError("Unknown spatial_graph_type '{}'".format(spatial_graph_type))

    return target_rna, target_atac


_SCIB_BACKEND = None


def require_scib_backend():
    global _SCIB_BACKEND
    if _SCIB_BACKEND is not None:
        return _SCIB_BACKEND

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ.setdefault("JAX_DISABLE_JIT", os.environ.get("BAKLAVA_JAX_DISABLE_JIT", "1"))

    try:
        from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
    except Exception as exc:
        raise ImportError(
            "Failed to import scib-metrics. Install missing dependencies in MultiGATEenv_py310_scib, "
            "e.g. `pip install chex scib-metrics` (or conda equivalents)."
        ) from exc

    _SCIB_BACKEND = {
        "Benchmarker": Benchmarker,
        "BioConservation": BioConservation,
        "BatchCorrection": BatchCorrection,
    }
    return _SCIB_BACKEND


def resolve_scib_labels(rna_adata, atac_adata, concat_adata, label_key, domain_name):
    if label_key is not None:
        if label_key in rna_adata.obs.columns and label_key in atac_adata.obs.columns:
            if label_key not in concat_adata.obs.columns:
                raise KeyError(
                    "Label key '{}' missing in concatenated obs for {} domain.".format(label_key, domain_name)
                )
            return label_key, "provided"
        warnings.warn(
            "Requested label key '{}' for {} not found in both RNA and ATAC obs. "
            "Falling back to pseudo labels.".format(label_key, domain_name)
        )

    sc.pp.neighbors(concat_adata, use_rep="X", n_neighbors=15, key_added="scib_eval")
    sc.tl.leiden(
        concat_adata,
        neighbors_key="scib_eval",
        key_added="scib_pseudo_leiden",
        resolution=1.5,
        random_state=0,
    )
    return "scib_pseudo_leiden", "pseudo_leiden"


def compute_scib_metrics_for_domain(
    rna_adata,
    atac_adata,
    domain_name,
    label_key=None,
    scib_n_jobs=1,
    embedding_key="MultiGATE",
):
    scib_backend = require_scib_backend()
    Benchmarker = scib_backend["Benchmarker"]
    BioConservation = scib_backend["BioConservation"]
    BatchCorrection = scib_backend["BatchCorrection"]

    concat_adata = build_concat_adata_for_umap(rna_adata, atac_adata, embedding_key=embedding_key)
    effective_label_key, label_mode = resolve_scib_labels(
        rna_adata=rna_adata,
        atac_adata=atac_adata,
        concat_adata=concat_adata,
        label_key=label_key,
        domain_name=domain_name,
    )

    concat_adata.obsm["multigate_latent"] = np.asarray(concat_adata.X)

    benchmarker = Benchmarker(
        adata=concat_adata,
        batch_key="modality",
        label_key=effective_label_key,
        embedding_obsm_keys=["multigate_latent"],
        bio_conservation_metrics=BioConservation(
            isolated_labels=False,
            nmi_ari_cluster_labels_leiden=True,
            nmi_ari_cluster_labels_kmeans=False,
            silhouette_label=True,
            clisi_knn=False,
        ),
        batch_correction_metrics=BatchCorrection(
            bras=False,
            ilisi_knn=True,
            kbet_per_label=False,
            graph_connectivity=False,
            pcr_comparison=False,
        ),
        pre_integrated_embedding_obsm_key="multigate_latent",
        n_jobs=scib_n_jobs,
        progress_bar=False,
    )
    benchmarker.benchmark()

    results = benchmarker.get_results(min_max_scale=False, clean_names=False)
    if "multigate_latent" not in results.index:
        raise KeyError("scib results missing expected embedding row 'multigate_latent'.")
    row = results.loc["multigate_latent"]

    metrics = {
        "label_mode": label_mode,
        "effective_label_key": effective_label_key,
    }

    if "silhouette_label" in row.index:
        metrics["silhouette_label"] = float(row["silhouette_label"])
    if "ilisi_knn" in row.index:
        metrics["ilisi"] = float(row["ilisi_knn"])
    if "bras" in row.index:
        metrics["bras"] = float(row["bras"])
    if "Bio conservation" in row.index:
        metrics["bio_conservation"] = float(row["Bio conservation"])
    if "Batch correction" in row.index:
        metrics["batch_correction"] = float(row["Batch correction"])
    if "Total" in row.index:
        metrics["total"] = float(row["Total"])

    return metrics


def log_scib_metrics(prefix, metrics, step):
    mapping = {
        "silhouette_label": "{}_scib_silhouette_label".format(prefix),
        "ilisi": "{}_scib_ilisi".format(prefix),
        "bras": "{}_scib_bras".format(prefix),
        "bio_conservation": "{}_scib_bio_conservation".format(prefix),
        "batch_correction": "{}_scib_batch_correction".format(prefix),
        "total": "{}_scib_total".format(prefix),
    }
    for key, metric_name in mapping.items():
        if key in metrics and np.isfinite(metrics[key]):
            mlflow.log_metric(metric_name, float(metrics[key]), step=step)


def log_umap_to_mlflow(mdata, artifact_path, title, color_key="wnn", size=20):
    if mdata.n_obs < 3:
        warnings.warn("Skipping UMAP artifact '{}' because n_obs < 3.".format(artifact_path))
        return
    if "X_umap" not in mdata.obsm:
        warnings.warn("Skipping UMAP artifact '{}' because X_umap is missing.".format(artifact_path))
        return

    umap_fig = None
    try:
        umap_fig, umap_ax = plt.subplots(figsize=(7, 5))
        plot_fn = mu.pl.umap if isinstance(mdata, mu.MuData) else sc.pl.umap
        plot_fn(mdata, color=color_key, title=title, ax=umap_ax, size=size, show=False)
        umap_fig.tight_layout()
        mlflow.log_figure(umap_fig, artifact_path)
    finally:
        if umap_fig is not None:
            plt.close(umap_fig)


def build_concat_adata_for_umap(rna_adata, atac_adata, embedding_key="MultiGATE"):
    if embedding_key not in rna_adata.obsm or embedding_key not in atac_adata.obsm:
        raise KeyError(
            "Missing '{}' in one or both modalities when building concat AnnData.".format(embedding_key)
        )

    rna_obs = rna_adata.obs.copy()
    atac_obs = atac_adata.obs.copy()
    rna_obs["modality"] = "rna"
    atac_obs["modality"] = "atac"
    rna_obs.index = rna_obs.index.astype(str) + "_rna"
    atac_obs.index = atac_obs.index.astype(str) + "_atac"

    concat_adata = AnnData(
        X=np.concatenate(
            [
                rna_adata.obsm[embedding_key],
                atac_adata.obsm[embedding_key],
            ],
            axis=0,
        ),
        obs=pd.concat([rna_obs, atac_obs], axis=0),
    )
    return concat_adata


def compute_concat_umap(
    concat_adata,
    n_neighbors=10,
    resolution=1.5,
    deterministic=False,
    random_state=0,
):
    if deterministic:
        sc.pp.neighbors(
            concat_adata,
            n_neighbors=n_neighbors,
            use_rep="X",
            knn=True,
            method="umap",
            metric="euclidean",
            random_state=random_state,
        )
        sc.tl.umap(concat_adata, random_state=random_state, init_pos="spectral")
        sc.tl.leiden(concat_adata, resolution=resolution, random_state=random_state)
    else:
        sc.pp.neighbors(concat_adata, n_neighbors=n_neighbors)
        sc.tl.umap(concat_adata)
        sc.tl.leiden(concat_adata, resolution=resolution)


def log_umap_panel_to_mlflow(adata, artifact_path, colors, titles, size=20):
    if adata.n_obs < 3:
        warnings.warn("Skipping UMAP artifact '{}' because n_obs < 3.".format(artifact_path))
        return
    if "X_umap" not in adata.obsm:
        warnings.warn("Skipping UMAP artifact '{}' because X_umap is missing.".format(artifact_path))
        return

    fig = None
    try:
        n_panels = len(colors)
        fig, axs = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
        if n_panels == 1:
            axs = [axs]
        for idx, (color_key, title) in enumerate(zip(colors, titles)):
            sc.pl.umap(
                adata,
                color=color_key,
                title=title,
                ax=axs[idx],
                size=size,
                show=False,
            )
        fig.tight_layout()
        mlflow.log_figure(fig, artifact_path)
    finally:
        if fig is not None:
            plt.close(fig)


def build_mudata_with_umap(rna_adata, atac_adata, embedding_key="MultiGATE", n_neighbors=10, resolution=1.5):
    if embedding_key not in rna_adata.obsm or embedding_key not in atac_adata.obsm:
        raise KeyError(
            "Missing '{}' in one or both modalities when building MuData.".format(embedding_key)
        )

    rna_eval = rna_adata.copy()
    atac_eval = atac_adata.copy()
    sc.pp.neighbors(rna_eval, use_rep=embedding_key, n_neighbors=n_neighbors)
    sc.pp.neighbors(atac_eval, use_rep=embedding_key, n_neighbors=n_neighbors)

    mdata = mu.MuData({"rna": rna_eval, "atac": atac_eval})
    mu.pp.intersect_obs(mdata)
    mu.pp.neighbors(mdata, n_neighbors=n_neighbors)
    mu.tl.umap(mdata)
    sc.tl.leiden(mdata, resolution=resolution)
    mdata.obs["wnn"] = mdata.obs["leiden"].astype(int).astype("category")
    return mdata


def log_stage_umap_artifacts(
    source_rna,
    source_atac,
    target_rna,
    target_atac,
    stage_label,
    log_mudata_umaps=False,
    embedding_key="MultiGATE",
):
    source_concat_adata = build_concat_adata_for_umap(source_rna, source_atac, embedding_key=embedding_key)
    target_concat_adata = build_concat_adata_for_umap(target_rna, target_atac, embedding_key=embedding_key)

    compute_concat_umap(source_concat_adata, n_neighbors=10, resolution=1.5)
    compute_concat_umap(target_concat_adata, n_neighbors=10, resolution=1.5)

    source_celltype_key = source_rna.uns['label_key']
    target_celltype_key = target_rna.uns['label_key']

    source_colors = ["modality", "leiden"]
    source_titles = [
        "Source Concat Modality ({})".format(stage_label),
        "Source Concat Leiden ({})".format(stage_label),
    ]
    if source_celltype_key is not None:
        source_colors.append(source_celltype_key)
        source_titles.append("Source Concat Cell Type ({})".format(stage_label))

    log_umap_panel_to_mlflow(
        source_concat_adata,
        artifact_path="umap/{}/source_concat_adata_umap.png".format(stage_label),
        colors=source_colors,
        titles=source_titles,
        size=20,
    )

    target_colors = ["modality", "leiden"]
    target_titles = [
        "Target Concat Modality ({})".format(stage_label),
        "Target Concat Leiden ({})".format(stage_label),
    ]
    if target_celltype_key is not None:
        target_colors.append(target_celltype_key)
        target_titles.append("Target Concat Cell Type ({})".format(stage_label))

    log_umap_panel_to_mlflow(
        target_concat_adata,
        artifact_path="umap/{}/target_concat_adata_umap.png".format(stage_label),
        colors=target_colors,
        titles=target_titles,
        size=20,
    )

    output = {
        "source_concat_adata": source_concat_adata,
        "target_concat_adata": target_concat_adata,
    }

    if log_mudata_umaps:
        source_mdata = build_mudata_with_umap(
            source_rna,
            source_atac,
            embedding_key=embedding_key,
            n_neighbors=10,
            resolution=1.5,
        )
        target_mdata = build_mudata_with_umap(
            target_rna,
            target_atac,
            embedding_key=embedding_key,
            n_neighbors=10,
            resolution=1.5,
        )

        log_umap_to_mlflow(
            source_mdata,
            artifact_path="umap/{}/source_mudata_umap.png".format(stage_label),
            title="Source MuData UMAP ({})".format(stage_label),
            color_key="wnn",
            size=20,
        )
        log_umap_to_mlflow(
            target_mdata,
            artifact_path="umap/{}/target_mudata_umap.png".format(stage_label),
            title="Target MuData UMAP ({})".format(stage_label),
            color_key="wnn",
            size=20,
        )

        output["source_mdata"] = source_mdata
        output["target_mdata"] = target_mdata

    return output


def load_run_params(client, run_id, run_name):
    run = client.get_run(run_id)
    params = run.data.params
    print("\nRun params for '{}' (ID: {}):".format(run_name, run_id))
    pprint(params)
    return params


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


def validate_args(args):
    if args.split_seed < 0:
        raise ValueError("--split-seed must be a non-negative integer.")
    if args.stage1_epochs <= 0:
        raise ValueError("--stage1-epochs must be a positive integer.")


def resolve_stage1_cache_config(args, mlflow_client):
    cache_config = Stage1CacheConfig(
        use_cache=args.stage1_mlflow_cache_run_name is not None,
        run_name=args.stage1_mlflow_cache_run_name,
        dual_source_kd=bool(args.stage1_dual_source_kd),
        student_graph_type=args.spatial_graph_type,
        vgp_anchor_mode=args.vgp_anchor_mode,
        skip_gp_attention=args.skip_gp_attention,
    )

    if cache_config.use_cache:
        cache_config.run_id = resolve_run_id_from_name(mlflow_client, cache_config.run_name)
        cache_config.run_params = load_run_params(mlflow_client, cache_config.run_id, cache_config.run_name)

        cache_config.dual_source_kd = (
            str(cache_config.run_params.get("stage1_dual_source_kd", "False")).lower() == "true"
        )
        cache_config.student_graph_type = cache_config.run_params.get("stage1_student_graph", "identity")
        if cache_config.student_graph_type in {"NA", "na", "None", "none", "", None}:
            cache_config.student_graph_type = "identity"

        cached_vgp_anchor_mode = cache_config.run_params.get("vgp_anchor_mode")
        if cached_vgp_anchor_mode in {"spot", "feature"}:
            cache_config.vgp_anchor_mode = cached_vgp_anchor_mode

        print(
            "[Stage1 Cache] run='{}' (id={}), dual_source_kd={}, student_graph={}, vgp_anchor_mode={}".format(
                cache_config.run_name,
                cache_config.run_id,
                cache_config.dual_source_kd,
                cache_config.student_graph_type,
                cache_config.vgp_anchor_mode,
            )
        )

    if cache_config.dual_source_kd and cache_config.student_graph_type == "tangram":
        raise ValueError(
            "Stage-1 dual-source KD with tangram student graph is not supported for source training. "
            "Use spatial, knn, or identity."
        )

    return cache_config


def resolve_domain_label_key(adata, requested_key, default_key, domain_name):
    if requested_key is not None:
        if requested_key in adata.obs.columns:
            adata.obs[requested_key] = adata.obs[requested_key].astype("category")
            return requested_key
        warnings.warn(
            "Requested {} label key '{}' not found. Falling back to default key '{}' if available.".format(
                domain_name,
                requested_key,
                default_key,
            )
        )

    if default_key in adata.obs.columns:
        adata.obs[default_key] = adata.obs[default_key].astype("category")
        return default_key

    warnings.warn(
        "No usable {} label key found (requested='{}', default='{}'). scIB will fall back to pseudo labels.".format(
            domain_name,
            requested_key,
            default_key,
        )
    )
    return None


def load_and_prepare_data_bundle(args):

    source_rna = sc.read_h5ad(os.path.join(BASE_PATH, "source_rna_aligned_SCT.h5ad"))
    source_atac = sc.read_h5ad(os.path.join(BASE_PATH, "source_atac_aligned.h5ad"))
    source_rna.obsm["spatial"] = source_rna.obsm["spatial"] * -1
    source_atac.obsm["spatial"] = source_atac.obsm["spatial"] * -1

    target_rna = sc.read_h5ad(os.path.join(BASE_PATH, "target_rna_aligned_SCT.h5ad"))
    target_atac = sc.read_h5ad(os.path.join(BASE_PATH, "target_atac_aligned.h5ad"))
    assert target_rna.obs_names.equals(target_atac.obs_names), "Target RNA and ATAC must have matching obs_names"

    source_label_key = resolve_domain_label_key(
        source_rna,
        requested_key=args.source_label_key,
        default_key=DEFAULT_SOURCE_LABEL_KEY,
        domain_name="source",
    )
    target_label_key = resolve_domain_label_key(
        target_rna,
        requested_key=args.target_label_key,
        default_key=DEFAULT_TARGET_LABEL_KEY,
        domain_name="target",
    )
    source_rna.uns["label_key"] = source_label_key
    target_rna.uns["label_key"] = target_label_key

    gtf_path = os.path.join(
        os.getenv("DATAPATH"),
        "gene_annotations",
        "gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz",
    )
    if not os.path.exists(gtf_path):
        raise FileNotFoundError("GTF annotation file not found: {}".format(gtf_path))

    MultiGATE.Cal_gene_peak_Net_new(source_rna, source_atac, 150000, file=gtf_path)
    gp_net = source_atac.uns["gene_peak_Net"].copy()
    del source_atac.uns["gene_peak_Net"]

    use_sct_genes = (args.top_n_genes==0 and args.top_n_peaks==0)

    if use_sct_genes:
        source_rna, source_atac, target_rna, target_atac, gp_net = get_sct_genes_and_gp_filtering(
            source_rna=source_rna,
            source_atac=source_atac,
            target_rna=target_rna,
            target_atac=target_atac,
            gp_net=gp_net,
        )

    else:
        source_rna, source_atac, target_rna, target_atac, gp_net = apply_hvg_and_gp_filtering(
            source_rna=source_rna,
            source_atac=source_atac,
            target_rna=target_rna,
            target_atac=target_atac,
            gp_net=gp_net,
            top_n_genes=args.top_n_genes,
            top_n_peaks=args.top_n_peaks,
            rank_type="fused",
        )

    print("Filtered {} genes and {} peaks from gene-peak net".format(len(target_rna.var_names), len(target_atac.var_names)))
    del gp_net

    MultiGATE.Cal_Spatial_Net(source_rna, rad_cutoff=40)
    MultiGATE.Stats_Spatial_Net(source_rna)
    MultiGATE.Cal_Spatial_Net(source_atac, rad_cutoff=40)
    MultiGATE.Stats_Spatial_Net(source_atac)
    source_rna = source_rna[:, source_rna.var["highly_variable"]].copy()
    source_atac = source_atac[:, source_atac.var["highly_variable"]].copy()

    target_rna, target_atac = prepare_target_for_spatial_graph_type(
        target_rna=target_rna,
        target_atac=target_atac,
        source_rna=source_rna,
        source_atac=source_atac,
        spatial_graph_type=args.spatial_graph_type,
        gtf_path=gtf_path,
    )
    source_rna, source_atac = pair_modalities(source_rna, source_atac, domain_name="Source")
    target_rna, target_atac = pair_modalities(target_rna, target_atac, domain_name="Target")

    source_split_bundle = build_domain_split_bundle(
        source_rna,
        source_atac,
        label_key=source_label_key,
        split_seed=int(args.split_seed),
        domain_name="Source",
    )
    target_split_bundle = build_domain_split_bundle(
        target_rna,
        target_atac,
        label_key=target_label_key,
        split_seed=int(args.split_seed + 1),
        domain_name="Target",
    )

    data_bundle = DataBundle(
        source=source_split_bundle,
        target=target_split_bundle,
        split_metadata={},
        combined_gp_dict=None,
    )

    data_bundle.source = configure_source_train_eval_bundle(
        data_bundle.source,
        use_source_split_train_eval=bool(args.source_split_train_eval),
    )
    data_bundle.split_metadata = build_split_metadata(data_bundle.source, data_bundle.target)

    return data_bundle


def build_graph_bundle_from_domains(source_domain, target_domain, bp_width=400, graph_type="ATAC", protein_value=0.001):
    source_graph_tf, source_gp_tf, source_x1, source_x2 = build_graph_inputs(
        source_domain.rna,
        source_domain.atac,
        bp_width=bp_width,
        graph_type=graph_type,
        protein_value=protein_value,
    )
    target_graph_tf, target_gp_tf, target_x1, target_x2 = build_graph_inputs(
        target_domain.rna,
        target_domain.atac,
        bp_width=bp_width,
        graph_type=graph_type,
        protein_value=protein_value,
    )

    if target_x1.shape[1] != source_x1.shape[1] or target_x2.shape[1] != source_x2.shape[1]:
        raise ValueError(
            "Target feature dimensions do not match source model dimensions: "
            "RNA {} vs {}, ATAC {} vs {}.".format(
                target_x1.shape[1],
                source_x1.shape[1],
                target_x2.shape[1],
                source_x2.shape[1],
            )
        )

    return GraphBundle(
        source=GraphInputBundle(graph_tf=source_graph_tf, gp_tf=source_gp_tf, x1=source_x1, x2=source_x2),
        target=GraphInputBundle(graph_tf=target_graph_tf, gp_tf=target_gp_tf, x1=target_x1, x2=target_x2),
        bp_width=bp_width,
        graph_type=graph_type,
        protein_value=protein_value,
    )


def build_graph_bundle(data_bundle, bp_width=400, graph_type="ATAC", protein_value=0.001):
    return build_graph_bundle_from_domains(
        source_domain=data_bundle.source.train,
        target_domain=data_bundle.target.train,
        bp_width=bp_width,
        graph_type=graph_type,
        protein_value=protein_value,
    )

def summarize_stage1_setup(num_epochs, data_bundle, cache_config, trainer_bundle):
    print("Training epochs for stage 1:", num_epochs)
    source_split_mode = "train/eval splits" if data_bundle.source.train.rna.n_obs != data_bundle.source.full.rna.n_obs else "full source train+eval"
    print(
        "Source mode: {} (train/val/test/eval/full = {}/{}/{}/{}/{})".format(
            source_split_mode,
            data_bundle.source.train.rna.n_obs,
            data_bundle.source.val.rna.n_obs,
            data_bundle.source.test.rna.n_obs,
            data_bundle.source.eval.rna.n_obs,
            data_bundle.source.full.rna.n_obs,
        )
    )
    print(
        "Target split sizes (train/val/test/eval): {}/{}/{}/{}".format(
            data_bundle.target.train.rna.n_obs,
            data_bundle.target.val.rna.n_obs,
            data_bundle.target.test.rna.n_obs,
            data_bundle.target.eval.rna.n_obs,
        )
    )
    if cache_config.dual_source_kd:
        print(
            "[Stage1 Dual KD] Enabled: teacher graph=spatial, student graph={}".format(
                cache_config.student_graph_type
            )
        )
        if trainer_bundle.nonspatial is not None:
            print(
                "[Stage1 Non-Spatial] Enabled: auxiliary non-spatial model trains on the student graph "
                "with teacher-style losses."
            )


#%%
def main():

    # setup runtime
    bootstrap_runtime()
    notebook_mode = is_notebook()
    args = parse_args(notebook=notebook_mode)
    validate_args(args)

    # load and prepare data
    data_bundle = load_and_prepare_data_bundle(args)
    mdata = mu.MuData({
        "rna": data_bundle.source.train.rna,
        "atac": data_bundle.source.train.atac
    })

    # setup mudata
    scvi.model.MULTIVI.setup_mudata(
        mdata,
        modalities={
            "rna_layer": "rna",
            "atac_layer": "atac",
        },
    )

    # setup model
    model = scvi.model.MULTIVI(
        mdata,
        n_genes=len(mdata.mod["rna"].var),
        n_regions=len(mdata.mod["atac"].var),
    )

    model.view_anndata_setup()

    # set csr counts data
    mdata.mod["rna"].X = mdata.mod["rna"].layers['counts'].tocsr()
    mdata.mod["atac"].X = mdata.mod["atac"].layers['counts'].tocsr()
    mdata.update()

    # train model
    num_epochs = args.stage1_epochs
    model.train(max_epochs=num_epochs)

    # save model
    model_dir = os.path.join(os.getenv("OUTPATH"), "multivi_mouse_brain_spatial_rna_atac")
    model.save(model_dir, overwrite=True)
    print(f"Model saved to {model_dir}")

    # extracting and visualizing the latent space
    MULTIVI_LATENT_KEY = "X_multivi"

    mdata.obsm[MULTIVI_LATENT_KEY] = model.get_latent_representation()
    sc.pp.neighbors(mdata, use_rep=MULTIVI_LATENT_KEY)
    sc.tl.umap(mdata, min_dist=0.2)
    sc.tl.leiden(mdata, resolution=0.25)

    # initialize the column first
    #mdata.obs["modality"] = ["rna"] * data_bundle.source.train.rna.n_obs + ["atac"] * data_bundle.source.train.atac.n_obs
    mdata.obs = mdata.obs.assign(
        RNA_clusters = mdata.mod['rna'].obs['RNA_clusters'],
        ATAC_clusters = mdata.mod['atac'].obs['ATAC_clusters'],
    )
    sc.pl.umap(mdata, color=["RNA_clusters", "ATAC_clusters", "leiden"])

    assert np.all(mdata.mod['rna'].obsm['spatial'] == mdata.mod['atac'].obsm['spatial']), "Spatial coordinates must have the same shape"
    mdata.obsm['spatial'] = mdata.mod['rna'].obsm['spatial']
    sc.pl.embedding(mdata, color=["RNA_clusters", "ATAC_clusters", "leiden"], basis="spatial", s=60)


if __name__ == "__main__":
    main()

# %%
