"""scGLUE helpers for integrating the disjoint MSI metabolite matrices.

Treats the two MALDI matrices (and optionally RNA) as separate modalities and aligns them
with a metalinks-derived guidance graph. Two pairing regimes are supported:

* paired   -> ``PairedSCGLUEModel`` with cells matched across modalities by Visium array
              barcode (same array position, different tissue section -> a pseudo-pairing).
* unpaired -> ``SCGLUEModel`` (diagonal), ignoring the barcode correspondence.

And two guidance regimes:

* ``rna_anchored``    -> nodes = RNA genes + both m/z sets; edges = gene<->m/z from metalinks,
                         so genes bridge the two metabolite spaces. Strong bridge.
* ``metabolite_only`` -> nodes = both m/z sets; edges = m/z(fmp10)<->m/z(dhb) projected
                         through shared metalinks genes. Sparse (only annotated m/z connect).

**Environment:** `conda activate nichecompass_liana` (scglue 0.4.0).
"""

from __future__ import annotations

import re
from itertools import product

import anndata as ad
import networkx as nx
import scanpy as sc
import scglue

from sma_fusion import tic_pseudocounts

# KNOWN ISSUE (unresolved) — scGLUE training does not progress in this environment.
# fit_SCGLUE reaches "Engine run starting" and then makes no progress:
#   * With the default GRAPH_SHUFFLE_NUM_WORKERS=1, scGLUE spawns a background worker to
#     shuffle graph minibatches that DEADLOCKS in this forked / `conda run` context
#     (process idle, 0% GPU, no epochs).
#   * Setting it to 0 (below) avoids the deadlock but the in-process graph shuffler is then
#     CPU-bound: the process pins one core at ~98% with 0% GPU utilisation and does not
#     finish even a single epoch on a 300-cell / 4292-node-graph smoke test in ~10 min,
#     despite autodevice correctly selecting a GPU and allocating model memory on it.
# So this is a scGLUE dataloader/runtime problem, not a defect in the code below: the
# guidance graph builds and passes scglue.graph.check_graph, configure_dataset/model init
# all succeed — only the training loop won't advance. Everything up to run_glue() is
# validated; run_glue() itself has not been confirmed end-to-end.
# To revisit: try running in a plain interactive shell (not `conda run`), set
# scglue.config.CPU_ONLY appropriately, experiment with GRAPH_SHUFFLE_NUM_WORKERS and
# DATALOADER_NUM_WORKERS, or upgrade scglue; consider `multiprocessing.set_start_method
# ("spawn")`. The MultiVI route (2__sma_metabolite_multivi.py) is fully working in the
# meantime.
scglue.config.GRAPH_SHUFFLE_NUM_WORKERS = 0


# --------------------------------------------------------------------------- #
# Per-modality preparation                                                    #
# --------------------------------------------------------------------------- #
def prep_glue_modality(
    adata: ad.AnnData, prob_model: str, paired: bool, n_comps: int = 50
) -> ad.AnnData:
    """Prepare one modality for ``fit_SCGLUE``.

    RNA (``prob_model='NB'``) reconstructs raw counts from the ``counts`` layer.
    MSI (``prob_model='ZILN'``) reconstructs continuous TIC-log-normalised intensities.
    Both encode from a PCA representation in ``obsm['X_pca']``.
    """
    out = adata.copy()
    if prob_model == "NB":
        if "counts" not in out.layers:
            out.layers["counts"] = out.X.copy()
        use_layer = "counts"
        sc.pp.normalize_total(out)
        sc.pp.log1p(out)
    else:  # ZILN / continuous MSI
        out = tic_pseudocounts(out)
        out.X = out.layers["lognorm"].copy()
        use_layer = None

    sc.pp.scale(out)
    n_comps = min(n_comps, out.n_vars - 1, out.n_obs - 1)
    sc.pp.pca(out, n_comps=n_comps)

    scglue.models.configure_dataset(
        out,
        prob_model,
        use_highly_variable=False,
        use_layer=use_layer,
        use_rep="X_pca",
        use_obs_names=paired,
    )
    return out


# --------------------------------------------------------------------------- #
# Guidance graph from metalinks                                               #
# --------------------------------------------------------------------------- #
def _metabolite_to_mz(msi: ad.AnnData, metabolite: str, var_col: str = "annotated") -> list[str]:
    """m/z features in ``msi`` whose METASPACE annotation mentions ``metabolite``."""
    if var_col not in msi.var:
        return []
    ann = msi.var[var_col].astype(str)
    mask = ann.str.contains(re.escape(metabolite), case=False, na=False)
    return list(msi.var_names[mask])


def _gene_to_mz_edges(
    msi_by_source: dict[str, ad.AnnData], metalinks
) -> dict[str, list[tuple[str, int]]]:
    """Map each metalinks gene to the (m/z, sign) features it links, across all sources."""
    gene_edges: dict[str, list[tuple[str, int]]] = {}
    pairs = metalinks.drop_duplicates(["metabolite", "gene_symbol", "mor"])
    for _, row in pairs.iterrows():
        gene = str(row["gene_symbol"])
        sign = 1 if int(row.get("mor", 1)) != 0 else -1
        for msi in msi_by_source.values():
            for mz in _metabolite_to_mz(msi, str(row["metabolite"])):
                gene_edges.setdefault(gene, []).append((mz, sign))
    return gene_edges


def _add_cross_edge(graph: nx.MultiDiGraph, u: str, v: str, sign: int) -> None:
    graph.add_edge(u, v, weight=1.0, sign=sign, type="fwd")
    graph.add_edge(v, u, weight=1.0, sign=sign, type="rev")


def _add_self_loops(graph: nx.MultiDiGraph, nodes) -> None:
    for n in nodes:
        graph.add_edge(n, n, weight=1.0, sign=1, type="loop")


def build_guidance_graph(
    modalities: dict[str, ad.AnnData],
    msi_by_source: dict[str, ad.AnnData],
    metalinks,
    mode: str,
    rna_key: str = "rna",
) -> nx.MultiDiGraph:
    """Build a scGLUE guidance graph (MultiDiGraph with weight/sign/type + self-loops)."""
    graph = nx.MultiDiGraph()
    all_nodes = [v for ad_ in modalities.values() for v in ad_.var_names]
    graph.add_nodes_from(all_nodes)
    gene_edges = _gene_to_mz_edges(msi_by_source, metalinks)

    if mode == "rna_anchored":
        rna_genes = set(modalities[rna_key].var_names)
        for gene, mz_list in gene_edges.items():
            if gene not in rna_genes:
                continue
            for mz, sign in mz_list:
                _add_cross_edge(graph, gene, mz, sign)

    elif mode == "metabolite_only":
        # Project gene<->m/z bipartite links to m/z(fmp10)<->m/z(dhb) via shared genes.
        sources = list(msi_by_source)
        for mz_list in gene_edges.values():
            by_src = {
                s: [mz for mz, _ in mz_list if mz in set(msi_by_source[s].var_names)]
                for s in sources
            }
            for u, v in product(by_src[sources[0]], by_src[sources[1]]):
                _add_cross_edge(graph, u, v, 1)
    else:
        raise ValueError(f"unknown guidance mode: {mode!r}")

    _add_self_loops(graph, all_nodes)
    scglue.graph.check_graph(graph, list(modalities.values()))
    return graph


# --------------------------------------------------------------------------- #
# Fit                                                                         #
# --------------------------------------------------------------------------- #
def run_glue(
    modalities: dict[str, ad.AnnData],
    graph: nx.MultiDiGraph,
    paired: bool,
    max_epochs: int | None = None,
    skip_balance: bool = False,
) -> dict:
    """Fit scGLUE; return model, per-modality cell latents, and feature embeddings."""
    model_cls = scglue.models.PairedSCGLUEModel if paired else scglue.models.SCGLUEModel
    fit_kws = {"max_epochs": max_epochs} if max_epochs else None
    glue = scglue.models.fit_SCGLUE(
        modalities, graph, model=model_cls, fit_kws=fit_kws, skip_balance=skip_balance
    )

    latents = {}
    for name, adata in modalities.items():
        adata.obsm["X_glue"] = glue.encode_data(name, adata)
        latents[name] = adata.obsm["X_glue"]
    feature_embeddings = glue.encode_graph(graph)

    consistency = scglue.models.integration_consistency(glue, modalities, graph)
    return {
        "model": glue,
        "latents": latents,
        "feature_embeddings": feature_embeddings,
        "consistency": consistency,
    }
