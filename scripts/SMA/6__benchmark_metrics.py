#%% Benchmark SMA joint RNA+MSI embeddings: SpatialJEPA teacher/student vs Multigrate.
#
# Loads joint (per-spot) embeddings of one SMA sample and scores them with a benchmark
# modeled on the RNA-ATAC block in MultiGATE/scripts/multigate_co_embed.py:
#
#   A. Bio-conservation        -> scib silhouette_label + Leiden NMI/ARI vs
#                                 RNA_clusters / MSI_clusters.
#   B. Spatial structure       -> niche recovery, within-cluster coherence, and an
#                                 (adapted) orthogonal-target recovery, all on the joint
#                                 embedding with spatial-block CV.
#   C. Cross-modal alignment   -> 1-FOSCTTM and iLISI modality mixing between each model's
#                                 per-modality RNA-only / MSI-only latents.
#
# All latents come precomputed from disk (no model is reloaded):
#   - SpatialJEPA teacher/student: joint_adata.obsm["{teacher,student}_X_emb{,_st,_sm}"],
#     written by 3__spatial_meta_modelling.py.
#   - Multigrate: ${OUTPATH}/multigrate_mouse_sma/<sample>/multigrate_modality_latents.npz
#     (joint/rna/msi), written by multigrate_mouse_sma.py. Multigrate exposes per-modality
#     latents, so (unlike the old TOTALVI baseline) it also joins the cross-modal metric.
#
# Env: `conda activate nichecompass_liana` (has scib_metrics + jax). FOSCTTM uses the jax
# evals_utils.foscttm_moscot; bio-conservation / iLISI use scib_metrics' functional API.

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from scib_metrics import ilisi_knn, nmi_ari_cluster_labels_leiden, silhouette_label
from scib_metrics.nearest_neighbors import NeighborsResults

import warnings
warnings.filterwarnings("ignore")

ENV_FILE_PATH = "/home/mcb/users/dmannk/BAKLAVA_base/BAKLAVA/.env"
load_dotenv(dotenv_path=ENV_FILE_PATH)

OUTPATH = Path(os.environ["OUTPATH"])
BAKLAVA_ROOT = os.environ["BAKLAVA_ROOT"]
# evals_utils.foscttm_moscot (jax) lives at the BAKLAVA repo root.
sys.path.insert(0, os.path.join(os.environ["BAKLAVA_BASE_DIR"], "BAKLAVA"))
from evals_utils import foscttm_moscot

FIG_DIR = Path("/home/mcb/users/dmannk/THESIS_base/overleaf-cibb-2026/figures")

RNA_LABEL_KEY = "RNA_clusters"
MSI_LABEL_KEY = "MSI_clusters"

# per-modality latent obsm keys written by 3__spatial_meta_modelling.py
PER_MODALITY_KEYS = {
    "teacher": ("teacher_X_emb_st", "teacher_X_emb_sm"),
    "student": ("student_X_emb_st", "student_X_emb_sm"),
}

# Spatial-structure metric hyperparameters (match multigate_co_embed.py).
N_SPATIAL_BLOCKS = 10
NICHE_K = 15
SSM_RNG = np.random.default_rng(0)


def is_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except Exception:
        return False


def parse_args(notebook: bool = False):
    parser = argparse.ArgumentParser(
        description="Benchmark SMA joint RNA+MSI embeddings.",
        allow_abbrev=False,
    )
    parser.add_argument("--sample-id", type=str, default="V11L12-109_B1")
    parser.add_argument("--lisi-perplexity", type=int, default=30)
    if notebook:
        return parser.parse_known_args()[0]
    return parser.parse_args()


def _dense(mat):
    return mat.toarray() if sp.issparse(mat) else np.asarray(mat)


def _standardize(X):
    return StandardScaler().fit_transform(np.asarray(X, dtype=float))


def _to_pca(X, n_comps=30):
    """Top components of a feature block (TruncatedSVD == PCA on centred input)."""
    X = _dense(X)
    n = min(n_comps, X.shape[1] - 1)
    return TruncatedSVD(n_components=n, random_state=0).fit_transform(
        StandardScaler(with_mean=False).fit_transform(X)
    )


def _neighbors(X, k):
    """Build a scib_metrics NeighborsResults from an embedding."""
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    dist, idx = nn.kneighbors(X)
    return NeighborsResults(indices=idx, distances=dist)


# ── data loading ─────────────────────────────────────────────────────────────
def load_inputs(sample_id):
    jepa_dir = OUTPATH / "spatialjepa_models" / sample_id
    mg_npz = OUTPATH / "multigrate_mouse_sma" / sample_id / "multigrate_modality_latents.npz"

    import scanpy as sc
    joint = sc.read_h5ad(jepa_dir / "joint_adata.h5ad")
    for key in ("teacher_X_emb", "student_X_emb", "spatial"):
        if key not in joint.obsm:
            raise KeyError(f"joint_adata missing obsm['{key}']")
    for lab in (RNA_LABEL_KEY, MSI_LABEL_KEY):
        if lab not in joint.obs.columns:
            raise KeyError(f"joint_adata missing obs['{lab}']")

    # Per-modality SpatialJEPA latents, precomputed by 3__spatial_meta_modelling.py.
    per_modality = {}
    for name, (st_key, sm_key) in PER_MODALITY_KEYS.items():
        if st_key in joint.obsm and sm_key in joint.obsm:
            per_modality[name] = {
                "st": np.asarray(joint.obsm[st_key]),
                "sm": np.asarray(joint.obsm[sm_key]),
            }
        else:
            warnings.warn(
                f"per-modality latents for '{name}' ({st_key}/{sm_key}) not in "
                f"joint_adata.obsm; its cross-modal metrics will be skipped. Re-run "
                f"3__spatial_meta_modelling.py for sample '{sample_id}' to add them."
            )

    # Multigrate joint + per-modality latents (multigrate_mouse_sma.py), aligned to the
    # joint_adata spot order. Multigrate exposes per-modality latents, so it also joins the
    # cross-modal metric (the old TOTALVI baseline could not).
    x_multigrate = None
    if mg_npz.exists():
        d = np.load(mg_npz, allow_pickle=True)
        mg_obs = pd.Index([str(x) for x in d["obs_names"]])
        pos = mg_obs.get_indexer(joint.obs_names)
        if (pos < 0).any():
            raise ValueError("Multigrate latents do not cover all joint_adata spots.")
        x_multigrate = np.asarray(d["joint"])[pos]
        per_modality["multigrate"] = {
            "st": np.asarray(d["rna"])[pos],
            "sm": np.asarray(d["msi"])[pos],
        }
    else:
        warnings.warn(
            f"Multigrate latents not found at {mg_npz}; Multigrate will be excluded. "
            f"Run multigrate_mouse_sma.py for sample '{sample_id}' first."
        )

    return joint, x_multigrate, per_modality


# ── Metric A: bio-conservation (scib_metrics) ────────────────────────────────
def bio_conservation(emb, labels_by_key):
    """scib silhouette_label (rescaled to [0,1]) + Leiden NMI/ARI vs each label set."""
    nr = _neighbors(emb, 15)
    rows = []
    for lab_name, labels in labels_by_key.items():
        labels = np.asarray(labels)
        rows.append((f"silhouette_{lab_name}",
                     float(silhouette_label(emb, labels, rescale=True))))
        na = nmi_ari_cluster_labels_leiden(nr, labels)
        rows.append((f"nmi_{lab_name}", float(na["nmi"])))
        rows.append((f"ari_{lab_name}", float(na["ari"])))
    return rows


# ── Metric C: cross-modal alignment (JEPA only) ──────────────────────────────
def foscttm(x, y):
    """Mean Fraction Of Samples Closer Than the True Match (jax, evals_utils)."""
    return float(np.asarray(foscttm_moscot(_standardize(x), _standardize(y))).mean())


def ilisi_modality_mixing(st, sm, perplexity=30):
    """iLISI in [0,1] of the modality label over the stacked RNA/MSI latents (scib)."""
    Z = np.vstack([_standardize(st), _standardize(sm)])
    batches = np.concatenate([np.zeros(len(st)), np.ones(len(sm))])
    nr = _neighbors(Z, 3 * perplexity)
    return float(ilisi_knn(nr, batches, perplexity=perplexity))


# ── Metric B: spatial structure (ported from multigate_co_embed.py) ──────────
def _codes_and_onehot(labels):
    cat = pd.Series(np.asarray(labels)).astype("category")
    return cat.cat.codes.to_numpy(), pd.get_dummies(cat).to_numpy().astype(float)


def _mean_spearman(y_true, y_pred):
    rhos = []
    for j in range(y_true.shape[1]):
        rho, _ = spearmanr(y_true[:, j], y_pred[:, j])
        if np.isfinite(rho):
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else np.nan


def _sample_pairs(idx, max_pairs, rng):
    m = idx.size
    n_pairs = min(max_pairs, m * (m - 1) // 2)
    a = rng.integers(0, m, size=n_pairs)
    b = rng.integers(0, m, size=n_pairs)
    keep = a != b
    return idx[a[keep]], idx[b[keep]]


class SpatialScorer:
    """Niche recovery, within-cluster coherence, and (adapted) orthogonal-target
    recovery for a set of joint embeddings, scored with spatial-block CV.

    Adaptation vs the RNA-ATAC reference: cluster partitions are RNA_clusters +
    MSI_clusters, and orthogonal-target predicts neighbour-averaged MSI PCA from
    the JOINT embedding (the joint emb already encodes MSI, so this measures
    MSI-decodability of the embedding, not held-out cross-modal transfer).
    """

    def __init__(self, coords, rna_labels, msi_labels, msi_pca):
        self.coords = np.asarray(coords, dtype=float)
        self.n_ref = self.coords.shape[0]
        self.rna_codes, self.rna_onehot = _codes_and_onehot(rna_labels)
        self.msi_codes, self.msi_onehot = _codes_and_onehot(msi_labels)
        self.cluster_partitions = [self.rna_codes, self.msi_codes]
        self.msi_pca = msi_pca
        self.block_labels = KMeans(
            n_clusters=N_SPATIAL_BLOCKS, random_state=0, n_init=10
        ).fit_predict(self.coords)

    def _build_targets(self, coords_for_targets):
        nn = NearestNeighbors(n_neighbors=NICHE_K + 1).fit(coords_for_targets)
        nbr = nn.kneighbors(coords_for_targets, return_distance=False)[:, 1:]
        y_niche = np.concatenate(
            [self.rna_onehot[nbr].mean(axis=1), self.msi_onehot[nbr].mean(axis=1)],
            axis=1,
        )
        y_orth = _standardize(self.msi_pca[nbr].mean(axis=1))
        return y_niche, y_orth

    def _cv_predict_per_block(self, X, Y, scorer):
        out = {}
        for b in np.unique(self.block_labels):
            test = self.block_labels == b
            train = self.block_labels != b
            if test.sum() < 5 or train.sum() < 20:
                continue
            scaler = StandardScaler().fit(X[train])
            ridge = Ridge(alpha=1.0).fit(scaler.transform(X[train]), Y[train])
            pred = ridge.predict(scaler.transform(X[test]))
            out[b] = scorer(Y[test], pred)
        return out

    def _coherence_per_block(self, Xjoint, coords_for_dist):
        out = {}
        for b in np.unique(self.block_labels):
            idx = np.where(self.block_labels == b)[0]
            if idx.size < 30:
                continue
            rhos, weights = [], []
            for codes in self.cluster_partitions:
                for c in np.unique(codes[idx]):
                    ci = idx[codes[idx] == c]
                    if ci.size < 8:
                        continue
                    pa, pb = _sample_pairs(ci, max_pairs=2000, rng=SSM_RNG)
                    if pa.size < 10:
                        continue
                    ed = np.linalg.norm(Xjoint[pa] - Xjoint[pb], axis=1)
                    pdist = np.linalg.norm(coords_for_dist[pa] - coords_for_dist[pb], axis=1)
                    rho, _ = spearmanr(ed, pdist)
                    if np.isfinite(rho):
                        rhos.append(rho)
                        weights.append(ci.size)
            if rhos:
                out[b] = float(np.average(rhos, weights=weights))
        return out

    def score_lineup(self, lineup, coords_for_dist, coords_for_targets, tag):
        y_niche, y_orth = self._build_targets(coords_for_targets)
        rows = []
        for name, joint_emb in lineup.items():
            niche = self._cv_predict_per_block(joint_emb, y_niche, _mean_spearman)
            orth = self._cv_predict_per_block(joint_emb, y_orth, _mean_spearman)
            coher = self._coherence_per_block(joint_emb, coords_for_dist)
            for metric, per_block in (
                ("niche_recovery", niche),
                ("orthogonal_target", orth),
                ("within_cluster_coherence", coher),
            ):
                for fold, value in per_block.items():
                    rows.append({"setting": tag, "model": name, "metric": metric,
                                 "fold": int(fold), "value": value})
        return pd.DataFrame(rows)


# ── outputs ──────────────────────────────────────────────────────────────────
def save_figure(fig, filename):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] wrote {path}")


def barplot(df, value_col, title, filename):
    # df is already aggregated to one value per (model, metric); no CI needed.
    g = sns.catplot(
        data=df, x="model", y=value_col, col="metric", col_wrap=4,
        kind="bar", palette="Set2", height=3.2, aspect=0.8, sharey=False, ci=None,
    )
    g.set_titles("{col_name}")
    g.set(xlabel="", ylabel="")
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.axhline(0.0, color="#999999", lw=0.8, ls="--")
        for patch in ax.patches:
            patch.set_edgecolor("#cccccc")
            patch.set_linewidth(1)
    g.fig.suptitle(title, y=1.02)
    save_figure(g.fig, filename)


#%%
def main():
    args = parse_args(notebook=is_notebook())
    sample_id = args.sample_id
    print(f"[INFO] Benchmarking SMA sample {sample_id}")

    joint, x_multigrate, per_modality = load_inputs(sample_id)
    n_spots = joint.n_obs
    rna_labels = joint.obs[RNA_LABEL_KEY].to_numpy()
    msi_labels = joint.obs[MSI_LABEL_KEY].to_numpy()
    coords = np.asarray(joint.obsm["spatial"], dtype=float)
    print(f"[INFO] {n_spots} spots; "
          f"RNA_clusters={len(np.unique(rna_labels))}, MSI_clusters={len(np.unique(msi_labels))}")

    # PCA floor: TruncatedSVD(30) on the ST block + on the SM block of `normalized`.
    st_mask = joint.var["type"].values == "ST"
    sm_mask = joint.var["type"].values == "SM"
    norm = joint.layers["normalized"]
    pca_floor = np.concatenate([_to_pca(norm[:, st_mask], 30), _to_pca(norm[:, sm_mask], 30)], axis=1)
    msi_pca = _to_pca(norm[:, sm_mask], 30)

    # lineup of joint embeddings (standardized so distances/ridge are comparable)
    lineup = {
        "teacher": _standardize(joint.obsm["teacher_X_emb"]),
        "student": _standardize(joint.obsm["student_X_emb"]),
    }
    if x_multigrate is not None:
        lineup["multigrate"] = _standardize(x_multigrate)
    lineup["pca"] = _standardize(pca_floor)

    # ── Metric A: bio-conservation ──────────────────────────────────────────
    labels_by_key = {RNA_LABEL_KEY: rna_labels, MSI_LABEL_KEY: msi_labels}
    bio_rows = []
    for name, emb in lineup.items():
        for metric, value in bio_conservation(emb, labels_by_key):
            bio_rows.append({"model": name, "metric": metric, "value": value})
    bio_df = pd.DataFrame(bio_rows)
    print("\n[bio-conservation]")
    print(bio_df.pivot(index="model", columns="metric", values="value").round(3).to_string())

    # ── Metric B: spatial structure ─────────────────────────────────────────
    scorer = SpatialScorer(coords, rna_labels, msi_labels, msi_pca)
    perm = SSM_RNG.permutation(n_spots)
    spatial_df = pd.concat(
        [
            scorer.score_lineup(lineup, coords, coords, tag="real"),
            scorer.score_lineup(lineup, coords[perm], coords[perm], tag="coord_permuted"),
        ],
        ignore_index=True,
    )
    spatial_summary = (
        spatial_df.groupby(["setting", "model", "metric"])["value"].mean().reset_index()
    )
    print("\n[spatial structure] real (mean over folds):")
    print(spatial_summary[spatial_summary.setting == "real"]
          .pivot(index="model", columns="metric", values="value").round(3).to_string())
    print("\n[spatial structure] coord-permuted control (~chance):")
    print(spatial_summary[spatial_summary.setting == "coord_permuted"]
          .pivot(index="model", columns="metric", values="value").round(3).to_string())

    # ── Metric C: cross-modal alignment (JEPA only) ─────────────────────────
    cross_rows = []
    for name, mods in per_modality.items():
        st, sm = mods["st"], mods["sm"]
        if st.shape[1] != sm.shape[1]:
            warnings.warn(f"[{name}] RNA/MSI latent dims differ "
                          f"({st.shape[1]} vs {sm.shape[1]}); skipping FOSCTTM.")
            continue
        cross_rows.append({"model": name, "metric": "1-FOSCTTM", "value": 1.0 - foscttm(st, sm)})
        cross_rows.append({"model": name, "metric": "iLISI",
                           "value": ilisi_modality_mixing(st, sm, args.lisi_perplexity)})
    cross_df = pd.DataFrame(cross_rows)
    print("\n[cross-modal alignment]:")
    if not cross_df.empty:
        print(cross_df.pivot(index="model", columns="metric", values="value").round(3).to_string())
    else:
        print("  (no per-modality latents found; see warning above)")

    # ── figures + tidy CSV ──────────────────────────────────────────────────
    joint_metrics_df = pd.concat([bio_df, spatial_summary[spatial_summary.setting == "real"]
                                  .drop(columns="setting")[["model", "metric", "value"]]],
                                 ignore_index=True)
    barplot(joint_metrics_df, "value",
            f"SMA {sample_id}: joint-embedding metrics",
            "sma_benchmark_joint_metrics_barplot.pdf")
    if not cross_df.empty:
        barplot(cross_df, "value",
                f"SMA {sample_id}: cross-modal alignment",
                "sma_benchmark_crossmodal_barplot.pdf")

    all_metrics = pd.concat([
        joint_metrics_df.assign(family="joint"),
        cross_df.assign(family="crossmodal") if not cross_df.empty else cross_df,
    ], ignore_index=True)
    out_dir = OUTPATH / "multigrate_mouse_sma" / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "benchmark_metrics.csv"
    all_metrics.to_csv(out_csv, index=False)
    print(f"\n[INFO] wrote {out_csv}")


if __name__ == "__main__":
    main()

# %%
