# %% [markdown]
# # SMA multi-modal modelling (Vicari et al., 2023)
#
# LIANA+ MISTy and local bivariate analysis for paired Visium RNA + MALDI-MSI slides.
# Adapted from `sma.py`; uses locally extracted Visium outs and METASPACE-annotated MSI h5ad files.
#
# **Environment:** `conda activate nichecompass_liana`

# %%
from dotenv import load_dotenv

load_dotenv(dotenv_path="/home/mcb/users/dmannk/BAKLAVA_base/BAKLAVA/.env")

# %%
import io
import os
import re
import zipfile
from pathlib import Path

import anndata as ad
import liana as li
import mudata as mu
import numpy as np
import pandas as pd
import scanpy as sc
from adjustText import adjust_text
from matplotlib import pyplot as plt

from load_aligned_mudata import load_sample

# %%
kwargs = {"frameon": False, "size": 1.5, "img_key": "lowres"}


def show_figure(fig=None) -> None:
    if fig is None:
        fig = plt.gcf()
    try:
        from IPython.display import display

        display(fig)
    except Exception:
        plt.show()

DATAPATH = Path(os.environ["DATAPATH"])
SMA_ROOT = DATAPATH / "vicari_2023" / "mendeley_sma"
SMA_ZIP = SMA_ROOT / "sma.zip"
MSI_H5AD = DATAPATH / "vicari_2023" / "msi_h5ad"
H5MU_EXPORT = DATAPATH / "vicari_2023" / "h5mu_export"
ORTHOLOGS = DATAPATH / "gene_annotations" / "human_mouse_gene_orthologs.csv"

BANDWIDTH = 500
CUTOFF = 0.1
N_TOP_RNA = 5000
N_TOP_MSI = 150
N_BIVARIATE_PERMS = 1000

SAMPLES = {
    "V11L12-038_B1": {
        "section": "B1",
        "msi_h5ad": MSI_H5AD / "V11L12-038_B1_DHB_lipids.metaspace_annotated.h5ad",
    },
    "V11L12-038_D1": {
        "section": "D1",
        "msi_h5ad": MSI_H5AD / "V11L12-038_D1_9AA_metabolites.metaspace_annotated.h5ad",
    },
}


def raw_barcode(obs_name: str) -> str:
    barcode = str(obs_name).split(":", 1)[-1]
    return re.sub(r"(-\d+)[_.]\d+$", r"\1", barcode)


def prep_fusion_modality(adata: ad.AnnData, source: str, sample_id: str) -> ad.AnnData:
    out = adata.copy()
    out.obs["source"] = source
    out.obs["source_sample_id"] = sample_id
    out.obs["raw_barcode"] = [raw_barcode(obs_name) for obs_name in out.obs_names]

    # Keep FMP-10 and DHB rows distinct after concatenation.
    out.obs_names = [f"{source}:{barcode}" for barcode in out.obs["raw_barcode"]]
    return out


def _combine_var_tables(inputs: dict[str, ad.AnnData], var_names: pd.Index) -> pd.DataFrame:
    """Restore var annotations after AnnData concatenation.

    anndata.concat drops var columns unless told how to merge them. Here we keep
    all columns and add the source(s) each feature was observed in.
    """
    frames = []
    for source, adata in inputs.items():
        var = adata.var.copy()
        var["feature_sources"] = source
        frames.append(var)

    var = pd.concat(frames, axis=0, sort=False)
    if var.index.has_duplicates:
        feature_sources = var.groupby(level=0)["feature_sources"].agg(
            lambda values: ";".join(sorted(pd.unique(values.astype(str))))
        )
        var = var.drop(columns="feature_sources").groupby(level=0).first()
        var["feature_sources"] = feature_sources

    return var.reindex(var_names)


def concat_modalities_keep_obs_var(inputs: dict[str, ad.AnnData]) -> ad.AnnData:
    out = ad.concat(
        inputs,
        label="source_batch",
        index_unique=None,
        join="outer",
        merge="same",
        uns_merge="same",
    )
    out.var = _combine_var_tables(inputs, out.var_names)
    return out


def load_fmp10_partner_fused_mudata(
    fmp10_sample_id: str = "V11L12-109_B1",
    partner_sample_id: str = "V11L12-038_B1",
    partner_source: str = "dhb",
    export_dir: Path = H5MU_EXPORT,
    write: bool = False,
) -> mu.MuData:
    fmp10 = load_sample(fmp10_sample_id, export_dir=export_dir)
    partner = load_sample(partner_sample_id, export_dir=export_dir)

    rna_inputs = {
        "fmp10": prep_fusion_modality(fmp10.mod["rna"], "fmp10", fmp10_sample_id),
        partner_source: prep_fusion_modality(
            partner.mod["rna"], partner_source, partner_sample_id
        ),
    }
    msi_inputs = {
        "fmp10": prep_fusion_modality(fmp10.mod["msi"], "fmp10", fmp10_sample_id),
        partner_source: prep_fusion_modality(
            partner.mod["msi"], partner_source, partner_sample_id
        ),
    }

    mdata = mu.MuData(
        {
            "rna": concat_modalities_keep_obs_var(rna_inputs),
            "msi": concat_modalities_keep_obs_var(msi_inputs),
        }
    )
    mdata.update()

    if write:
        out_path = export_dir / (
            f"{fmp10_sample_id}__{partner_sample_id}.fmp10_{partner_source}_concat.h5mu"
        )
        mdata.write(out_path)

    return mdata


def load_fmp10_dhb_fused_mudata(
    fmp10_sample_id: str = "V11L12-109_B1",
    dhb_sample_id: str = "V11L12-038_B1",
    export_dir: Path = H5MU_EXPORT,
    write: bool = False,
) -> mu.MuData:
    return load_fmp10_partner_fused_mudata(
        fmp10_sample_id=fmp10_sample_id,
        partner_sample_id=dhb_sample_id,
        partner_source="dhb",
        export_dir=export_dir,
        write=write,
    )


def load_fmp10_nineaa_fused_mudata(
    fmp10_sample_id: str = "V11L12-109_B1",
    nineaa_sample_id: str = "V11L12-038_D1",
    export_dir: Path = H5MU_EXPORT,
    write: bool = False,
) -> mu.MuData:
    return load_fmp10_partner_fused_mudata(
        fmp10_sample_id=fmp10_sample_id,
        partner_sample_id=nineaa_sample_id,
        partner_source="nineaa",
        export_dir=export_dir,
        write=write,
    )


def _dense_X(adata: ad.AnnData) -> np.ndarray:
    return adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)


def dopamine_correlations_in_intact_striatum(mdata: mu.MuData) -> pd.Series:
    from scipy.stats import spearmanr

    msi = mdata["msi"]
    annotations = msi.var["annotation"].astype(str)
    annotated_metabolites = msi.var[~annotations.eq("")].index
    intact_striatum = msi.obs["lesion"].eq("intact") & msi.obs["region"].eq("striatum")
    dopamine_idx = msi.var[annotations.eq("Dopamine")].index

    x = _dense_X(msi[intact_striatum, dopamine_idx])
    y = _dense_X(msi[intact_striatum, annotated_metabolites])

    spear = spearmanr(x, y).statistic
    spear_dopamine = spear[0][1:]
    spear_dopamine_series = pd.Series(
        spear_dopamine,
        index=msi.var.loc[annotated_metabolites, "annotation"].values,
    )
    return spear_dopamine_series.sort_values(ascending=False)


# %%
fused_mdata = load_fmp10_dhb_fused_mudata()
fused_mdata

# %%
fused_nineaa_mdata = load_fmp10_nineaa_fused_mudata()
fused_nineaa_mdata

# %%
spear_dopamine_series = dopamine_correlations_in_intact_striatum(fused_mdata)
spear_dopamine_series.plot(rot=90)

#spear_nineaa_dopamine_series = dopamine_correlations_in_intact_striatum(fused_nineaa_mdata)
#spear_nineaa_dopamine_series.plot(rot=90)

# top 12 metabolites have same rank (+ve correlation), i.e.: 
    # dhb_rank = spear_dopamine_series.rank(ascending=False).rename("dhb_rank")
    # nineaa_rank = spear_nineaa_dopamine_series.rank(ascending=False).rename("nineaa_rank")
    # ranks_df = pd.merge(dhb_rank, nineaa_rank, how='outer', left_index=True, right_index=True)
    # print(ranks_df.sort_values('dhb_rank').dropna())


def visium_outs(section: str) -> Path:
    return (
        SMA_ROOT
        / "sma"
        / "V11L12-038"
        / f"V11L12-038_{section}"
        / "output_data"
        / f"V11L12-038_{section}_RNA"
        / "outs"
    )


def read_outs_csv(section: str, fname: str) -> pd.DataFrame:
    """Read Space Ranger sidecar CSV from disk or fall back to sma.zip."""
    rel = (
        f"sma/V11L12-038/V11L12-038_{section}/output_data/"
        f"V11L12-038_{section}_RNA/outs/{fname}"
    )
    local = SMA_ROOT / rel
    if local.exists():
        return pd.read_csv(local, index_col=0)
    with zipfile.ZipFile(SMA_ZIP) as zf:
        return pd.read_csv(io.BytesIO(zf.read(rel)), index_col=0)


def load_rna(section: str) -> sc.AnnData:
    rna = sc.read_visium(visium_outs(section))
    rna.var_names_make_unique()

    for col, fname in [("lesion", "lesion.csv"), ("region", "region.csv")]:
        sidecar = read_outs_csv(section, fname)
        rna.obs[col] = pd.Categorical(sidecar.reindex(rna.obs_names).iloc[:, 0])

    sc.pp.calculate_qc_metrics(rna, inplace=True)
    sc.pp.filter_genes(rna, min_cells=10)
    rna.obs["log1p_total_counts"] = np.log1p(rna.obs["total_counts"])
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.highly_variable_genes(rna, flavor="cell_ranger", n_top_genes=N_TOP_RNA)
    return rna


def load_msi(msi_path: Path) -> sc.AnnData:
    msi = sc.read_h5ad(msi_path)
    msi.var_names_make_unique()
    sc.pp.normalize_total(msi, target_sum=1e4)
    sc.pp.log1p(msi)
    sc.pp.highly_variable_genes(msi, flavor="cell_ranger", n_top_genes=N_TOP_MSI)
    msi = msi[:, msi.var["highly_variable"]].copy()
    sc.pp.scale(msi, max_value=5)
    return msi


def interpolate_msi_to_visium(msi: sc.AnnData, rna: sc.AnnData) -> sc.AnnData:
    """Project MSI pixels onto the Visium spot grid (unaligned native coordinates)."""
    metabs = li.ut.interpolate_adata(
        target=msi, reference=rna, use_raw=False, spatial_key="spatial"
    )
    for col in ("lesion", "region"):
        if col in rna.obs:
            metabs.obs[col] = pd.Categorical(rna.obs[col].values)
    metabs.obsm["spatial"] = rna.obsm["spatial"].copy()
    metabs.uns["spatial"] = rna.uns["spatial"].copy()
    return metabs


def load_metalinks() -> pd.DataFrame:
    metalinks = li.rs.get_metalinks(
        tissue_location="Brain",
        biospecimen_location="Cerebrospinal Fluid (CSF)",
        source=["CellPhoneDB", "NeuronChat"],
    )
    map_df = pd.read_csv(ORTHOLOGS)
    map_df = map_df.rename(columns={"Gene name": "source", "Mouse gene name": "target"})
    map_df = map_df.drop(columns=["Gene stable ID", "Mouse gene stable ID"])
    return li.rs.translate_column(
        resource=metalinks, map_df=map_df, column="gene_symbol", one_to_many=1
    )


def receptor_view(rna: sc.AnnData, metalinks: pd.DataFrame) -> sc.AnnData:
    receptors = np.intersect1d(metalinks["gene_symbol"].unique(), rna.var_names)
    return rna[:, receptors].copy()


def metalinks_mz_interactions(
    msi: sc.AnnData, rna: sc.AnnData, metalinks: pd.DataFrame
) -> list[tuple[str, str]]:
    """Map MetalinksDB metabolite names to m/z features via METASPACE annotation strings."""
    if "annotated" not in msi.var:
        return []

    annotations = msi.var["annotated"].astype(str)
    interactions = []
    seen = set()
    for _, row in metalinks.drop_duplicates(["metabolite", "gene_symbol"]).iterrows():
        metab, gene = row["metabolite"], row["gene_symbol"]
        if gene not in rna.var_names:
            continue
        mask = annotations.str.contains(re.escape(metab), case=False, na=False)
        if not mask.any():
            continue
        mz = msi.var_names[mask][0]
        key = (mz, gene)
        if key not in seen:
            interactions.append(key)
            seen.add(key)
    return interactions


def run_misty(
    msi: sc.AnnData,
    rec: sc.AnnData,
    ct: sc.AnnData | None = None,
) -> li.mt.MistyData:
    # msi is the interpolated (Visium-grid) MSI; its spatial coords are the
    # reference so that connectivity dimensions match the intra view obs count.
    # Unlike sma.py (where native MSI carried lesion labels and was used as
    # reference), here lesion labels only exist on the Visium grid.
    reference = msi.obsm["spatial"]
    views = {"intra": msi, "receptor": rec}
    if ct is not None:
        li.ut.spatial_neighbors(
            ct,
            bandwidth=BANDWIDTH,
            cutoff=CUTOFF,
            spatial_key="spatial",
            reference=reference,
            set_diag=False,
            standardize=False,
        )
        views["ct"] = ct

    li.ut.spatial_neighbors(
        rec,
        bandwidth=BANDWIDTH,
        cutoff=CUTOFF,
        spatial_key="spatial",
        reference=reference,
        set_diag=False,
        standardize=False,
    )

    misty = li.mt.MistyData(views, enforce_obs=False)
    misty(
        model=li.mt.sp.LinearModel,
        verbose=True,
        bypass_intra=True,
        maskby="lesion",
    )
    return misty


def draw_misty_target_metrics(
    misty,
    sample_id: str,
    hemisphere: str,
    stat: str = "multi_R2",
    top_n: int = 20,
) -> None:
    """Matplotlib reimplementation of ``li.pl.target_metrics`` for interactive display."""
    df = misty.uns["target_metrics"].copy()
    df = df[df["intra_group"] == hemisphere].dropna(subset=[stat])
    if df.empty:
        print(f"No plottable {stat} values for {sample_id} ({hemisphere}).")
        return

    targets = df.sort_values(stat, ascending=False)["target"].drop_duplicates().head(top_n)
    df = df[df["target"].isin(targets)]
    x = np.arange(len(targets))

    fig, ax = plt.subplots(figsize=(max(6, len(targets) * 0.35), 5))
    for i, target in enumerate(targets):
        vals = df.loc[df["target"] == target, stat].to_numpy()
        ax.scatter(np.full(len(vals), i), vals, s=30, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=90)
    ax.set_ylabel(stat)
    ax.set_xlabel("Target")
    ax.set_title(f"{sample_id} — {hemisphere}")
    ax.axhline(0, color="grey", linewidth=0.5, zorder=1)
    fig.tight_layout()
    show_figure(fig)


def run_bivariate(
    msi: sc.AnnData,
    rna: sc.AnnData,
    interactions: list[tuple[str, str]],
) -> sc.AnnData | None:
    if not interactions:
        return None

    rna_hv = rna[:, rna.var["highly_variable"]].copy()
    interactions = [(m, g) for m, g in interactions if m in msi.var_names and g in rna_hv.var_names]
    if not interactions:
        return None

    mdata = mu.MuData({"msi": msi, "rna": rna_hv}, obsm=rna.obsm, obs=rna.obs, uns=rna.uns)
    li.ut.spatial_neighbors(mdata, bandwidth=BANDWIDTH, cutoff=CUTOFF, set_diag=True)
    return li.mt.bivariate(
        mdata,
        local_name="cosine",
        x_mod="msi",
        y_mod="rna",
        x_use_raw=False,
        y_use_raw=False,
        verbose=True,
        mask_negatives=True,
        n_perms=N_BIVARIATE_PERMS,
        interactions=interactions,
        x_transform=sc.pp.scale,
        y_transform=sc.pp.scale,
    )


def process_sample(sample_id: str, cfg: dict, metalinks: pd.DataFrame) -> dict:
    section = cfg["section"]
    print(f"\n=== {sample_id} ({section}) ===")

    rna = load_rna(section)
    msi_native = load_msi(cfg["msi_h5ad"])
    msi = interpolate_msi_to_visium(msi_native, rna)
    rec = receptor_view(rna, metalinks)

    misty = run_misty(msi=msi, rec=rec)
    interactions = metalinks_mz_interactions(msi_native, rna, metalinks)
    lrdata = run_bivariate(msi, rna, interactions)

    return {
        "sample_id": sample_id,
        "rna": rna,
        "msi": msi,
        "rec": rec,
        "misty": misty,
        "interactions": interactions,
        "lrdata": lrdata,
    }


# %%
metalinks = load_metalinks()
metalinks.head()

# %% V11L12-038 B1 (DHB lipids) and D1 (9-AA metabolites)

results = {sample_id: process_sample(sample_id, cfg, metalinks) for sample_id, cfg in SAMPLES.items()}

# %%
fig, axes = plt.subplots(len(SAMPLES), 2, figsize=(8, 4 * len(SAMPLES)))
if len(SAMPLES) == 1:
    axes = np.array([axes])

for ax_row, (sample_id, res) in zip(axes, results.items()):
    sc.pl.spatial(
        res["rna"],
        color="log1p_total_counts",
        ax=ax_row[0],
        title=f"{sample_id} RNA",
        **kwargs,
        show=False,
    )
    top_mz = res["msi"].var_names[0]
    sc.pl.spatial(
        res["msi"],
        color=top_mz,
        ax=ax_row[1],
        title=f"{sample_id} MSI ({top_mz})",
        cmap="magma",
        **kwargs,
        show=False,
    )

fig.tight_layout()
show_figure(fig)

# %%
for sample_id, res in results.items():
    print(f"\n--- MISTy target metrics: {sample_id} (intact hemisphere) ---")
    draw_misty_target_metrics(res["misty"], sample_id, "intact")

    print(f"--- MISTy target metrics: {sample_id} (lesioned hemisphere) ---")
    draw_misty_target_metrics(res["misty"], sample_id, "lesioned")

# %% Local metabolite–receptor co-localization (MetalinksDB matches only)

for sample_id, res in results.items():
    lrdata = res["lrdata"]
    interactions = res["interactions"]
    print(f"{sample_id}: {len(interactions)} MetalinksDB interactions mapped to m/z features")
    if lrdata is None or not interactions:
        continue

    plot_cols = [f"{m}^{g}" for m, g in interactions[:2]]
    plot_cols = [c for c in plot_cols if c in lrdata.obs.columns or c in lrdata.layers]
    if not plot_cols:
        plot_cols = [f"{res['interactions'][0][0]}^{res['interactions'][0][1]}"]

    sc.pl.spatial(
        lrdata,
        color=plot_cols,
        cmap="cividis_r",
        vmax=1,
        layer="pvals",
        **kwargs,
    )

# %% Top MISTy predictors for the best-predicted m/z peak (intact hemisphere)

for sample_id, res in results.items():
    misty = res["misty"]
    metrics = misty.uns["target_metrics"]
    intact = metrics[metrics["intra_group"] == "intact"].copy()
    if intact.empty:
        continue

    target = intact.sort_values("multi_R2", ascending=False).iloc[0]["target"]
    interactions = misty.uns["interactions"]
    subset = interactions[
        (interactions["intra_group"] == "intact") & (interactions["target"] == target)
    ].copy()
    if subset.empty:
        continue

    subset["rank"] = subset["importances"].rank(ascending=False)
    plt.figure(figsize=(5, 4))
    plt.scatter(
        subset["rank"],
        subset["importances"],
        s=11,
        c=subset["view"].map({"receptor": "#a11838", "ct": "#008B8B"}).fillna("#aaaaaa"),
    )
    texts = []
    for _, row in subset[subset["rank"] <= 10].iterrows():
        texts.append(plt.text(row["rank"], row["importances"], row["predictor"], fontsize=10))
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color="grey", lw=1.5))
    plt.title(f"{sample_id}: {target}")
    plt.tight_layout()
    show_figure()

# %%
