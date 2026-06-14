"""
Annotate the SMA MSI modality in aligned RNA+MSI MuData files via METASPACE.

This script works from the locally exported ``.h5mu`` files produced from
``se.multi.list``. For each sample it:

1. loads ``mdata.mod["msi"]`` with ``load_aligned_mudata.load_sample``;
2. reuses cached METASPACE result CSVs when available, or exports annotations
   from an already processed public METASPACE dataset with the matching sample
   name;
3. matches METASPACE annotation m/z values back to MSI features by PPM
   tolerance; and
4. overwrites the original ``.h5mu`` with the annotated MSI ``var`` table.

No Figshare imzML/IBD download or METASPACE upload is performed here.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from metaspace import SMInstance


SCRIPT_DIR = Path(os.path.join(os.getenv("BAKLAVA_ROOT"), "scripts", "SMA"))
BAKLAVA_ROOT = SCRIPT_DIR.parents[1]
load_dotenv(dotenv_path=BAKLAVA_ROOT / ".env")

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from load_aligned_mudata import load_sample, list_samples  # noqa: E402


BAKLAVA_ROOT = Path(os.environ.get("BAKLAVA_ROOT", BAKLAVA_ROOT)).expanduser()
BAKLAVA_BASE = BAKLAVA_ROOT.parent
DATAPATH = Path(os.environ.get("DATAPATH", BAKLAVA_BASE / "data")).expanduser()

H5MU_EXPORT_DIR = DATAPATH / "vicari_2023" / "h5mu_export"
OUT_DIR = BAKLAVA_BASE / "outputs" / "metaspace_output"

FDR = 0.20
# Slightly wider than the METASPACE search PPM to account for rounded m/z
# values stored in the exported h5mu MSI var table.
PPM_MATCH_TOL = 5.0


# METASPACE availability notes from the SMA samples.
#
# Stem             METASPACE dataset name                         Databases
# ---------------  ---------------------------------------------  --------------------------------
# v11l12-038-b1  * V11L12-038-B1.from_smamsi                     HMDB/v4, CoreMetabolome, ChEBI, KEGG
# v11l12-038-d1  * V11L12-038-D1.from_smamsi                     HMDB/v4, CoreMetabolome, ChEBI, KEGG
#                * v11l12-038-d1.from_smamsi.lipid_anno           HMDB/v4, LipidMaps, SwissLipids
# v11l12-109-a1  * V11L12-109_A1.Visium.FMP.220826_smamsi        HMDB/v4
# v11l12-109-b1  * V11L12-109-B1.from_smamsi                     HMDB/v4, CoreMetabolome, ChEBI, KEGG
#                * V11L12-109_B1.Visium.FMP.220826_smamsi         HMDB/v4
# v11l12-109-c1  * V11L12-109_C1.Visium.FMP.220826_smamsi        HMDB/v4


def fdr_label(fdr: float) -> str:
    """Return the filename representation used by METASPACE result exports."""
    return f"{fdr:g}"


def sample_id_to_metaspace_stem(sample_id: str) -> str:
    """Convert an h5mu sample id such as V11L12-038_D1 to v11l12-038-d1."""
    return sample_id.lower().replace("_", "-")


def dataset_prefix(sample_id: str) -> str:
    """Return the local CSV prefix used for one SMA sample."""
    return f"SMA_{sample_id_to_metaspace_stem(sample_id)}"


def annotation_csvs(prefix: str, out_dir: Path, fdr: float) -> list[Path]:
    """Return non-summary, non-aggregated METASPACE CSVs for one sample."""
    label = fdr_label(fdr)
    return sorted(
        path
        for path in out_dir.glob(f"{prefix}.*.fdr{label}.csv")
        if "summary" not in path.name.lower()
        and ".aggregated." not in path.name.lower()
    )


def load_annotation_results(prefix: str, out_dir: Path, fdr: float) -> pd.DataFrame | None:
    """Load cached METASPACE result CSVs for one sample, if present."""
    paths = annotation_csvs(prefix, out_dir, fdr)
    if not paths:
        return None

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        db_label = path.name[len(prefix) + 1 :].rsplit(".fdr", 1)[0]
        if "database" not in df.columns:
            df["database"] = db_label
        df["source_csv"] = str(path)
        frames.append(df)

    results = pd.concat(frames, ignore_index=True)
    if "mz" not in results.columns:
        raise ValueError(f"METASPACE CSVs for {prefix} do not contain an 'mz' column")

    dedup_cols = [
        col
        for col in ("database", "mz", "adduct", "ion", "formula", "moleculeNames")
        if col in results.columns
    ]
    if dedup_cols:
        results = results.drop_duplicates(subset=dedup_cols)

    print(
        f"Loaded {len(results):,} unique annotations for {prefix} "
        f"from {len(paths)} cached CSV(s)."
    )
    return results

# FMP-10 (neurotransmitter) plates: dopamine etc. live in the derivatization-aware
# ".Visium.FMP." dataset, NOT the underivatized "from_smamsi" generic re-run.
FMP10_PLATES = {"v11l12-109", "v11t16-085", "v11t17-102"}

def _is_fmp10(sample_stem: str) -> bool:
    return sample_stem.lower().rsplit("-", 1)[0] in FMP10_PLATES

def find_processed_dataset(sm, sample_stem):
    parts = sample_stem.lower().rsplit("-", 1)
    plate_prefix, section = (parts[0], parts[1]) if len(parts) == 2 else (sample_stem.lower(), "")

    finished = [d for d in sm.datasets(nameMask=plate_prefix)
                if getattr(d, "status", None) == "FINISHED"]
    if section:
        finished = [d for d in finished
                    if f"-{section}" in d.name.lower() or f"_{section}" in d.name.lower()]
    if not finished:
        return None

    if _is_fmp10(sample_stem):
        preferred = [d for d in finished if "fmp" in d.name.lower()]      # derivatization-aware
    else:
        preferred = [d for d in finished if "from_smamsi" in d.name.lower()]
    chosen = (preferred or finished)[0]
    for d in finished:
        print(f"    {d.id} | {d.name}{' <-- using' if d is chosen else ''}")
    return chosen


def export_results(ds, prefix: str, out_dir: Path, fdr: float) -> None:
    """Export METASPACE annotations for all databases on one processed dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = ds.id
    print(f"Exporting annotations from {ds.name} ({dataset_id})...")

    available = [(d.name, d.version) for d in getattr(ds, "database_details", [])]
    if not available:
        raise ValueError(f"{ds.name} has no database_details; cannot export results")

    exported = []
    for db in available:
        try:
            results = ds.results(database=db, fdr=fdr).reset_index()
        except Exception as exc:
            print(f"  skip {db}: {type(exc).__name__}: {exc}")
            continue

        db_label = "_".join(map(str, db)).replace(" ", "_").replace("/", "_")
        out_csv = out_dir / f"{prefix}.{db_label}.fdr{fdr_label(fdr)}.csv"
        results.to_csv(out_csv, index=False)

        print(f"  {db}: {len(results):,} annotations at FDR <= {fdr} -> {out_csv}")
        exported.append(
            {
                "dataset_id": dataset_id,
                "dataset_name": ds.name,
                "database": str(db),
                "fdr": fdr,
                "n_annotations": len(results),
                "csv": str(out_csv),
            }
        )

    if exported:
        summary_csv = out_dir / f"{prefix}.summary.csv"
        pd.DataFrame(exported).to_csv(summary_csv, index=False)
        print(f"Wrote summary: {summary_csv}")


def parse_molecule_names(value) -> str:
    """Parse METASPACE moleculeNames values into a display string."""
    if isinstance(value, list):
        return ", ".join(map(str, value))
    if pd.isna(value):
        return ""
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return str(value)
    if isinstance(parsed, list):
        return ", ".join(map(str, parsed))
    return str(parsed)


def build_peak_annotations(
    msi_var: pd.DataFrame,
    results: pd.DataFrame,
    ppm_match_tol: float,
) -> pd.Series:
    """Return an annotation string per MSI feature index."""
    mz_col = "mz_raw" if "mz_raw" in msi_var.columns else "mz"
    if mz_col not in msi_var.columns:
        raise KeyError("MSI var must contain either 'mz_raw' or 'mz'")

    mz_peaks = pd.to_numeric(msi_var[mz_col], errors="coerce").to_numpy()
    mz_anno = pd.to_numeric(results["mz"], errors="coerce").to_numpy()

    valid_peaks = np.isfinite(mz_peaks)
    valid_anno = np.isfinite(mz_anno)
    if not valid_peaks.any() or not valid_anno.any():
        return pd.Series("", index=msi_var.index, dtype="object", name="annotated")

    peak_indices = np.flatnonzero(valid_peaks)
    anno_indices = np.flatnonzero(valid_anno)
    mz_peaks_valid = mz_peaks[valid_peaks]
    mz_anno_valid = mz_anno[valid_anno]

    ppm_dist = (
        np.abs(mz_anno_valid[:, None] - mz_peaks_valid[None, :])
        / mz_peaks_valid[None, :]
        * 1e6
    )
    anno_match_idx, peak_match_idx = np.where(ppm_dist <= ppm_match_tol)

    annotations = pd.Series("", index=msi_var.index, dtype="object", name="annotated")
    if len(anno_match_idx) == 0:
        return annotations

    matched_results = results.iloc[anno_indices[anno_match_idx]]
    matched = pd.DataFrame(
        {
            "peak_var_name": msi_var.index[peak_indices[peak_match_idx]],
            "ion": matched_results.get(
                "ion", pd.Series("", index=matched_results.index)
            )
            .astype(str)
            .values,
            "moleculeNames": [
                parse_molecule_names(value)
                for value in matched_results.get(
                    "moleculeNames", pd.Series("", index=matched_results.index)
                ).values
            ],
            "database": matched_results.get(
                "database", pd.Series("", index=matched_results.index)
            ).astype(str).values,
            "fdr": pd.to_numeric(
                matched_results.get(
                    "fdr", pd.Series(np.nan, index=matched_results.index)
                ),
                errors="coerce",
            ).values,
        }
    )

    def format_group(group: pd.DataFrame) -> str:
        parts = []
        for row in group.drop_duplicates().itertuples(index=False):
            fdr = "" if pd.isna(row.fdr) else f" FDR={row.fdr:.2f}"
            database = "" if not row.database else f" [{row.database}]"
            name = f" ({row.moleculeNames})" if row.moleculeNames else ""
            parts.append(f"{row.ion}{name}{fdr}{database}".strip())
        return " | ".join(parts)

    grouped = matched.groupby("peak_var_name", sort=False)[
        ["ion", "moleculeNames", "database", "fdr"]
    ].apply(format_group)
    annotations.update(grouped)
    return annotations


def annotate_h5mu(
    sample_id: str,
    export_dir: Path,
    results: pd.DataFrame,
    ppm_match_tol: float,
) -> dict[str, int | str]:
    """Annotate mdata.mod['msi'].var and overwrite the sample h5mu file."""
    h5mu_path = export_dir / f"{sample_id}.h5mu"
    mdata = load_sample(sample_id, export_dir=export_dir)
    if "msi" not in mdata.mod:
        raise KeyError(f"{h5mu_path} has no 'msi' modality")

    msi = mdata.mod["msi"]
    annotations = build_peak_annotations(msi.var, results, ppm_match_tol)
    msi.var["annotated"] = annotations.reindex(msi.var.index).fillna("")

    n_annotated = int(msi.var["annotated"].astype(bool).sum())
    print(
        f"Annotated {n_annotated}/{msi.n_vars} MSI features in {sample_id} "
        f"(PPM_MATCH_TOL={ppm_match_tol})."
    )

    # Keep MuData-level axis metadata consistent with the updated modality var.
    mdata.update()
    mdata.write(h5mu_path)
    print(f"Overwrote {h5mu_path}")

    return {
        "sample_id": sample_id,
        "h5mu": str(h5mu_path),
        "n_msi_features": int(msi.n_vars),
        "n_annotated": n_annotated,
    }


def is_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        elif shell == "TerminalInteractiveShell":
            return False
        else:
            return False
    except Exception:
        return False


def parse_args(notebook: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=H5MU_EXPORT_DIR,
        help="Directory containing aligned SMA .h5mu files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for cached/exported METASPACE CSVs.",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="Optional h5mu sample IDs to annotate. Defaults to all .h5mu files.",
    )
    parser.add_argument("--fdr", type=float, default=FDR)
    parser.add_argument("--ppm-match-tol", type=float, default=PPM_MATCH_TOL)
    parser.add_argument(
        "--cached-only",
        action="store_true",
        help="Only use existing METASPACE CSVs; do not query METASPACE.",
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="Call SMInstance.save_login() before querying METASPACE.",
    )
    if notebook:
        return parser.parse_args([])
    return parser.parse_args()

#%%
def main() -> None:
    NOTEBOOK = is_notebook()
    args = parse_args(notebook=NOTEBOOK)

    export_dir = args.export_dir.expanduser()
    out_dir = args.out_dir.expanduser()

    sample_ids = args.samples or list_samples(export_dir)
    if not sample_ids:
        raise FileNotFoundError(f"No .h5mu files found in {export_dir}")

    sm = None
    if not args.cached_only:
        sm = SMInstance()
        if args.login:
            sm.save_login()

    summaries = []
    for sample_id in sample_ids:
        prefix = dataset_prefix(sample_id)
        sample_stem = sample_id_to_metaspace_stem(sample_id)
        print(f"\n=== {sample_id} ({sample_stem}) ===")

        results = load_annotation_results(prefix, out_dir, args.fdr)
        if results is None and not args.cached_only:
            ds = find_processed_dataset(sm, sample_stem)
            if ds is not None:
                export_results(ds, prefix, out_dir, args.fdr)
                results = load_annotation_results(prefix, out_dir, args.fdr)
            else:
                print(f"No processed METASPACE dataset found for {sample_stem!r}.")

        if results is None:
            print(f"No annotations available for {sample_id}; h5mu left unchanged.")
            summaries.append(
                {
                    "sample_id": sample_id,
                    "h5mu": str(export_dir / f"{sample_id}.h5mu"),
                    "n_msi_features": 0,
                    "n_annotated": 0,
                    "status": "skipped",
                }
            )
            continue

        summary = annotate_h5mu(sample_id, export_dir, results, args.ppm_match_tol)
        summary["status"] = "annotated"
        summaries.append(summary)

    print("\nAnnotation summary:")
    print(pd.DataFrame(summaries))


if __name__ == "__main__":
    main()
