"""Resolve local or gs:// paths for BAKLAVA data access without full local copies."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _ensure_gcsfs() -> None:
    try:
        import gcsfs  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "gcsfs is required for gs:// paths. Install with: pip install gcsfs"
        ) from exc


def gcs_uri(relative: str, *, bucket: str | None = None, prefix: str | None = None) -> str:
    """Build gs://bucket/prefix/relative from env defaults."""
    bucket = bucket or os.environ.get("GCS_DATA_BUCKET")
    if not bucket:
        raise ValueError("GCS_DATA_BUCKET is not set")
    prefix = (prefix if prefix is not None else os.environ.get("GCS_DATA_PREFIX", "")).strip("/")
    rel = relative.strip("/")
    key = f"{prefix}/{rel}" if prefix else rel
    return f"gs://{bucket}/{key}"


def resolve_data_path(path: str | Path, *, datapath: str | None = None) -> str:
    """Return a path usable by scanpy/anndata (local absolute or gs://)."""
    path_str = str(path)
    if path_str.startswith("gs://"):
        _ensure_gcsfs()
        return path_str
    if path_str.startswith("/"):
        return path_str
    base = datapath or os.environ.get("DATAPATH", "")
    return str(Path(base) / path_str)


def target_rna_path(dataset: str) -> str:
    """Resolve the projection target h5ad for a named dataset."""
    if override := os.getenv("TARGET_RNA_PATH"):
        return resolve_data_path(override)

    relative_paths = {
        "mouse_spatial_atac_rna_seq": (
            "Spatial_ATAC_RNA/mouse/spatial_omics/spatial_atac_rna_seq_mouse_brain.h5ad"
        ),
        "human_sead_mtg": "Spatial_ATAC_RNA/human/sead_mtg/sead_mtg.h5ad",
    }
    if dataset not in relative_paths:
        raise ValueError(f"Unknown TARGET_DATASET={dataset!r}")

    if os.getenv("GCS_DATA_BUCKET"):
        return gcs_uri(relative_paths[dataset])

    return resolve_data_path(relative_paths[dataset])


def read_h5ad(path: str | Path, **kwargs: Any):
    import scanpy as sc

    resolved = resolve_data_path(path)
    if resolved.startswith("gs://"):
        _ensure_gcsfs()
    return sc.read_h5ad(resolved, **kwargs)
