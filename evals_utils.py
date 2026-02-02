from anndata import AnnData
from typing import List

import numpy as np
import jax.numpy as jnp
from jax import jit
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

## cdist and foscttm_moscot from ECLARE/evals_utils.py
@jit
def cdist(
    x: np.ndarray,
    y: np.ndarray,
    metric: str='cosine'
) -> np.ndarray:
    """Compute the pairwise distance matrix between each row of x and y."""
    x2 = jnp.sum(x**2, axis=1, keepdims=True)
    y2 = jnp.sum(y**2, axis=1, keepdims=True)
    xy = jnp.dot(x, y.T)

    if metric == 'euclidean':
        return jnp.sqrt(x2 - 2*xy + y2.T)
    elif metric == 'cosine':
        return 1 - (xy / (jnp.sqrt(x2) * jnp.sqrt(y2.T)))
    elif metric == 'inner':
        return -xy

@jit
def foscttm_moscot( # from https://moscot.readthedocs.io/en/latest/notebooks/tutorials/600_tutorial_translation.html#define-utility-functions
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    c = cdist(x, y)
    foscttm_x = (c < jnp.expand_dims(jnp.diag(c), axis=1)).mean(axis=1)
    foscttm_y = (c < jnp.expand_dims(jnp.diag(c), axis=0)).mean(axis=0)
    foscttm_full = (foscttm_x + foscttm_y) / 2 #jnp.mean(foscttm_x + foscttm_y) / 2
    return foscttm_full


def benchmark_embeddings(
    adata: AnnData,
    batch_key: str,
    label_key: str,
    embedding_obsm_keys: List[str],
    n_jobs: int=6,
) -> None:
    bm = Benchmarker(
        adata,
        batch_key=batch_key,
        label_key=label_key,
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(),
        embedding_obsm_keys=embedding_obsm_keys,
        n_jobs=n_jobs,
    )
    bm.benchmark()
    return bm.results_dict