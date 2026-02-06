import inspect
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple, List

import anndata as ad
import mlflow
from optuna.distributions import CategoricalDistribution

from nichecompass_utils import CustomNicheCompass

DEFAULT_TUNED_HPARAM_KEYS = [
    "encoder_input_key",
    "multimodal_temperature",
    "contrastive_logits_pos_ratio",
    "contrastive_logits_neg_ratio",
    "multimodal_embedding_size",
    "node_batch_size",
]

# HPO convenience override for TrainConfig.n_epochs
HPO_N_EPOCHS = 60

DEFAULT_COUNTS_KEY = "counts"
DEFAULT_ADJ_KEY = "spatial_connectivities"
DEFAULT_GP_NAMES_KEY = "nichecompass_gp_names"
DEFAULT_ACTIVE_GP_NAMES_KEY = "nichecompass_active_gp_names"
DEFAULT_GP_TARGETS_MASK_KEY = "nichecompass_gp_targets"
DEFAULT_GP_TARGETS_CATEGORIES_MASK_KEY = "nichecompass_gp_targets_categories"
DEFAULT_GP_SOURCES_MASK_KEY = "nichecompass_gp_sources"
DEFAULT_GP_SOURCES_CATEGORIES_MASK_KEY = "nichecompass_gp_sources_categories"
DEFAULT_LATENT_KEY = "nichecompass_latent"

@dataclass(frozen=True)
class TrialParams:
    encoder_input_key: Optional[str] = None
    multimodal_layer_series: Optional[bool] = None
    lambda_multimodal_contrastive_loss: Optional[float] = None
    multimodal_temperature: Optional[float] = None
    multimodal_contrastive_anneal: Optional[bool] = None
    contrastive_logits_pos_ratio: Optional[float] = None
    contrastive_logits_neg_ratio: Optional[float] = None
    multimodal_embedding_size: Optional[int] = None
    node_batch_size: Optional[int] = None


@dataclass(frozen=True)
class TrainConfig:
    n_epochs: int = 10
    n_epochs_all_gps: int = 10
    lr: float = 0.001
    lambda_edge_recon: float = 500000.0
    lambda_gene_expr_recon: float = 300.0
    lambda_chrom_access_recon: float = 300.0
    lambda_l1_masked: float = 0.0
    lambda_l1_addon: float = 30.0
    lambda_multimodal_contrastive_loss: float = 100000.0
    edge_batch_size: int = 64
    node_batch_size: int = 256
    use_cuda_if_available: bool = True
    n_sampled_neighbors: int = 4
    multimodal_contrastive_anneal: bool = False
    target_holdout_frac: float = 0.1
    target_holdout_n: Optional[int] = 2000
    target_holdout_seed: int = 0
    target_paired_data: bool = True
    target_encoder_input_key: str = "pseudocounts"
    target_counts_key: str = DEFAULT_COUNTS_KEY
    log_target_multimodal_contrastive: bool = True
    use_early_stopping: bool = False
    verbose: bool = False

def _find_latest_cache_dir(root: str) -> Optional[str]:
    if not os.path.isdir(root):
        return None
    candidates = []
    for entry in os.listdir(root):
        path = os.path.join(root, entry)
        model_path = os.path.join(path, "model", "adata.h5ad")
        target_path = os.path.join(path, "target_model", "target_adata.h5ad")
        if os.path.isfile(model_path) and os.path.isfile(target_path):
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def resolve_cache_dir(cache_dir: Optional[str]) -> str:
    if cache_dir:
        return cache_dir
    default_root = (
        "/home/mcb/users/dmannk/BAKLAVA_base/outputs/"
        "nichecompass_mouse_brain_multimodal/artifacts/multimodal"
    )
    latest = _find_latest_cache_dir(default_root)
    if latest is None:
        raise FileNotFoundError(
            "No cached model inputs found. Provide --cache-dir pointing to a "
            "timestamp folder containing model/adata.h5ad and "
            "target_model/target_adata.h5ad."
        )
    return latest


def load_cached_inputs(cache_dir: str) -> Tuple[ad.AnnData, ad.AnnData]:
    adata_path = os.path.join(cache_dir, "model", "adata.h5ad")
    adata_atac_path = os.path.join(cache_dir, "model", "adata_atac.h5ad")
    if not os.path.isfile(adata_path) or not os.path.isfile(adata_atac_path):
        raise FileNotFoundError(
            "Cache dir must contain model/adata.h5ad and "
            f"model/adata_atac.h5ad: {cache_dir}"
        )
    return ad.read_h5ad(adata_path), ad.read_h5ad(adata_atac_path)


def _trial_params_to_overrides(params: Optional[TrialParams]) -> Dict[str, Any]:
    if params is None:
        return {}
    overrides = asdict(params)
    return {key: value for key, value in overrides.items() if value is not None}


def _hparam_defaults() -> Dict[str, Any]:
    defaults = {}
    for key, spec in get_hparams().items():
        if isinstance(spec, dict) and "default" in spec:
            defaults[key] = spec["default"]
    return defaults


def resolve_hparams(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    resolved = _hparam_defaults()
    if overrides:
        resolved.update(overrides)
    return resolved


def _filter_hparams_for_signature(func, hparams: Dict[str, Any]) -> Dict[str, Any]:
    signature = inspect.signature(func)
    allowed = set(signature.parameters)
    allowed.discard("self")
    allowed.discard("adata")
    allowed.discard("adata_atac")
    allowed.discard("kwargs")
    allowed.discard("trainer_kwargs")
    return {key: value for key, value in hparams.items() if key in allowed}


def filter_model_hparams(hparams: Dict[str, Any]) -> Dict[str, Any]:
    return _filter_hparams_for_signature(CustomNicheCompass.__init__, hparams)


def filter_train_hparams(hparams: Dict[str, Any]) -> Dict[str, Any]:
    return _filter_hparams_for_signature(CustomNicheCompass.train, hparams)


def get_search_space(tuned_keys: Optional[List[str]] = None) -> Dict[str, List[Any]]:
    tuned_keys = tuned_keys or DEFAULT_TUNED_HPARAM_KEYS
    hparams = get_hparams()
    search_space: Dict[str, List[Any]] = {}
    for key in tuned_keys:
        if key not in hparams:
            raise KeyError(f"Unknown HPARAMS key requested for search space: {key}")
        spec = hparams[key]
        dist = spec.get("suggest_distribution") if isinstance(spec, dict) else None
        if dist is None:
            raise ValueError(f"No suggest_distribution defined for HPARAMS key: {key}")
        if isinstance(dist, CategoricalDistribution):
            search_space[key] = list(dist.choices)
        else:
            raise TypeError(
                f"Unsupported distribution type for HPARAMS key '{key}': {type(dist)}"
            )
    return search_space


def build_model(
    adata: ad.AnnData,
    adata_atac: ad.AnnData,
    params: TrialParams,
) -> CustomNicheCompass:
    hparam_overrides = _trial_params_to_overrides(params)
    resolved_hparams = resolve_hparams(hparam_overrides)
    model_hparams = filter_model_hparams(resolved_hparams)
    return CustomNicheCompass(
        adata,
        adata_atac,
        counts_key=DEFAULT_COUNTS_KEY,
        adj_key=DEFAULT_ADJ_KEY,
        gp_names_key=DEFAULT_GP_NAMES_KEY,
        active_gp_names_key=DEFAULT_ACTIVE_GP_NAMES_KEY,
        gp_targets_mask_key=DEFAULT_GP_TARGETS_MASK_KEY,
        gp_targets_categories_mask_key=DEFAULT_GP_TARGETS_CATEGORIES_MASK_KEY,
        gp_sources_mask_key=DEFAULT_GP_SOURCES_MASK_KEY,
        gp_sources_categories_mask_key=DEFAULT_GP_SOURCES_CATEGORIES_MASK_KEY,
        latent_key=DEFAULT_LATENT_KEY,
        **model_hparams,
    )


def get_or_create_experiment_id(
    experiment_name: str,
    artifact_location: Optional[str] = None,
) -> str:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return mlflow.create_experiment(
            experiment_name, artifact_location=artifact_location
        )
    return experiment.experiment_id


def load_cached_targets(cache_dir: str) -> Tuple[ad.AnnData, ad.AnnData]:
    target_rna_path = os.path.join(cache_dir, "target_model", "target_adata.h5ad")
    target_atac_path = os.path.join(
        cache_dir, "target_model", "target_adata_atac.h5ad"
    )
    if not os.path.isfile(target_rna_path) or not os.path.isfile(target_atac_path):
        raise FileNotFoundError(
            "Cache dir must contain target_model/target_adata.h5ad and "
            f"target_model/target_adata_atac.h5ad: {cache_dir}"
        )
    return ad.read_h5ad(target_rna_path), ad.read_h5ad(target_atac_path)


def get_hparams(key=None):
    return CustomNicheCompass.get_hparams(key)


def run_trial(
    params: TrialParams,
    cache_dir: str,
    train_cfg: TrainConfig,
    mlflow_experiment_id: Optional[str] = None,
    optimization_metric: str = "compound_metric",
) -> float:
    adata, adata_atac = load_cached_inputs(cache_dir)
    hparam_overrides = _trial_params_to_overrides(params)
    resolved_hparams = resolve_hparams(hparam_overrides)
    model = build_model(adata, adata_atac, params)

    target_rna, target_atac = load_cached_targets(cache_dir)

    # Log only trial params to the child (current) run.
    mlflow.log_params(hparam_overrides)

    train_kwargs = {key: value for key, value in asdict(train_cfg).items() if value is not None}
    # Ensure HPARAMS defaults / tuned overrides take precedence over TrainConfig.
    train_kwargs.update(filter_train_hparams(resolved_hparams))

    model.train(
        **train_kwargs,
        target_adata=target_rna,
        target_adata_atac=target_atac,
        mlflow_experiment_id=mlflow_experiment_id,
    )

    logs = model.trainer.epoch_logs.get(optimization_metric, [])
    if not logs:
        raise RuntimeError(
            f"{optimization_metric} not logged. Ensure "
            "log_target_multimodal_contrastive=True."
        )
    metric = float(logs[-1])
    return metric
