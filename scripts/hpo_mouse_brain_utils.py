import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import anndata as ad
import mlflow

from nichecompass_utils import CustomNicheCompass


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
    encoder_input_key: str
    multimodal_layer_series: bool
    # Hyperparameters that most directly affect the (target) multimodal
    # contrastive loss used as the objective.
    lambda_multimodal_contrastive_loss: float
    multimodal_temperature: float
    multimodal_contrastive_anneal: bool
    contrastive_logits_pos_ratio: float
    contrastive_logits_neg_ratio: float
    multimodal_embedding_size: Optional[int]


@dataclass(frozen=True)
class TrainConfig:
    n_epochs: int = 100
    n_epochs_all_gps: int = 100
    lr: float = 0.001
    lambda_edge_recon: float = 500000.0
    lambda_gene_expr_recon: float = 300.0
    lambda_chrom_access_recon: float = 300.0
    lambda_l1_masked: float = 0.0
    lambda_l1_addon: float = 30.0
    lambda_multimodal_contrastive_loss: float = 100.0
    edge_batch_size: int = 64
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


def build_model(
    adata: ad.AnnData,
    adata_atac: ad.AnnData,
    params: TrialParams,
) -> CustomNicheCompass:
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
        active_gp_thresh_ratio=0.01,
        latent_key=DEFAULT_LATENT_KEY,
        conv_layer_encoder="gatv2conv",
        encoder_input_key=params.encoder_input_key,
        multimodal_layer_series=params.multimodal_layer_series,
        multimodal_embedding_size=params.multimodal_embedding_size,
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


def run_trial(
    params: TrialParams,
    cache_dir: str,
    train_cfg: TrainConfig,
    mlflow_experiment_id: Optional[str] = None,
    optimization_metric: str = "compound_metric",
) -> float:
    adata, adata_atac = load_cached_inputs(cache_dir)
    model = build_model(adata, adata_atac, params)

    target_rna, target_atac = load_cached_targets(cache_dir)

    # Log only trial params to the child (current) run.
    mlflow.log_params(asdict(params))

    model.train(
        n_epochs=train_cfg.n_epochs,
        n_epochs_all_gps=train_cfg.n_epochs_all_gps,
        lr=train_cfg.lr,
        lambda_edge_recon=train_cfg.lambda_edge_recon,
        lambda_gene_expr_recon=train_cfg.lambda_gene_expr_recon,
        lambda_chrom_access_recon=train_cfg.lambda_chrom_access_recon,
        lambda_l1_masked=train_cfg.lambda_l1_masked,
        lambda_l1_addon=train_cfg.lambda_l1_addon,
        lambda_multimodal_contrastive_loss=params.lambda_multimodal_contrastive_loss,
        multimodal_temperature=params.multimodal_temperature,
        multimodal_contrastive_anneal=params.multimodal_contrastive_anneal,
        contrastive_logits_pos_ratio=params.contrastive_logits_pos_ratio,
        contrastive_logits_neg_ratio=params.contrastive_logits_neg_ratio,
        edge_batch_size=train_cfg.edge_batch_size,
        use_cuda_if_available=train_cfg.use_cuda_if_available,
        n_sampled_neighbors=train_cfg.n_sampled_neighbors,
        target_adata=target_rna,
        target_adata_atac=target_atac,
        target_holdout_frac=train_cfg.target_holdout_frac,
        target_holdout_n=train_cfg.target_holdout_n,
        target_holdout_seed=train_cfg.target_holdout_seed,
        target_paired_data=train_cfg.target_paired_data,
        target_encoder_input_key=train_cfg.target_encoder_input_key,
        target_counts_key=train_cfg.target_counts_key,
        log_target_multimodal_contrastive=train_cfg.log_target_multimodal_contrastive,
        use_early_stopping=train_cfg.use_early_stopping,
        verbose=train_cfg.verbose,
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
