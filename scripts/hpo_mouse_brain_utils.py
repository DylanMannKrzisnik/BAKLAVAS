import inspect
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
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
HPO_N_EPOCHS = 5

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
        "nichecompass_mouse_brain_multimodal/stable/multimodal"
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


def find_images_by_pattern(
    base_dir: str,
    pattern: str,
) -> List[str]:
    """
    Find all images matching a pattern in the base directory.
    
    Args:
        base_dir: Base directory to search in
        pattern: Glob pattern to match (e.g., "*target_holdout_umap_epoch_5.png")
    
    Returns:
        List of relative paths to matching images, sorted
    """
    result = subprocess.run(
        ["find", ".", "-name", pattern],
        capture_output=True,
        text=True,
        cwd=base_dir,
    )
    
    image_paths = sorted([
        line.strip()
        for line in result.stdout.strip().split('\n')
        if line.strip()
    ])
    
    return image_paths


def get_trial_name_mapping(parent_run_id: Optional[str] = None) -> Dict[str, str]:
    """
    Get mapping of MLflow run_id to run_name for all child runs.
    
    Args:
        parent_run_id: Parent run ID to query child runs from. If None, uses active run.
    
    Returns:
        Dictionary mapping run_id to run_name (e.g., {"abc123...": "trial_0001"})
    """
    if parent_run_id is None:
        active_run = mlflow.active_run()
        if active_run:
            parent_run_id = active_run.info.run_id
        else:
            return {}
    
    # Get all runs with this parent
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        # Get the parent run to find its experiment_id
        parent_run = client.get_run(parent_run_id)
        experiment_id = parent_run.info.experiment_id
        
        # Search for runs with the parent_run_id tag
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
            max_results=10000,
        )
        
        mapping = {}
        for run in runs:
            mapping[run.info.run_id] = run.info.run_name
        
        return mapping
    except Exception as e:
        print(f"Warning: Could not fetch trial name mapping: {e}")
        return {}


def extract_run_id_from_path(image_path: str) -> Optional[str]:
    """
    Extract MLflow run ID from image path.
    
    Args:
        image_path: Path like "./a6e99895149647e9a174ac9a880c076f/artifacts/umap/image.png"
    
    Returns:
        Run ID (hex string) or None if not found
    """
    parts = Path(image_path).parts
    # Look for a part that looks like a run ID (32 character hex string)
    for part in parts:
        if len(part) == 32 and all(c in '0123456789abcdef' for c in part):
            return part
    return None


def generate_image_viewer_html(
    base_dir: str,
    pattern: str,
    output_filename: str = "image_viewer.html",
    title: Optional[str] = None,
    parent_run_id: Optional[str] = None,
) -> str:
    """
    Generate an HTML file that displays all matching images in a grid layout.
    
    Args:
        base_dir: Base directory to search for images
        pattern: Glob pattern to match (e.g., "*target_holdout_umap_epoch_5.png")
        output_filename: Name of the output HTML file
        title: Optional title for the HTML page (defaults to pattern)
        parent_run_id: Parent MLflow run ID to extract trial names from
    
    Returns:
        Absolute path to the generated HTML file
    """
    image_paths = find_images_by_pattern(base_dir, pattern)
    
    if not image_paths:
        print(f"Warning: No images found matching pattern '{pattern}' in {base_dir}")
        return None
    
    # Get trial name mapping if parent_run_id is provided
    trial_mapping = get_trial_name_mapping(parent_run_id) if parent_run_id else {}
    
    if title is None:
        title = f"Image Viewer - {pattern}"
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
            padding: 20px;
        }}
        .image-container {{
            background: white;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .image-container:hover {{
            transform: scale(1.02);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .image-title {{
            font-size: 12px;
            color: #666;
            margin-bottom: 8px;
            word-break: break-all;
            font-family: monospace;
        }}
        .trial-name {{
            font-size: 14px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 4px;
            font-family: Arial, sans-serif;
        }}
        img {{
            width: 100%;
            height: auto;
            display: block;
            border-radius: 4px;
        }}
        .stats {{
            text-align: center;
            color: #666;
            margin-bottom: 20px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="stats">Total images: {len(image_paths)}</div>
    <div class="image-grid">
"""
    
    # Add each image
    for img_path in image_paths:
        # Extract trial name if mapping is available
        trial_name = None
        if trial_mapping:
            run_id = extract_run_id_from_path(img_path)
            if run_id and run_id in trial_mapping:
                trial_name = trial_mapping[run_id]
        
        # Build HTML with or without trial name
        if trial_name:
            html_content += f"""        <div class="image-container">
            <div class="trial-name">{trial_name}</div>
            <div class="image-title">{img_path}</div>
            <img src="{img_path}" alt="{img_path}" loading="lazy">
        </div>
"""
        else:
            html_content += f"""        <div class="image-container">
            <div class="image-title">{img_path}</div>
            <img src="{img_path}" alt="{img_path}" loading="lazy">
        </div>
"""
    
    html_content += """    </div>
</body>
</html>
"""
    
    # Write HTML file
    output_path = os.path.join(base_dir, output_filename)
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Generated {output_filename} with {len(image_paths)} images")
    return os.path.abspath(output_path)


def generate_image_viewers_for_study(
    artifact_dir: str,
    patterns: Optional[List[str]] = None,
    parent_run_id: Optional[str] = None,
) -> List[str]:
    """
    Generate HTML image viewers for common HPO artifact patterns.
    
    Args:
        artifact_dir: MLflow artifact directory for the study
        patterns: List of image patterns to create viewers for. If None, uses defaults.
        parent_run_id: Parent MLflow run ID to extract trial names from
    
    Returns:
        List of paths to generated HTML files
    """
    if patterns is None:
        patterns = [
            "*target_holdout_umap_epoch_*.png",
            "*loss_curves.png",
            "*embedding_*.png",
        ]
    
    generated_files = []
    
    for pattern in patterns:
        # Create a safe filename from the pattern
        safe_name = pattern.replace("*", "").replace("/", "_").replace(".", "_")
        output_filename = f"image_viewer_{safe_name}.html"
        
        html_path = generate_image_viewer_html(
            base_dir=artifact_dir,
            pattern=pattern,
            output_filename=output_filename,
            title=f"Image Viewer - {pattern}",
            parent_run_id=parent_run_id,
        )
        
        if html_path:
            generated_files.append(html_path)
    
    return generated_files
