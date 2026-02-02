import argparse
import os
import subprocess
import sys
from dataclasses import asdict
from typing import Any, List, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna HPO for NicheCompass mouse brain multimodal."
    )
    parser.add_argument(
        "--experiment-name",
        default="hpo_nichecompass_mouse_brain_multimodal",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--study-name-prefix",
        default="hpo",
        help="Prefix for Optuna study name (used when --study-name is not set).",
    )
    parser.add_argument(
        "--study-name",
        default=None,
        help="Full Optuna study name. If set (e.g. by launcher), overrides prefix+timestamp.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Path to cache timestamp dir containing model/ and target_model/. "
            "If omitted, the latest outputs/.../multimodal/<timestamp>/ is used."
        ),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=3,
        help=(
            "Max number of trials. With GridSampler the study stops after "
            "all grid points are tried once (e.g. 4 trials for 2×2 search space)."
        ),
    )
    parser.add_argument(
        "--storage",
        default="sqlite:///optuna.db",
        help="Optuna storage URI (required for multi-process runs).",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="CUDA_VISIBLE_DEVICES value for this worker (e.g. '0').",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU ids for launcher mode (e.g. '0,1,2').",
    )
    parser.add_argument(
        "--launch-workers",
        action="store_true",
        help="Launch one worker per GPU in --gpus.",
    )
    parser.add_argument(
        "--parent-run-id",
        default=None,
        help="Use an existing MLflow parent run id.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import mlflow
    import optuna
    import datetime

    from hpo_mouse_brain_utils import (
        TrainConfig,
        TrialParams,
        get_or_create_experiment_id,
        resolve_cache_dir,
        run_trial,
    )

    cache_dir = resolve_cache_dir(args.cache_dir)
    mlflow.set_experiment(args.experiment_name)
    experiment_id = get_or_create_experiment_id(args.experiment_name)
    if args.study_name:
        study_name = args.study_name
    else:
        now = datetime.datetime.now()
        timestamp = now.strftime("%d%m%Y_%H%M%S")
        study_name = f"{args.study_name_prefix}_{timestamp}"

    train_cfg = TrainConfig()
    search_space = {
        "multimodal_layer_series": [True, False],
        "encoder_input_key": ["counts", "pseudocounts"],
        # Contrastive-loss–relevant knobs (objective is target_multimodal_contrastive_loss)
        "lambda_multimodal_contrastive_loss": [10.0, 30.0, 100.0, 300.0],
        "multimodal_temperature": [0.1, 0.2, 0.5, 1.0],
        "multimodal_contrastive_anneal": [False, True],
        "contrastive_logits_pos_ratio": [0.0, 0.125, 0.25],
        "contrastive_logits_neg_ratio": [0.0, 0.125, 0.25],
        # Model capacity for multimodal fusion. None means "keep full GP size".
        "multimodal_embedding_size": [None, 128, 256, 512],
    }

    if args.launch_workers:
        if not args.gpus:
            raise ValueError("--gpus is required when using --launch-workers.")
        if not args.storage:
            raise ValueError("--storage is required when using --launch-workers.")

        gpu_list = _parse_gpus(args.gpus)
        with mlflow.start_run(run_name=study_name) as parent_run:
            parent_run_id = parent_run.info.run_id
            _log_parent_params(
                mlflow=mlflow,
                cache_dir=cache_dir,
                args=args,
                train_cfg=train_cfg,
                search_space=search_space,
                study_name=study_name,
                extra={"n_workers": len(gpu_list)},
            )
            _launch_workers(args, cache_dir, parent_run_id, gpu_list, study_name)
        return

    # GridSampler suggests each combination exactly once; study stops when grid is exhausted.
    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        storage=args.storage,
        load_if_exists=True,
    )

    if args.parent_run_id:
        mlflow.start_run(run_id=args.parent_run_id)
        parent_run_id = args.parent_run_id
    else:
        mlflow.start_run(run_name=study_name)
        parent_run_id = mlflow.active_run().info.run_id
        _log_parent_params(
            mlflow=mlflow,
            cache_dir=cache_dir,
            args=args,
            train_cfg=train_cfg,
            search_space=search_space,
            study_name=study_name,
        )

    def objective(trial: optuna.Trial) -> float:
        params = TrialParams(
            encoder_input_key=trial.suggest_categorical(
                "encoder_input_key", search_space["encoder_input_key"]
            ),
            multimodal_layer_series=trial.suggest_categorical(
                "multimodal_layer_series",
                search_space["multimodal_layer_series"],
            ),
            lambda_multimodal_contrastive_loss=trial.suggest_categorical(
                "lambda_multimodal_contrastive_loss",
                search_space["lambda_multimodal_contrastive_loss"],
            ),
            multimodal_temperature=trial.suggest_categorical(
                "multimodal_temperature",
                search_space["multimodal_temperature"],
            ),
            multimodal_contrastive_anneal=trial.suggest_categorical(
                "multimodal_contrastive_anneal",
                search_space["multimodal_contrastive_anneal"],
            ),
            contrastive_logits_pos_ratio=trial.suggest_categorical(
                "contrastive_logits_pos_ratio",
                search_space["contrastive_logits_pos_ratio"],
            ),
            contrastive_logits_neg_ratio=trial.suggest_categorical(
                "contrastive_logits_neg_ratio",
                search_space["contrastive_logits_neg_ratio"],
            ),
            multimodal_embedding_size=trial.suggest_categorical(
                "multimodal_embedding_size",
                search_space["multimodal_embedding_size"],
            ),
        )
        with mlflow.start_run(
            run_name=f"trial_{trial.number:04d}", nested=True
        ):
            return run_trial(
                params=params,
                cache_dir=cache_dir,
                train_cfg=train_cfg,
                mlflow_experiment_id=experiment_id,
                mlflow_parent_run_id=parent_run_id,
            )

    study.optimize(objective, n_trials=args.n_trials)

    if not args.parent_run_id:
        mlflow.end_run()


def _parse_gpus(gpus: str) -> List[str]:
    return [gpu.strip() for gpu in gpus.split(",") if gpu.strip()]


def _log_parent_params(
    mlflow,
    cache_dir: str,
    args: argparse.Namespace,
    train_cfg: Any,
    search_space: dict,
    study_name: str,
    extra: Optional[dict] = None,
) -> None:
    mlflow.log_param("cache_dir", cache_dir)
    mlflow.log_param("study_name", study_name)
    mlflow.log_param("n_trials", args.n_trials)
    mlflow.log_param("storage", args.storage)
    for key, values in search_space.items():
        mlflow.log_param(f"search_space_{key}", str(values))
    if extra:
        mlflow.log_params(extra)
    mlflow.log_params(asdict(train_cfg))


def _launch_workers(
    args: argparse.Namespace,
    cache_dir: str,
    parent_run_id: str,
    gpus: List[str],
    study_name: str,
) -> None:
    procs = []
    for gpu in gpus:
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--experiment-name",
            args.experiment_name,
            "--study-name",
            study_name,
            "--n-trials",
            str(args.n_trials),
            "--storage",
            args.storage,
            "--parent-run-id",
            parent_run_id,
            "--gpu",
            gpu,
        ]
        if cache_dir:
            cmd.extend(["--cache-dir", cache_dir])
        procs.append(subprocess.Popen(cmd))

    for proc in procs:
        proc.wait()


if __name__ == "__main__":
    main()
