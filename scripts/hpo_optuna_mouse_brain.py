import argparse
import os
import sys
from dataclasses import asdict

import mlflow
import optuna

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..")
)

from hpo_mouse_brain_utils import (
    TrainConfig,
    TrialParams,
    get_or_create_experiment_id,
    resolve_cache_dir,
    run_trial,
)


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
        "--study-name",
        default="hpo_multimodal_layer_series_encoder_input",
        help="Optuna study name.",
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
        help="Number of trials to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = resolve_cache_dir(args.cache_dir)

    mlflow.set_experiment(args.experiment_name)
    experiment_id = get_or_create_experiment_id(args.experiment_name)

    search_space = {
        "multimodal_layer_series": [True, False],
        "encoder_input_key": ["counts", "pseudocounts"],
    }
    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )

    train_cfg = TrainConfig()

    def objective(trial: optuna.Trial) -> float:
        params = TrialParams(
            encoder_input_key=trial.suggest_categorical(
                "encoder_input_key", search_space["encoder_input_key"]
            ),
            multimodal_layer_series=trial.suggest_categorical(
                "multimodal_layer_series", search_space["multimodal_layer_series"]
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
            )

    with mlflow.start_run(run_name=args.study_name):
        mlflow.log_param("cache_dir", cache_dir)
        mlflow.log_param("study_name", args.study_name)
        mlflow.log_param("n_trials", args.n_trials)
        for key, values in search_space.items():
            mlflow.log_param(f"search_space_{key}", str(values))
        mlflow.log_params(asdict(train_cfg))
        study.optimize(objective, n_trials=args.n_trials)


if __name__ == "__main__":
    main()
