#%% Load environment variables
from dotenv import load_dotenv, dotenv_values
load_dotenv()

from pprint import pprint
print("Loaded environment variables from .env or env:", end="\n\n")
pprint(dotenv_values())

import os
import sys
sys.path.append(os.environ.get("BAKLAVA_ROOT"))
sys.path.append(os.path.join(os.environ.get("BAKLAVA_ROOT"), "scripts"))

#%%
import argparse
import io
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Optional

def parse_args(notebook: bool = False) -> argparse.Namespace:
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
        "--n-trials-per-gpu",
        type=int,
        default=2,
        help=(
            "Max number of trials per GPU worker. With GridSampler the study "
            "stops after all grid points are tried once (e.g. 4 trials for "
            "2×2 search space)."
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
    parser.add_argument(
        "--mlflow-base-dir",
        default=None,
        help=(
            "Base directory used for MLflow tracking state and artifacts "
            "(e.g. BAKLAVA_base). Also respects MLFLOW_BASE_DIR env var."
        ),
    )
    parser.add_argument(
        "--mlflow-backend",
        default="auto",
        choices=["auto", "sqlite", "file"],
        help=(
            "How to configure MLflow tracking when --mlflow-tracking-uri is not set. "
            "'auto' uses 'file' for worker/parallel contexts and 'sqlite' otherwise."
        ),
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help=(
            "Explicit MLflow tracking URI (e.g. postgresql://..., "
            "mysql+pymysql://..., sqlite:///..., file:///...). "
            "Overrides --mlflow-backend and MLFLOW_TRACKING_URI."
        ),
    )
    parser.add_argument(
        "--allow-sqlite-lock-risk",
        action="store_true",
        help=(
            "Allow SQLite tracking in worker/parallel contexts. By default this is "
            "blocked because concurrent MLflow writers frequently lock SQLite."
        ),
    )
    if notebook:
        return parser.parse_known_args()[0]
    else:
        return parser.parse_args()


class TeeLogger:
    """Capture stdout/stderr to both terminal and a buffer."""
    def __init__(self, original_stream):
        self.original_stream = original_stream
        self.buffer = io.StringIO()
    
    def write(self, data):
        # Write to original stream (terminal)
        self.original_stream.write(data)
        self.original_stream.flush()
        # Write to buffer
        self.buffer.write(data)
    
    def flush(self):
        self.original_stream.flush()
        self.buffer.flush()
    
    def get_output(self):
        return self.buffer.getvalue()


def _dump_hpo_log_to_file(
    study: Any,  # optuna.Study
    stdout_output: str,
    stderr_output: str,
    log_file_path: str,
) -> None:
    """
    Dump comprehensive HPO log including study statistics, all trials, and terminal output.
    
    Args:
        study: Optuna study object
        stdout_output: Captured stdout output
        stderr_output: Captured stderr output
        log_file_path: Path to write the log file
    """
    import optuna
    
    with open(log_file_path, "w") as f:
        f.write(f"Optuna Study: {study.study_name}\n")
        f.write(f"=" * 80 + "\n\n")
        
        # Study summary
        f.write(f"Study Statistics:\n")
        f.write(f"  Number of trials: {len(study.trials)}\n")
        f.write(f"  Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}\n")
        f.write(f"  Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}\n")
        f.write(f"  Number of failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}\n")
        
        if study.best_trial:
            f.write(f"\nBest Trial:\n")
            f.write(f"  Trial number: {study.best_trial.number}\n")
            f.write(f"  Value: {study.best_trial.value}\n")
            f.write(f"  Params:\n")
            for key, value in study.best_trial.params.items():
                f.write(f"    {key}: {value}\n")
        
        f.write(f"\n" + "=" * 80 + "\n")
        f.write(f"All Trials:\n")
        f.write(f"=" * 80 + "\n\n")
        
        # All trials
        for trial in study.trials:
            f.write(f"Trial {trial.number}:\n")
            f.write(f"  State: {trial.state.name}\n")
            if trial.value is not None:
                f.write(f"  Value: {trial.value}\n")
            f.write(f"  Params:\n")
            for key, value in trial.params.items():
                f.write(f"    {key}: {value}\n")
            if trial.user_attrs:
                f.write(f"  User attributes:\n")
                for key, value in trial.user_attrs.items():
                    f.write(f"    {key}: {value}\n")
            if trial.system_attrs:
                f.write(f"  System attributes:\n")
                for key, value in trial.system_attrs.items():
                    f.write(f"    {key}: {value}\n")
            f.write(f"\n")
        
        # Terminal output
        f.write(f"\n" + "=" * 80 + "\n")
        f.write(f"Terminal Output (stdout):\n")
        f.write(f"=" * 80 + "\n\n")
        if stdout_output:
            f.write(stdout_output)
        else:
            f.write("(no stdout output captured)\n")
        
        f.write(f"\n" + "=" * 80 + "\n")
        f.write(f"Terminal Output (stderr):\n")
        f.write(f"=" * 80 + "\n\n")
        if stderr_output:
            f.write(stderr_output)
        else:
            f.write("(no stderr output captured)\n")


def _configure_mlflow_tracking(
    args: argparse.Namespace,
    mlflow,
    mlflow_base_dir: str,
    is_parallel_context: bool,
) -> tuple[str, str]:
    explicit_tracking_uri = args.mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    selected_backend = "uri"

    if explicit_tracking_uri:
        tracking_uri = explicit_tracking_uri
    else:
        selected_backend = args.mlflow_backend
        if selected_backend == "auto":
            selected_backend = "file" if is_parallel_context else "sqlite"

        if selected_backend == "sqlite":
            mlflow_db_path = os.path.join(mlflow_base_dir, "mlflow.db")
            tracking_uri = f"sqlite:///{mlflow_db_path}"
        elif selected_backend == "file":
            mlruns_dir = os.path.join(mlflow_base_dir, "mlruns")
            os.makedirs(mlruns_dir, exist_ok=True)
            tracking_uri = f"file://{mlruns_dir}"
        else:
            raise ValueError(f"Unsupported MLflow backend: {selected_backend}")

    using_sqlite = tracking_uri.startswith("sqlite:")
    if using_sqlite and is_parallel_context and not args.allow_sqlite_lock_risk:
        raise ValueError(
            "SQLite MLflow tracking is disabled for worker/parallel contexts because "
            "it often fails with 'database is locked'. Use --mlflow-backend file, "
            "--mlflow-tracking-uri <postgres/mysql URI>, or pass "
            "--allow-sqlite-lock-risk to override."
        )

    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri} (backend={selected_backend})")
    return tracking_uri, selected_backend


#%%
def main() -> None:
    args = parse_args()
    #args = parse_args(notebook=True); args.study_name_prefix = "hpo_test"; args.experiment_name = "hpo_test"

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import mlflow
    import optuna
    import datetime

    from hpo_mouse_brain_utils import (
        TrainConfig,
        TrialParams,
        get_search_space,
        HPO_N_EPOCHS,
        get_or_create_experiment_id,
        resolve_cache_dir,
        run_trial,
    )

    def _log_param_importances_to_mlflow(study: "optuna.Study") -> None:
        """
        Log Optuna parameter importance figure(s) to the active MLflow run.

        - Prefer Plotly (Optuna default). Try to log PNG; if Plotly image export
          dependencies (e.g. kaleido) are missing, fall back to logging HTML.
        - If Plotly plotting fails entirely, fall back to matplotlib if available.
        """
        # Plotly figure (preferred).
        try:
            importance_fig = optuna.visualization.plot_param_importances(study)
            try:
                mlflow.log_figure(importance_fig, "hyperparameter_importance.png")
            except Exception:
                # Common case: Plotly static export not available (e.g. kaleido missing).
                mlflow.log_text(
                    importance_fig.to_html(), "hyperparameter_importance.html"
                )
            return
        except Exception as plotly_err:
            # Matplotlib fallback (more reliable for PNG in many environments).
            try:
                from optuna.visualization.matplotlib import plot_param_importances

                ax = plot_param_importances(study)
                # Fix clipping of long labels (common in Matplotlib)
                ax.figure.tight_layout()
                mlflow.log_figure(ax.figure, "hyperparameter_importance.png")
                return
            except Exception:
                # Neither Plotly nor Matplotlib succeeded; skip logging.
                pass

    cache_dir = resolve_cache_dir(args.cache_dir)
    if args.study_name:
        study_name = args.study_name
    else:
        now = datetime.datetime.now()
        timestamp = now.strftime("%d%m%Y_%H%M%S")
        study_name = f"{args.study_name_prefix}_{timestamp}"

    # Keep MLflow in one place so `mlflow ui` can consistently discover runs.
    default_base_dir = str(Path(__file__).resolve().parents[2])
    mlflow_base_dir = args.mlflow_base_dir or os.environ.get("MLFLOW_BASE_DIR") or default_base_dir
    mlflow_base_dir = os.path.abspath(mlflow_base_dir)
    os.makedirs(mlflow_base_dir, exist_ok=True)
    is_parallel_context = args.launch_workers or (
        args.parent_run_id is not None and args.gpu is not None
    )
    _tracking_uri, mlflow_backend = _configure_mlflow_tracking(
        args=args,
        mlflow=mlflow,
        mlflow_base_dir=mlflow_base_dir,
        is_parallel_context=is_parallel_context,
    )

    mlflow_artifact_dir = os.path.join(mlflow_base_dir, "mlflow_artifacts", study_name)
    os.makedirs(mlflow_artifact_dir, exist_ok=True)
    experiment_id = get_or_create_experiment_id(
        args.experiment_name,
        artifact_location=os.path.abspath(mlflow_artifact_dir),
    )
    # Now that the experiment exists (with the desired artifact location), activate it.
    mlflow.set_experiment(args.experiment_name)

    train_cfg = TrainConfig(n_epochs=HPO_N_EPOCHS, n_epochs_all_gps=HPO_N_EPOCHS)
    search_space = get_search_space()

    if args.launch_workers:
        if not args.gpus:
            raise ValueError("--gpus is required when using --launch-workers.")
        if not args.storage:
            raise ValueError("--storage is required when using --launch-workers.")

        gpu_list = _parse_gpus(args.gpus)
        total_trials = args.n_trials_per_gpu * len(gpu_list)
        print(
            f"Launching {len(gpu_list)} workers; "
            f"n_trials_per_gpu={args.n_trials_per_gpu}; "
            f"total_trials={total_trials}"
        )
        with mlflow.start_run(run_name=study_name) as parent_run:
            parent_run_id = parent_run.info.run_id
            _log_parent_params(
                mlflow=mlflow,
                cache_dir=cache_dir,
                args=args,
                train_cfg=train_cfg,
                search_space=search_space,
                study_name=study_name,
                extra={
                    "n_workers": len(gpu_list),
                    "total_trials": total_trials,
                    "mlflow_backend": mlflow_backend,
                },
            )
            _launch_workers(args, cache_dir, parent_run_id, gpu_list, study_name)
        return

    # GridSampler suggests each combination exactly once; study stops when grid is exhausted.
    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=args.storage,
        load_if_exists=True, # if study already exists, load it and continue trial count from there
    )

    if args.parent_run_id:
        mlflow.start_run(run_id=args.parent_run_id)
        parent_run_id = args.parent_run_id
    else:
        mlflow.start_run(run_name=study_name)
        parent_run_id = mlflow.active_run().info.run_id
        total_trials = args.n_trials_per_gpu
        print(f"Single worker; n_trials_per_gpu={total_trials}; total_trials={total_trials}")
        _log_parent_params(
            mlflow=mlflow,
            cache_dir=cache_dir,
            args=args,
            train_cfg=train_cfg,
            search_space=search_space,
            study_name=study_name,
            extra={
                "total_trials": total_trials,
                "mlflow_backend": mlflow_backend,
            },
        )

    def objective(trial: optuna.Trial) -> float:
        kwargs = {
            key: trial.suggest_categorical(key, values)
            for key, values in search_space.items()
        }
        params = TrialParams(**kwargs)
        with mlflow.start_run(
            run_name=f"trial_{trial.number:04d}", nested=True
        ) as child_run:
            # Link Optuna <-> MLflow for easy cross-referencing.
            trial.set_user_attr("mlflow_run_id", child_run.info.run_id)
            mlflow.set_tag("optuna_trial_number", trial.number)
            mlflow.set_tag("optuna_study_name", study.study_name)
            return run_trial(
                params=params,
                cache_dir=cache_dir,
                train_cfg=train_cfg,
                mlflow_experiment_id=experiment_id,
                optimization_metric="compound_metric",
            )

    #%%
    # Capture stdout/stderr during optimization
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stdout_logger = TeeLogger(original_stdout)
    stderr_logger = TeeLogger(original_stderr)
    
    try:
        sys.stdout = stdout_logger
        sys.stderr = stderr_logger
        
        # catch=(Exception,) prevents the study from stopping if a trial fails (e.g. NaNs).
        study.optimize(objective, n_trials=args.n_trials_per_gpu, catch=(Exception,))
    finally:
        # Restore original streams
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        stdout_output = stdout_logger.get_output()
        stderr_output = stderr_logger.get_output()

    # Calculate importance
    try:
        _log_param_importances_to_mlflow(study)
    except Exception as e:
        print(f"Error logging parameter importance: {e}")

    # Dump log into file and add to mlflow study
    log_file_path = os.path.join(mlflow_artifact_dir, "hpo_log.txt")
    _dump_hpo_log_to_file(study, stdout_output, stderr_output, log_file_path)
    mlflow.log_artifact(log_file_path, "hpo_log.txt")
    
    if not args.parent_run_id:
        mlflow.end_run()

#%%
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
    mlflow.log_param("n_trials_per_gpu", args.n_trials_per_gpu)
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
    mlflow_base = args.mlflow_base_dir or os.environ.get("MLFLOW_BASE_DIR") or ""
    procs = []
    for gpu in gpus:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        if mlflow_base:
            env["MLFLOW_BASE_DIR"] = mlflow_base
        if args.mlflow_tracking_uri:
            env["MLFLOW_TRACKING_URI"] = args.mlflow_tracking_uri
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--experiment-name",
            args.experiment_name,
            "--study-name",
            study_name,
            "--n-trials-per-gpu",
            str(args.n_trials_per_gpu),
            "--storage",
            args.storage,
            "--parent-run-id",
            parent_run_id,
            "--gpu",
            gpu,
            "--mlflow-backend",
            args.mlflow_backend,
        ]
        if args.allow_sqlite_lock_risk:
            cmd.append("--allow-sqlite-lock-risk")
        if args.mlflow_tracking_uri:
            cmd.extend(["--mlflow-tracking-uri", args.mlflow_tracking_uri])
        if args.mlflow_base_dir:
            cmd.extend(["--mlflow-base-dir", args.mlflow_base_dir])
        if cache_dir:
            cmd.extend(["--cache-dir", cache_dir])
        procs.append(subprocess.Popen(cmd, env=env))

    for proc in procs:
        proc.wait()

#%%
if __name__ == "__main__":
    main()
