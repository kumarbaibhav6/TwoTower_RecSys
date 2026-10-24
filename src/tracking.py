"""MLflow setup that works locally, on Colab/Kaggle and inside Azure ML jobs."""

import os
import resource
import subprocess
import sys

import mlflow


def init_mlflow(experiment: str) -> None:
    """Point MLflow at the Azure ML workspace and select the experiment.

    Inside an Azure ML job the tracking URI and experiment are already set by the platform.
    Elsewhere, MLFLOW_TRACKING_URI wins if set; otherwise the project workspace is used (from
    .azureml/config.json, or the constants in infra/azure_setup.py on Colab/Kaggle where that file is absent).
    Set MLFLOW_TRACKING_URI=file:./mlruns to log locally instead.
    """
    if os.environ.get("AZUREML_RUN_ID"):
        return
    if "MLFLOW_TRACKING_URI" not in os.environ:
        from infra.download_asset import get_client

        client = get_client()
        uri = client.workspaces.get(client.workspace_name).mlflow_tracking_uri
        mlflow.set_tracking_uri(uri)
        print(f"mlflow tracking: workspace {client.workspace_name}", flush=True)
    mlflow.set_experiment(experiment)


def git_sha() -> str:
    """Commit of the running code; Azure ML jobs get it from the GIT_SHA env var set at submission."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return os.environ.get("GIT_SHA", "unknown")


def peak_rss_gb() -> float:
    """Peak resident memory of this process so far (ru_maxrss is bytes on macOS, KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak * 1024 / 1e9
