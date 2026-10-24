"""MLflow setup that works locally, on Colab/Kaggle and inside Azure ML jobs."""

import os
import resource
import subprocess
import sys
from pathlib import Path

import mlflow

AZURE_CONFIG = Path(".azureml/config.json")


def init_mlflow(experiment: str) -> None:
    """Point MLflow at the Azure ML workspace and select the experiment.

    Inside an Azure ML job the tracking URI and experiment are already set by the platform.
    Elsewhere, MLFLOW_TRACKING_URI wins if set; otherwise the workspace in .azureml/config.json is used,
    falling back to a local ./mlruns directory.
    """
    if os.environ.get("AZUREML_RUN_ID"):
        return
    if "MLFLOW_TRACKING_URI" not in os.environ and AZURE_CONFIG.exists():
        from azure.ai.ml import MLClient
        from azure.identity import AzureCliCredential

        client = MLClient.from_config(AzureCliCredential(), path=str(AZURE_CONFIG))
        mlflow.set_tracking_uri(client.workspaces.get(client.workspace_name).mlflow_tracking_uri)
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
