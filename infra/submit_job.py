"""Run a project module as an Azure ML command job on the CPU cluster, against the hm-processed data asset.

Usage:
    python -m infra.submit_job src.baselines            # submit, stream logs, download outputs to benchmarks/
    python -m infra.submit_job src.baselines --no-wait  # submit and return
    python -m infra.submit_job src.features --requirements requirements-job-torch.txt \
        --register hm-features:1 -- --check --publish-dir {artifacts}

The module receives DATA_DIR (the mounted data asset) and `--out-dir` (a job output folder); whatever it
writes there is downloaded into benchmarks/ when the job finishes. With --register NAME:VERSION, a second
output is registered as that data asset and `{artifacts}` in the module arguments is replaced by its path.
"""

import argparse
import shutil
import tempfile
import time
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import AmlCompute, Environment
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import AzureCliCredential

from infra.azure_setup import CONFIG_PATH, DATA_ASSET, DATA_VERSION
from src.tracking import git_sha

CLUSTER = "cpu-e4"
CLUSTER_SIZE = "Standard_E4s_v3"  # 4 vCPU / 32 GB RAM: fits the 4-vCPU subscription limit
IDLE_SECONDS = 120  # scale back to 0 nodes two minutes after the last job
# Azure ML needs the fully qualified image name; a public image is pulled as-is, with no registry or build.
ENVIRONMENT = Environment(name="python312-slim", image="docker.io/library/python:3.12-slim")


def ensure_cluster(client: MLClient) -> None:
    try:
        client.compute.get(CLUSTER)
    except ResourceNotFoundError:
        print(f"creating cluster {CLUSTER} ({CLUSTER_SIZE}, 0-1 nodes)...")
        client.compute.begin_create_or_update(
            AmlCompute(name=CLUSTER, size=CLUSTER_SIZE, min_instances=0, max_instances=1,
                       idle_time_before_scale_down=IDLE_SECONDS)
        ).result()


def submit(client: MLClient, module: str, module_args: list[str], experiment: str,
           requirements: str = "requirements-job.txt", register: str | None = None):
    outputs = {"results": Output(type=AssetTypes.URI_FOLDER)}
    if register:
        name, version = register.split(":")
        outputs["artifacts"] = Output(type=AssetTypes.URI_FOLDER, name=name, version=version)
    module_args = [a.replace("{artifacts}", "${{outputs.artifacts}}") for a in module_args]
    job = command(
        code=".",
        command=(
            f"pip install -q -r {requirements} && "
            # ${{inputs.*}} is only substituted inside the command, not in environment_variables.
            f"DATA_DIR=${{{{inputs.data}}}} python -m {module} --out-dir ${{{{outputs.results}}}} {' '.join(module_args)}"
        ),
        inputs={"data": Input(type=AssetTypes.URI_FOLDER, path=f"azureml:{DATA_ASSET}:{DATA_VERSION}",
                              mode=InputOutputModes.RO_MOUNT)},
        outputs=outputs,
        environment=ENVIRONMENT,
        environment_variables={"GIT_SHA": git_sha()},
        compute=CLUSTER,
        experiment_name=experiment,
        display_name=module.split(".")[-1],
    )
    return client.jobs.create_or_update(job)


def wait(client: MLClient, job_name: str, poll_seconds: int = 30) -> str:
    """Stream logs; if the SDK's log streamer fails (it intermittently mis-signs blob URLs), poll the status."""
    try:
        client.jobs.stream(job_name)
    except Exception as exc:  # the job itself is unaffected by a streaming failure
        print(f"log streaming failed ({type(exc).__name__}); polling job status instead")
    while (status := client.jobs.get(job_name).status) not in {"Completed", "Failed", "Canceled"}:
        time.sleep(poll_seconds)
    print(f"job {job_name}: {status}")
    return status


def download_outputs(client: MLClient, job_name: str, dest: Path) -> list[Path]:
    with tempfile.TemporaryDirectory() as tmp:
        client.jobs.download(job_name, download_path=tmp, output_name="results")
        files = [p for p in Path(tmp).rglob("*") if p.is_file()]
        dest.mkdir(parents=True, exist_ok=True)
        return [Path(shutil.copy(p, dest / p.name)) for p in files]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("module", nargs="?", help="e.g. src.baselines")
    parser.add_argument("module_args", nargs="*", help="extra arguments passed to the module")
    parser.add_argument("--experiment", default=None, help="defaults to the module name")
    parser.add_argument("--requirements", default="requirements-job.txt")
    parser.add_argument("--register", default=None, help="register a second output as data asset NAME:VERSION")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--attach", default=None, help="wait for / download an already submitted job instead")
    args = parser.parse_args()

    client = MLClient.from_config(AzureCliCredential(), path=str(CONFIG_PATH))
    if args.attach:
        job_name = args.attach
    else:
        ensure_cluster(client)
        job = submit(client, args.module, args.module_args, args.experiment or args.module.split(".")[-1],
                     args.requirements, args.register)
        job_name = job.name
        print(f"submitted {job_name}: {job.studio_url}")

    if not args.no_wait and wait(client, job_name) == "Completed":
        for path in download_outputs(client, job_name, Path("benchmarks")):
            print(f"downloaded {path}")
