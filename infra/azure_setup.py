"""Create the Azure ML workspace and upload the processed Parquet as a versioned data asset.

Usage:
    python -m infra.azure_setup

Idempotent: an existing workspace or data-asset version is reused, not recreated.
Requires `az login` (authenticates through the Azure CLI) and the resource group below.
"""

import json
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data, Workspace
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import AzureCliCredential

SUBSCRIPTION_ID = "e00b6cd4-18c0-4e2d-87d9-432fef357d37"
RESOURCE_GROUP = "rg-twotower-recsys"
WORKSPACE = "mlw-twotower"
LOCATION = "eastus"

DATA_ASSET = "hm-processed"
DATA_VERSION = "1"
DATA_PATH = Path("data/processed")
CONFIG_PATH = Path(".azureml/config.json")


def ensure_workspace(credential) -> Workspace:
    client = MLClient(credential, SUBSCRIPTION_ID, RESOURCE_GROUP)
    try:
        ws = client.workspaces.get(WORKSPACE)
        print(f"workspace {WORKSPACE} already exists")
    except ResourceNotFoundError:
        print(f"creating workspace {WORKSPACE} (takes a few minutes)...")
        ws = client.workspaces.begin_create(
            Workspace(name=WORKSPACE, location=LOCATION, tags={"project": "twotower-recsys"})
        ).result()
    return ws


def ensure_data_asset(client: MLClient) -> Data:
    try:
        asset = client.data.get(DATA_ASSET, version=DATA_VERSION)
        print(f"data asset {DATA_ASSET}:{DATA_VERSION} already exists")
    except ResourceNotFoundError:
        print(f"uploading {DATA_PATH} as {DATA_ASSET}:{DATA_VERSION}...")
        asset = client.data.create_or_update(
            Data(
                name=DATA_ASSET,
                version=DATA_VERSION,
                path=str(DATA_PATH),
                type=AssetTypes.URI_FOLDER,
                description="H&M transactions/customers/articles as compact Parquet (src/data_loader.py), "
                "plus a 5% customer sample in sample_0.05/",
            )
        )
    return asset


if __name__ == "__main__":
    credential = AzureCliCredential()
    ws = ensure_workspace(credential)

    CONFIG_PATH.parent.mkdir(exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(
            {"subscription_id": SUBSCRIPTION_ID, "resource_group": RESOURCE_GROUP, "workspace_name": WORKSPACE},
            indent=2,
        )
        + "\n"
    )

    client = MLClient(credential, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE)
    asset = ensure_data_asset(client)

    print(f"\nmlflow tracking uri: {ws.mlflow_tracking_uri}")
    print(f"data asset: {asset.name}:{asset.version} -> {asset.path}")
