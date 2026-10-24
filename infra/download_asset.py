"""Download an Azure ML data asset (or some of its files) to a local folder. Works locally and on Colab/Kaggle.

Usage:
    python -m infra.download_asset hm-features:1 data/features/full
    python -m infra.download_asset hm-features:1 data/features/sample_0.05 --only item_text_emb.npy

Authentication: the Azure CLI login if present, otherwise a device-code prompt (Colab/Kaggle have no CLI).
Files are read with the workspace datastore's own credential (account key or SAS token), so no extra storage role assignment is needed.
"""

import argparse
import json
import re
from pathlib import Path

from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential, DeviceCodeCredential
from azure.storage.blob import ContainerClient

from infra.azure_setup import CONFIG_PATH, RESOURCE_GROUP, SUBSCRIPTION_ID, WORKSPACE


def get_client() -> MLClient:
    try:
        credential = AzureCliCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception:
        credential = DeviceCodeCredential()
    if CONFIG_PATH.exists():
        cfg = json.loads(CONFIG_PATH.read_text())
        return MLClient(credential, cfg["subscription_id"], cfg["resource_group"], cfg["workspace_name"])
    return MLClient(credential, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE)


def download(asset: str, dest: Path, only: list[str] | None = None) -> list[Path]:
    client = get_client()
    name, version = asset.split(":")
    uri = client.data.get(name, version=version).path
    match = re.search(r"/datastores/([^/]+)/paths/(.+)$", uri)
    if not match:
        raise ValueError(f"unexpected data asset path: {uri}")
    datastore_name, prefix = match.group(1), match.group(2).rstrip("/") + "/"

    datastore = client.datastores.get(datastore_name, include_secrets=True)
    creds = datastore.credentials  # an account key or a SAS token, depending on how the datastore was set up
    container = ContainerClient(
        account_url=f"https://{datastore.account_name}.blob.core.windows.net",
        container_name=datastore.container_name,
        credential=getattr(creds, "account_key", None) or getattr(creds, "sas_token", None),
    )
    written = []
    for blob in container.list_blobs(name_starts_with=prefix):
        relative = blob.name[len(prefix):]
        if not relative or (only and relative not in only):
            continue
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            container.download_blob(blob.name, max_concurrency=4).readinto(f)
        print(f"{relative}: {blob.size / 1e6:.1f} MB")
        written.append(target)
    if not written:
        raise FileNotFoundError(f"nothing downloaded from {asset} (only={only})")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("asset", help="NAME:VERSION, e.g. hm-features:1")
    parser.add_argument("dest", type=Path)
    parser.add_argument("--only", nargs="*", default=None, help="file names within the asset")
    args = parser.parse_args()
    download(args.asset, args.dest, args.only)
