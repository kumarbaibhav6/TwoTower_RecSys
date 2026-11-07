#!/usr/bin/env bash
# Build the serving image, push it to Azure Container Registry and deploy it to Azure Container Apps (CPU, scale to zero).
#
# Usage:
#   infra/deploy_containerapp.sh models/serve/full-main-6ep        # tag defaults to <bundle>-<git sha>
#   infra/deploy_containerapp.sh models/serve/full-main-6ep v2
#
# Idempotent: the registry, environment and app are created on the first run and updated afterwards.
# The image is built locally with Docker (linux/amd64, what Container Apps runs) and pushed: Free Trial subscriptions
# may not use ACR Tasks, so `az acr build` fails with TasksOperationsNotAllowed. On a paid subscription,
# `az acr build -r $ACR -t twotower-api:$TAG --build-arg BUNDLE=$BUNDLE .` builds in Azure instead.
# The app pulls from the registry with its system-assigned managed identity (AcrPull), so no registry password exists.
set -euo pipefail

BUNDLE=${1:?usage: $0 <bundle dir> [tag]}
TAG=${2:-$(basename "$BUNDLE")-$(git rev-parse --short HEAD)}
RG=rg-twotower-recsys
LOCATION=eastus
ACR=acrtwotowerrecsys
ENVIRONMENT=cae-twotower
APP=twotower-api
IMAGE="$ACR.azurecr.io/twotower-api:$TAG"

az extension add --name containerapp --upgrade --only-show-errors

if ! az acr show -n "$ACR" -g "$RG" >/dev/null 2>&1; then
  az acr create -n "$ACR" -g "$RG" -l "$LOCATION" --sku Basic --only-show-errors >/dev/null
fi
docker build --platform linux/amd64 --build-arg BUNDLE="$BUNDLE" -t "$IMAGE" .
az acr login -n "$ACR"
docker push "$IMAGE"

if ! az containerapp env show -n "$ENVIRONMENT" -g "$RG" >/dev/null 2>&1; then
  az containerapp env create -n "$ENVIRONMENT" -g "$RG" -l "$LOCATION" --only-show-errors >/dev/null
fi

if az containerapp show -n "$APP" -g "$RG" >/dev/null 2>&1; then
  az containerapp update -n "$APP" -g "$RG" --image "$IMAGE" --only-show-errors >/dev/null
else
  # 1 vCPU / 2 GiB holds the model, user state and index (~1 GB). One replica max for now, so the load test
  # measures a single replica; min 0 scales to zero when idle (the first request after that is a cold start).
  az containerapp create -n "$APP" -g "$RG" --environment "$ENVIRONMENT" --image "$IMAGE" \
    --registry-server "$ACR.azurecr.io" --registry-identity system \
    --target-port 8000 --ingress external --cpu 1 --memory 2Gi --min-replicas 0 --max-replicas 1 \
    --only-show-errors >/dev/null
fi

echo "https://$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
