#!/usr/bin/env bash
# Build and push the marketsimulator container image to a private registry.

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: build_and_push_simulator.sh

Environment variables:
  REGISTRY_URL       Registry base (e.g. ghcr.io/my-org).
  IMAGE_NAME         Repository name (default: marketsimulator).
  IMAGE_TAG          Tag to push (default: $(git rev-parse --short HEAD)).
  REGISTRY_USERNAME  Optional username for docker login.
  REGISTRY_PASSWORD  Optional password or token for docker login.
  DOCKER_BUILD_ARGS  Optional extra args for docker build (quoted string).

The script builds marketsimulator/Dockerfile using the repo root as context
and pushes the resulting image to $REGISTRY_URL/$IMAGE_NAME:$IMAGE_TAG.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

: "${REGISTRY_URL:?Set REGISTRY_URL to your private registry host (e.g. ghcr.io/your-org)}"
IMAGE_NAME="${IMAGE_NAME:-marketsimulator}"
DEFAULT_TAG=$(git rev-parse --short HEAD 2>/dev/null || echo latest)
IMAGE_TAG="${IMAGE_TAG:-$DEFAULT_TAG}"
DOCKER_BUILD_ARGS="${DOCKER_BUILD_ARGS:-}"

IMAGE_URI="${REGISTRY_URL%/}/${IMAGE_NAME}:${IMAGE_TAG}"

if [[ -n "${REGISTRY_USERNAME:-}" && -n "${REGISTRY_PASSWORD:-}" ]]; then
    echo "Logging into ${REGISTRY_URL%/}..."
    echo "$REGISTRY_PASSWORD" | docker login --username "$REGISTRY_USERNAME" --password-stdin "${REGISTRY_URL%/}"
fi

echo "Building ${IMAGE_URI}..."
docker build \
    -f marketsimulator/Dockerfile \
    -t "$IMAGE_URI" \
    ${DOCKER_BUILD_ARGS:+$DOCKER_BUILD_ARGS} \
    .

echo "Pushing ${IMAGE_URI}..."
docker push "$IMAGE_URI"

echo "Image pushed: $IMAGE_URI"
