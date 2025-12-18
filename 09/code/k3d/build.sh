#!/bin/bash
set -e

# Default values
K3S_TAG="${K3S_TAG:-v1.33.6-k3s1}"
CUDA_TAG="${CUDA_TAG:-12.2.0-base-ubuntu22.04}"
IMAGE_NAME="${IMAGE_NAME:-k3s-cuda:v1.33.6-cuda-12.2.0}"

echo "Building custom k3s-cuda image..."
echo "K3S_TAG: $K3S_TAG"
echo "CUDA_TAG: $CUDA_TAG"
echo "IMAGE_NAME: $IMAGE_NAME"
echo ""

# Enable BuildKit
export DOCKER_BUILDKIT=1

# Check if buildx is available
if docker buildx version &> /dev/null; then
    echo "Using docker buildx..."
    docker buildx build \
      --build-arg K3S_TAG="$K3S_TAG" \
      --build-arg CUDA_TAG="$CUDA_TAG" \
      -t "$IMAGE_NAME" \
      --load .
else
    echo "Using standard docker build..."
    echo "Note: If you encounter issues with --exclude flag, install docker buildx:"
    echo "  docker plugin install --grant-all-permissions moby/buildx"
    echo ""
    docker build \
      --build-arg K3S_TAG="$K3S_TAG" \
      --build-arg CUDA_TAG="$CUDA_TAG" \
      -t "$IMAGE_NAME" \
      .
fi

echo ""
echo "✓ Image built successfully: $IMAGE_NAME"
echo ""
echo "Verify the image:"
echo "  docker images | grep k3s-cuda"
echo ""
echo "Note: CUDA 12.4.1 also works, but 12.2.0 is tested and recommended"
echo ""
echo "Next step: Create the cluster with ./create-cluster.sh"
