#!/bin/bash
set -e

# Default values
CLUSTER_NAME="${CLUSTER_NAME:-mycluster-gpu}"
IMAGE_NAME="${IMAGE_NAME:-k3s-cuda:v1.33.6-cuda-12.2.0}"
GPUS="${GPUS:-all}"
MODEL_PATH="${MODEL_PATH:-}"

echo "Creating k3d GPU cluster..."
echo "Cluster name: $CLUSTER_NAME"
echo "Image: $IMAGE_NAME"
echo "GPUs: $GPUS"
echo ""

# Check if image exists
if ! docker images | grep -q "$IMAGE_NAME"; then
    echo "ERROR: Image $IMAGE_NAME not found!"
    echo "Please build it first with: ./build.sh"
    exit 1
fi

# Delete existing cluster if it exists
if k3d cluster list | grep -q "$CLUSTER_NAME"; then
    echo "Deleting existing cluster $CLUSTER_NAME..."
    k3d cluster delete "$CLUSTER_NAME" 2>/dev/null || true
    sleep 2
fi

# Build cluster creation command
CMD="k3d cluster create $CLUSTER_NAME \
  --image $IMAGE_NAME \
  --gpus=$GPUS \
  --servers 1 \
  --agents 1"

# Add volume mount if MODEL_PATH is specified
if [ -n "$MODEL_PATH" ]; then
    echo "Mounting model directory: $MODEL_PATH -> /models"
    CMD="$CMD --volume $MODEL_PATH:/models"
fi

echo "Executing: $CMD"
echo ""
eval $CMD

echo ""
echo "✓ Cluster created successfully!"
echo ""
echo "Next steps:"
echo "1. Configure kubectl: ./setup-kubectl.sh"
echo "2. Verify GPU: ./verify-gpu.sh"
