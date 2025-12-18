#!/bin/bash
# Run vLLM Llama-4-Scout-17B-16E-Instruct with Docker
# Configuration: 8x H200, 2M context length (2097152 tokens)
# Local Docker run (no Kubernetes)

set -e

# Use HuggingFace model ID - vLLM will resolve from HF_HOME cache
# The model is already cached, so it won't try to download
MODEL_ID="meta-llama/Llama-4-Scout-17B-16E-Instruct"
CONTAINER_NAME="vllm-llama-4-scout"
PORT=8000
IMAGE="vllm/vllm-openai:v0.12.0"

echo "=== Run vLLM Llama-4-Scout-17B-16E-Instruct with Docker ==="
echo "Configuration: 8x H200, 2M context length (2097152 tokens)"
echo ""

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "⚠️  Container ${CONTAINER_NAME} already exists"
    read -p "Do you want to remove it and start a new one? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Stopping and removing existing container..."
        docker stop ${CONTAINER_NAME} 2>/dev/null || true
        docker rm ${CONTAINER_NAME} 2>/dev/null || true
    else
        echo "ℹ️  Starting existing container..."
        docker start ${CONTAINER_NAME}
        echo ""
        echo "✅ Container started!"
        echo ""
        echo "📝 View logs:"
        echo "   docker logs -f ${CONTAINER_NAME}"
        echo ""
        echo "🔗 Access service:"
        echo "   curl http://localhost:${PORT}/health"
        exit 0
    fi
fi

# Check HF_TOKEN environment variable
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  Warning: HF_TOKEN environment variable not set"
    echo "   The model may require authentication. Set it with:"
    echo "   export HF_TOKEN='your_token_here'"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if HF_HOME is accessible (model should be in cache)
HF_HOME_PATH="/mnt/co-research/shared-models/hub"
if [ ! -d "$HF_HOME_PATH" ]; then
    echo "❌ Error: HF_HOME path does not exist: $HF_HOME_PATH"
    exit 1
fi

echo "✅ Using HuggingFace model ID: $MODEL_ID"
echo "✅ HF_HOME configured: $HF_HOME_PATH"
echo ""

# Build docker run command
echo "🚀 Starting vLLM container..."
echo ""

docker run -d \
  --name ${CONTAINER_NAME} \
  --gpus all \
  --shm-size 10g \
  -p ${PORT}:8000 \
  -v /mnt/co-research/shared-models:/mnt/co-research/shared-models \
  -e HF_HOME=/mnt/co-research/shared-models/hub \
  -e TRANSFORMERS_CACHE=/mnt/co-research/shared-models/hub \
  -e HF_HUB_CACHE=/mnt/co-research/shared-models/hub \
  ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
  --ulimit nofile=65535:65535 \
  --entrypoint python3 \
  ${IMAGE} \
  -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_ID} \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --max-model-len 2097152 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code

if [ $? -eq 0 ]; then
    echo "✅ Container started successfully!"
    echo ""
    echo "📊 Container status:"
    echo "   docker ps | grep ${CONTAINER_NAME}"
    echo ""
    echo "📝 View logs:"
    echo "   docker logs -f ${CONTAINER_NAME}"
    echo ""
    echo "🛑 Stop container:"
    echo "   docker stop ${CONTAINER_NAME}"
    echo ""
    echo "🗑️  Remove container:"
    echo "   docker rm ${CONTAINER_NAME}"
    echo ""
    echo "🔗 Access service:"
    echo "   curl http://localhost:${PORT}/health"
    echo ""
    echo "🧪 Test with 2M context + 200 output:"
    echo "   ./run-test.sh --backend vllm --input-length 2097152 --output-length 200"
    echo ""
    echo "⏳ Waiting for service to be ready (this may take several minutes)..."
    echo "   Check logs with: docker logs -f ${CONTAINER_NAME}"
else
    echo "❌ Failed to start container"
    exit 1
fi
