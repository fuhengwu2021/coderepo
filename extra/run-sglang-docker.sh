#!/bin/bash
# Run SGLang Llama-4-Scout-17B-16E-Instruct with Docker
# Configuration: 8x H200, 2M context length (2097152 tokens)
# Local Docker run (no Kubernetes)

set -e

MODEL_PATH="/mnt/co-research/shared-models/hub/models--meta-llama--Llama-4-Scout-17B-16E-Instruct/snapshots/4bd10c4dc905b4000d76640d07a552344146faec"
CONTAINER_NAME="sglang-llama-4-scout"
PORT=8000
IMAGE="lmsysorg/sglang:v0.5.6.post2-runtime"

echo "=== Run SGLang Llama-4-Scout-17B-16E-Instruct with Docker ==="
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

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

echo "✅ Model path found: $MODEL_PATH"
echo ""

# Build docker run command
echo "🚀 Starting SGLang container..."
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
  ${IMAGE} \
  python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --host 0.0.0.0 \
    --port 8000 \
    --tp 8 \
    --context-length 2097152 \
    --mem-fraction-static 0.80 \
    --disable-cuda-graph \
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
    echo "   ./run-test.sh --backend sglang --input-length 2097152 --output-length 200"
    echo ""
    echo "⏳ Waiting for service to be ready (this may take several minutes)..."
    echo "   Check logs with: docker logs -f ${CONTAINER_NAME}"
else
    echo "❌ Failed to start container"
    exit 1
fi
