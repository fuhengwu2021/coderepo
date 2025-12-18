#!/bin/bash
# Run SGLang Llama-4-Scout-17B-16E-Instruct with Docker
# Configuration: 8x H200, 2M context length (2097152 tokens)
# Local Docker run (no Kubernetes)
#
# Usage:
#   ./run-sglang-docker.sh [OPTIONS]
#
# Options:
#   --context-length <num>         Maximum context length in tokens (default: 2097152)
#   --kv-cache-dtype <dtype>       KV cache dtype: auto, fp8_e4m3, fp8_e5m2 (default: auto)
#   --mem-fraction-static <num>    Static memory fraction 0.0-1.0 (default: 0.80)
#   --enable-hierarchical-cache   Enable HiCache (hierarchical cache)
#   --hicache-ratio <num>          HiCache ratio for CPU memory (default: 2.0, requires --enable-hierarchical-cache)
#   --tensor-parallel-size <num>   Tensor parallel size (default: 8)
#   --port <num>                   Server port (default: 8000)
#   --shm-size <size>              Shared memory size (default: 10g)
#   --help                         Show this help message
#
# 使用示例 (Examples):
#
# 1. 默认配置（2M context）:
#    ./run-sglang-docker.sh
#
# 2. 启用 FP8 KV Cache 支持 10M context:
#    ./run-sglang-docker.sh \
#      --context-length 10000000 \
#      --kv-cache-dtype fp8_e5m2
#
# 3. 启用 HiCache 扩展内存:
#    ./run-sglang-docker.sh \
#      --context-length 10000000 \
#      --kv-cache-dtype fp8_e5m2 \
#      --enable-hierarchical-cache \
#      --hicache-ratio 2.0
#
# 4. 调整内存分配:
#    ./run-sglang-docker.sh \
#      --context-length 2097152 \
#      --mem-fraction-static 0.85
#
# 5. 完整配置示例（10M + FP8 + HiCache）:
#    ./run-sglang-docker.sh \
#      --context-length 10000000 \
#      --kv-cache-dtype fp8_e5m2 \
#      --mem-fraction-static 0.80 \
#      --enable-hierarchical-cache \
#      --hicache-ratio 2.0 \
#      --shm-size 128g
#
# 6. 查看帮助信息:
#    ./run-sglang-docker.sh --help

set -e

# Default values
MODEL_PATH="/mnt/co-research/shared-models/hub/models--meta-llama--Llama-4-Scout-17B-16E-Instruct/snapshots/4bd10c4dc905b4000d76640d07a552344146faec"
CONTAINER_NAME="sglang-llama-4-scout"
PORT=8000
IMAGE="lmsysorg/sglang:v0.5.6.post2-runtime"
CONTEXT_LENGTH=2097152
KV_CACHE_DTYPE="auto"
MEM_FRACTION_STATIC=0.80
TENSOR_PARALLEL_SIZE=8
SHM_SIZE="10g"
ENABLE_HIERARCHICAL_CACHE=false
HICACHE_RATIO=2.0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --context-length)
            CONTEXT_LENGTH="$2"
            shift 2
            ;;
        --kv-cache-dtype)
            KV_CACHE_DTYPE="$2"
            shift 2
            ;;
        --mem-fraction-static)
            MEM_FRACTION_STATIC="$2"
            shift 2
            ;;
        --enable-hierarchical-cache)
            ENABLE_HIERARCHICAL_CACHE=true
            shift
            ;;
        --hicache-ratio)
            HICACHE_RATIO="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --shm-size)
            SHM_SIZE="$2"
            shift 2
            ;;
        --help)
            grep -A 30 "^# Usage:" "$0" | head -30
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=== Run SGLang Llama-4-Scout-17B-16E-Instruct with Docker ==="
echo "Configuration:"
echo "  - Model: ${MODEL_PATH}"
echo "  - Context length: ${CONTEXT_LENGTH} tokens"
echo "  - KV cache dtype: ${KV_CACHE_DTYPE}"
echo "  - Memory fraction static: ${MEM_FRACTION_STATIC}"
echo "  - Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "  - HiCache enabled: ${ENABLE_HIERARCHICAL_CACHE}"
if [ "$ENABLE_HIERARCHICAL_CACHE" = true ]; then
    echo "  - HiCache ratio: ${HICACHE_RATIO}"
fi
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
  --shm-size ${SHM_SIZE} \
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
    --tp ${TENSOR_PARALLEL_SIZE} \
    --context-length ${CONTEXT_LENGTH} \
    --mem-fraction-static ${MEM_FRACTION_STATIC} \
    $([ "$KV_CACHE_DTYPE" != "auto" ] && echo "--kv-cache-dtype ${KV_CACHE_DTYPE}") \
    $([ "$ENABLE_HIERARCHICAL_CACHE" = true ] && echo "--enable-hierarchical-cache") \
    $([ "$ENABLE_HIERARCHICAL_CACHE" = true ] && echo "--hicache-ratio ${HICACHE_RATIO}") \
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
    echo "🧪 Test examples:"
    echo "   # Test with 2M context:"
    echo "   ./run-test.sh --backend sglang --input-length 2097152 --output-length 200"
    echo ""
    echo "   # Test with 10M context (if configured):"
    echo "   ./run-test.sh --backend sglang --input-length 10000000 --output-length 200"
    echo ""
    echo "⏳ Waiting for service to be ready (this may take several minutes)..."
    echo "   Check logs with: docker logs -f ${CONTAINER_NAME}"
else
    echo "❌ Failed to start container"
    exit 1
fi
