#!/bin/bash
# 部署 vLLM Llama-3.2-1B-Instruct
# 从环境变量 $HF_TOKEN 创建 Secret

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_FILE="$SCRIPT_DIR/llama-3.2-1b.yaml"

echo "=== 部署 vLLM Llama-3.2-1B-Instruct ==="
echo ""

# 检查 HF_TOKEN 环境变量
if [ -z "$HF_TOKEN" ]; then
    echo "❌ 错误: HF_TOKEN 环境变量未设置"
    echo ""
    echo "请先设置环境变量："
    echo "  export HF_TOKEN='your_token_here'"
    echo ""
    echo "或者："
    echo "  HF_TOKEN='your_token_here' $0"
    exit 1
fi

echo "✅ 检测到 HF_TOKEN 环境变量"
echo ""

# 创建或更新 Secret
echo "📝 创建/更新 Secret: hf-token-secret"
kubectl delete secret hf-token-secret 2>/dev/null || true
kubectl create secret generic hf-token-secret \
  --from-literal=token="$HF_TOKEN"

if [ $? -eq 0 ]; then
    echo "✅ Secret 创建成功"
else
    echo "❌ Secret 创建失败"
    exit 1
fi

echo ""
echo "📝 部署 Pod 和 Service..."
kubectl apply -f "$YAML_FILE"

echo ""
echo "✅ 部署完成！"
echo ""
echo "📊 检查 Pod 状态："
echo "   kubectl get pod vllm-llama-32-1b -w"
echo ""
echo "📝 查看日志："
echo "   kubectl logs -f vllm-llama-32-1b"
echo ""
echo "🔗 访问服务："
echo "   kubectl port-forward svc/vllm-llama-32-1b 8000:8000"
echo "   curl http://localhost:8000/health"
