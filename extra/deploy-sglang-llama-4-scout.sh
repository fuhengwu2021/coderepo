#!/bin/bash
# Deploy SGLang Llama-4-Scout-17B-16E-Instruct
# Configuration: 8x H200, 2M context length (2097152 tokens)
# Create Secret from environment variable $HF_TOKEN

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_FILE="$SCRIPT_DIR/sglang-llama-4-scout.yaml"

echo "=== Deploy SGLang Llama-4-Scout-17B-16E-Instruct ==="
echo "Configuration: 8x H200, 2M context length (2097152 tokens)"
echo ""

# Check HF_TOKEN environment variable
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN environment variable not set"
    echo ""
    echo "Please set the environment variable first:"
    echo "  export HF_TOKEN='your_token_here'"
    echo ""
    echo "Or:"
    echo "  HF_TOKEN='your_token_here' $0"
    exit 1
fi

echo "✅ HF_TOKEN environment variable detected"
echo ""

# Create or update Secret
echo "📝 Creating/updating Secret: hf-token-secret"
kubectl delete secret hf-token-secret 2>/dev/null || true
kubectl create secret generic hf-token-secret \
  --from-literal=token="$HF_TOKEN"

if [ $? -eq 0 ]; then
    echo "✅ Secret created successfully"
else
    echo "❌ Secret creation failed"
    exit 1
fi

echo ""
echo "📝 Deploying Pod and Service..."
kubectl apply -f "$YAML_FILE"

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Check Pod status:"
echo "   kubectl get pod sglang-llama-4-scout -w"
echo ""
echo "📝 View logs:"
echo "   kubectl logs -f sglang-llama-4-scout"
echo ""
echo "🔗 Access service:"
echo "   kubectl port-forward svc/sglang-llama-4-scout 8000:8000"
echo "   curl http://localhost:8000/health"
echo ""
echo "🧪 Test with 2M context + 200 output:"
echo "   ./run-test.sh --backend sglang --input-length 2097152 --output-length 200"
