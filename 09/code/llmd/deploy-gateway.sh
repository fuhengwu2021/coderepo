#!/bin/bash
# Deploy llm-d Custom API Gateway
# This gateway supports routing by both 'model' and 'owned_by' fields

set -e

NAMESPACE="${NAMESPACE:-llm-d-multi-model}"
GATEWAY_FILE="${GATEWAY_FILE:-$(dirname "$0")/llmd-api-gateway.yaml}"

echo "🚀 Deploying llm-d Custom API Gateway..."
echo "   Namespace: ${NAMESPACE}"
echo "   Gateway file: ${GATEWAY_FILE}"

# Check if namespace exists
if ! kubectl get namespace "${NAMESPACE}" &>/dev/null; then
    echo "❌ Namespace '${NAMESPACE}' does not exist. Creating it..."
    kubectl create namespace "${NAMESPACE}"
fi

# Apply the gateway
echo "📦 Applying Gateway resources..."
kubectl apply -f "${GATEWAY_FILE}"

# Wait for Gateway to be ready
echo "⏳ Waiting for Gateway to be ready..."
kubectl wait --for=condition=ready pod -l app=llmd-gateway -n "${NAMESPACE}" --timeout=120s

# Verify deployment
echo ""
echo "✅ Gateway deployed successfully!"
echo ""
echo "📊 Gateway Status:"
kubectl get pod,svc -l app=llmd-gateway -n "${NAMESPACE}"

echo ""
echo "🔍 To test the Gateway:"
echo "   1. Port-forward: kubectl port-forward -n ${NAMESPACE} svc/llmd-api-gateway 8001:8000"
echo "   2. List models: curl http://localhost:8001/v1/models"
echo "   3. Trigger discovery: curl -X POST http://localhost:8001/admin/discover"
echo "   4. Check routing: curl http://localhost:8001/admin/api/routing"
echo ""
echo "📝 Example request with owned_by:"
echo "   curl -X POST http://localhost:8001/v1/chat/completions \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"model\": \"meta-llama/Llama-3.2-1B-Instruct\", \"owned_by\": \"vllm\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}], \"max_tokens\": 50}'"
