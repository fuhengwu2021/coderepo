# llm-d Custom API Gateway

This custom API Gateway provides routing support for both `model` and `owned_by` fields, which is not available in the default llm-d InferencePool.

## Features

✅ **Multi-Engine Routing**: Route requests based on both `model` and `owned_by` fields  
✅ **Automatic Service Discovery**: Discovers ModelService instances from Kubernetes  
✅ **Multiple Input Sources**: Reads `owned_by` from:
   - Request body JSON: `{"model": "...", "owned_by": "vllm"}`
   - HTTP header: `x-owned-by: vllm` or `X-Owned-By: vllm`
   - Query parameter: `?owned_by=vllm`  
✅ **Backward Compatibility**: Also supports `engine` and `inference_server` fields  
✅ **Admin API**: Manage routing configuration dynamically  
✅ **Health Checks**: Built-in health and readiness probes  

## Quick Start

### 1. Deploy the Gateway

```bash
export NAMESPACE=llm-d-multi-model
cd /home/fuhwu/workspace/coderepo/09/code/llmd

# Option A: Using deployment script (recommended)
./deploy-gateway.sh

# Option B: Manual deployment
kubectl apply -f llmd-api-gateway.yaml
kubectl wait --for=condition=ready pod -l app=llmd-gateway -n ${NAMESPACE} --timeout=60s
```

### 2. Port-forward to Gateway

```bash
kubectl port-forward -n ${NAMESPACE} svc/llmd-api-gateway 8001:8000
```

### 3. Trigger Service Discovery

The Gateway automatically discovers services on startup, but you can manually trigger discovery:

```bash
curl -X POST http://localhost:8001/admin/discover
```

### 4. Test the Gateway

```bash
# List available models
curl http://localhost:8001/v1/models

# Test with owned_by in request body
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "owned_by": "vllm",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 50
  }'

# Test with owned_by in HTTP header
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-owned-by: vllm" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 50
  }'
```

## API Endpoints

### Standard OpenAI-compatible Endpoints

- `GET /v1/models` - List all available models
- `POST /v1/chat/completions` - Chat completions
- `POST /v1/completions` - Text completions
- `GET /health` - Health check

### Admin Endpoints

- `GET /admin/api/routing` - List all routing mappings
- `POST /admin/api/routing` - Add a routing mapping
- `DELETE /admin/api/routing?model=...&owned_by=...` - Delete a routing mapping
- `POST /admin/discover` - Manually trigger service discovery

## How It Works

1. **Service Discovery**: The Gateway uses Kubernetes API to discover ModelService instances with label `llm-d.ai/role=decode`
2. **Routing Logic**: 
   - Extracts `model` and `owned_by` from request (body, header, or query)
   - Matches against discovered services
   - Routes to the appropriate ModelService
3. **Request Forwarding**: Forwards requests to the target ModelService and returns responses

## Configuration

The Gateway automatically discovers services, but you can also manually configure routing:

```bash
# Add a routing mapping
curl -X POST http://localhost:8001/admin/api/routing \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "owned_by": "vllm",
    "service_name": "ms-llama-32-1b-llm-d-modelservice-decode"
  }'

# List all routing mappings
curl http://localhost:8001/admin/api/routing
```

## Troubleshooting

### Gateway not discovering services

1. Check if services exist:
   ```bash
   kubectl get svc -n ${NAMESPACE} -l llm-d.ai/role=decode
   ```

2. Manually trigger discovery:
   ```bash
   curl -X POST http://localhost:8001/admin/discover
   ```

3. Check Gateway logs:
   ```bash
   kubectl logs -n ${NAMESPACE} -l app=llmd-gateway
   ```

### Service not found error

1. Verify the service name matches the routing config:
   ```bash
   curl http://localhost:8001/admin/api/routing
   ```

2. Check if the ModelService pod is running:
   ```bash
   kubectl get pods -n ${NAMESPACE} -l llm-d.ai/role=decode
   ```

### Permission issues

The Gateway needs RBAC permissions to list services. Verify:

```bash
kubectl get role,rolebinding -n ${NAMESPACE} | grep llmd-api-gateway
```

## Architecture

```
Client Request
    ↓
Custom API Gateway (llmd-api-gateway)
    ↓ (routes by model + owned_by)
ModelService (ms-{model}-llm-d-modelservice-decode)
    ↓
vLLM Container
```

## Comparison with InferencePool

| Feature | InferencePool | Custom API Gateway |
|---------|--------------|-------------------|
| Routing by `model` | ✅ | ✅ |
| Routing by `owned_by` | ❌ | ✅ |
| Automatic discovery | ✅ | ✅ |
| Header-based routing | ❌ | ✅ |
| Admin API | ❌ | ✅ |
| Requires source modification | ❌ | ❌ |

## Files

- `llmd-api-gateway.yaml` - Complete Gateway deployment manifest
- `deploy-gateway.sh` - Deployment script
- `GATEWAY_README.md` - This file

## See Also

- Main documentation: `09/code/tmp.md` (Step 5: Deploy Custom API Gateway)
- Reference implementation: `09/code/k3d/ref.md` (Example 2)
