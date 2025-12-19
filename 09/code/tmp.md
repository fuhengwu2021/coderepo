# Hands-On Examples: LLM serving in Kubernetes with llm-d

This section demonstrates how to achieve the same architecture as Example 1 (different models, same inference engine) using llm-d's production-ready Helm charts and intelligent inference scheduling.

## Testing Guide: llm-d Multi-Model Setup with k3d

This guide helps you test this document using your existing k3d setup.

### ✅ Yes, k3d works with llm-d!

Your existing setup already demonstrates this:
- `09/code/llmd/` - Simple llm-d setup with LLMInferenceService CRDs
- `09/code/k3d/` - GPU-enabled k3d cluster setup

### 🎯 Cluster Strategy

**Important:** This guide creates a **NEW separate cluster** (`llmd-cluster`) for testing llm-d production setup. This keeps it isolated from your existing `mycluster-gpu` cluster which is used for:
- Manual vLLM deployments (`09/code/vllm/`)
- Manual SGLang deployments (`09/code/sglang/`)

This separation ensures:
- ✅ No conflicts between manual deployments and llm-d production setup
- ✅ You can test llm-d without affecting your working deployments
- ✅ Both clusters can run simultaneously (if you have enough resources)
- ✅ Easy cleanup: delete `llmd-cluster` when done testing

### 📍 Quick Path Reference

When following this document, replace paths like this:

| Document references | Actual path on your system |
|-------------------|---------------------------|
| `resources/llm-d/guides/prereq/client-setup` | `/home/fuhwu/workspace/llm-d/guides/prereq/client-setup` |
| `resources/llm-d/guides/prereq/gateway-provider` | `/home/fuhwu/workspace/llm-d/guides/prereq/gateway-provider` |
| `resources/llm-d/guides/inference-scheduling` | `/home/fuhwu/workspace/llm-d/guides/inference-scheduling` |

**Good news!** The llm-d repository is already available at `/home/fuhwu/workspace/llm-d`, so all the guides and scripts referenced in this document are available.

### Cluster Separation Summary

You now have (or will have) **two separate k3d clusters**:

| Cluster Name | Purpose | Location | Gateway |
|--------------|---------|----------|---------|
| `mycluster-gpu` | Manual deployments (vLLM, SGLang) | `09/code/vllm/`, `09/code/sglang/` | Traefik or custom API gateway |
| `llmd-cluster` | llm-d production setup | This guide | Istio/kGateway |

### Switching Between Clusters

```bash
# Work with your existing manual deployments
kubectl config use-context k3d-mycluster-gpu
kubectl get pods -n default

# Switch to llm-d cluster for testing
kubectl config use-context k3d-llmd-cluster
kubectl get pods -n llm-d-multi-model

# List all contexts
kubectl config get-contexts
```

### Managing Both Clusters

```bash
# List all clusters
k3d cluster list

# Start/stop clusters independently
k3d cluster start mycluster-gpu
k3d cluster start llmd-cluster

k3d cluster stop mycluster-gpu
k3d cluster stop llmd-cluster

# Delete llm-d cluster when done testing (keeps mycluster-gpu intact)
k3d cluster delete llmd-cluster
```

### Key Differences: Manual Setup vs llm-d Production

| Component | llm-d (Production) | Manual Setup |
|-----------|---------------------|-------------------|
| **Cluster** | New `llmd-cluster` | Existing `mycluster-gpu` |
| **Gateway** | Istio/kGateway (Gateway API) | Traefik (k3d default) or custom API gateway |
| **Model Deployment** | ModelService Helm charts | LLMInferenceService CRDs or direct Pods |
| **Routing** | InferencePool (automatic discovery) | Manual routing config in api-gateway.py |
| **Complexity** | High (production-ready) | Medium (development/testing) |

### Troubleshooting

#### Issue: Can't find llm-d guides

The guides are already available at `/home/fuhwu/workspace/llm-d`. If you need to update the repository:
```bash
cd /home/fuhwu/workspace/llm-d
git pull
```

#### Issue: Gateway provider installation fails

For k3d, Istio can be resource-intensive. Consider:
- Using kGateway instead (lighter)
- Or using Traefik with manual routing (simpler, but less features)

#### Issue: ModelService Helm charts not found

The ModelService charts might be in a different repository. Check:
- llm-d official documentation
- GitHub releases for chart locations
- OCI registry: `oci://us-central1-docker.pkg.dev/k8s-staging-images/...`

#### Issue: GPU not detected

```bash
# Verify NVIDIA device plugin
kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds

# Check node GPU capacity
kubectl describe node | grep -i gpu
```

**Important Notes:**
- All resources are deployed in the `llm-d-multi-model` namespace, not the default namespace
- Use `kubectl get pods -n llm-d-multi-model` to see deployed pods (not `kubectl get pods` which queries default namespace)
- The cluster must have GPU support and `nvidia` RuntimeClass configured
- `RELEASE_NAME_POSTFIX` cannot contain dots (.) - use hyphens instead (e.g., `llama-32-1b` not `llama-3.2-1b`)
- For k3d clusters, use `kubectl port-forward` to access services from outside the cluster

### Architecture Overview

llm-d provides a production-grade solution using:
- **Inference Gateway (IGW)**: Kubernetes-native gateway with intelligent load balancing
- **InferencePool**: Routes requests to appropriate ModelService instances
- **ModelService**: Helm chart for deploying vLLM model servers
- **Intelligent Scheduler**: Load-aware and prefix-cache aware routing

**Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ HTTP Request with 'model' field
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         Inference Gateway (Kubernetes Gateway API)          │
│  - Load balancing with prefix-cache awareness               │
│  - Intelligent request scheduling                           │
│  - Traffic routing to InferencePool                         │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│              InferencePool (Routing Layer)                  │
│  - Routes requests to appropriate ModelService              │
│  - Model-aware request distribution                         │
│  - Health checking and load balancing                       │
└───────────────┬───────────────────────────┬─────────────────┘
                │                           │
                │                           │
    ┌───────────▼──────────┐    ┌───────────▼─────────┐
    │  ModelService 1      │    │  ModelService 2     │
    │  (Llama-3.2-1B)      │    │  (Qwen2.5-0.5B)     │
    │  vLLM Pods (1x)      │    │  vLLM Pods (1x)     │
    └──────────────────────┘    └─────────────────────┘
```

### Prerequisites

**Set environment variables:**
```bash
export LLMD_HOME=/home/fuhwu/workspace/llm-d
export LLMD_CONFIG_DIR=<path-to-your-config-directory>  # Directory containing values YAML files
```

1. **Install Client Tools:**
   ```bash
   cd ${LLMD_HOME}/guides/prereq/client-setup
   ./install-deps.sh
   ```
   This installs: `helm`, `helmfile`, `kubectl`, `yq`, `git`

```
 🌈 $   export LLMD_HOME=/home/fuhwu/workspace/llm-d
   cd ${LLMD_HOME}/guides/prereq/client-setup
   ./install-deps.sh
Installing yq...
[sudo] password for xxx: 
📦 helm-diff plugin not found. Installing v3.11.0...
Downloading https://github.com/databus23/helm-diff/releases/download/v3.11.0/helm-diff-linux-amd64.tgz
Preparing to install into /home/fuhwu/.local/share/helm/plugins/helm-diff
Installed plugin: diff
📦 helmfile not found. Installing 1.2.1...
✅ All tools installed successfully.
```

2. **Create k3d Cluster for llm-d:**
   ```bash
   # Clean up existing cluster if it exists (optional, if recreating)
   # Check if cluster exists
   k3d cluster list | grep llmd-cluster && echo "Cluster exists, deleting..." && k3d cluster delete llmd-cluster || echo "Cluster does not exist, proceeding to create..."
   
   # Create a new k3d cluster specifically for llm-d
   k3d cluster create llmd-cluster \
     --image k3s-cuda:v1.33.6-cuda-12.2.0-working \
     --gpus=all \
     --servers 1 \
     --agents 2 \
     --volume /raid/models:/models \
     --k3s-arg '--disable=traefik@server:0' \
     --port "8080:80@loadbalancer" \
     --port "8443:443@loadbalancer"
   
   # Merge kubeconfig
   k3d kubeconfig merge llmd-cluster --kubeconfig-merge-default
   
   # Switch to the new cluster context
   kubectl config use-context k3d-llmd-cluster
   
   # Verify cluster is ready
   kubectl get nodes
   
   # Label nodes for GPU assignment (optional, for better Pod scheduling)
   # This helps ensure Pods are scheduled to specific nodes
   kubectl label node k3d-llmd-cluster-agent-0 gpu-preference=cuda-0
   kubectl label node k3d-llmd-cluster-agent-1 gpu-preference=cuda-1
   ```
   
   **Note:** k3d doesn't support assigning specific GPUs to specific nodes during cluster creation. All nodes will have access to all GPUs. To ensure Pods use specific GPUs, use `nodeSelector` in your deployment values (see Step 1 and Step 2 below).

3. **Create Namespace:**
   ```bash
   export NAMESPACE=llm-d-multi-model
   kubectl create namespace ${NAMESPACE}
   ```

4. **Create HuggingFace Token Secret:**
   ```bash
   export HF_TOKEN='your_huggingface_token_here'
   kubectl create secret generic llm-d-hf-token \
     --from-literal="HF_TOKEN=${HF_TOKEN}" \
     --namespace "${NAMESPACE}" \
     --dry-run=client -o yaml | kubectl apply -f -
   ```

5. **Deploy Gateway Provider:**
   ```bash
   cd ${LLMD_HOME}/guides/prereq/gateway-provider
   ./install-gateway-provider-dependencies.sh
   
   # Deploy Istio Gateway Provider
   helmfile apply -f istio.helmfile.yaml
   ```

Results:
```
 🌈 $   ./install-gateway-provider-dependencies.sh
✅ 📜 Base CRDs: Installing...
customresourcedefinition.apiextensions.k8s.io/gatewayclasses.gateway.networking.k8s.io created
customresourcedefinition.apiextensions.k8s.io/gateways.gateway.networking.k8s.io created
customresourcedefinition.apiextensions.k8s.io/grpcroutes.gateway.networking.k8s.io created
customresourcedefinition.apiextensions.k8s.io/httproutes.gateway.networking.k8s.io created
customresourcedefinition.apiextensions.k8s.io/referencegrants.gateway.networking.k8s.io created
✅ 🚪 GAIE CRDs: Installing...
customresourcedefinition.apiextensions.k8s.io/inferenceobjectives.inference.networking.x-k8s.io created
customresourcedefinition.apiextensions.k8s.io/inferencepoolimports.inference.networking.x-k8s.io created
customresourcedefinition.apiextensions.k8s.io/inferencepools.inference.networking.k8s.io created
customresourcedefinition.apiextensions.k8s.io/inferencepools.inference.networking.x-k8s.io created
```

### Step 1: Deploy First Model (Llama-3.2-1B-Instruct)

Create a custom values file for the first model. The file should be created in your config directory:

**File:** `${LLMD_CONFIG_DIR}/llama-3.2-1b-values.yaml`

**Note:** Ensure `LLMD_CONFIG_DIR` environment variable is set (see Prerequisites) before creating the file.

```yaml
multinode: false

modelArtifacts:
  uri: "hf://meta-llama/Llama-3.2-1B-Instruct"
  name: "meta-llama/Llama-3.2-1B-Instruct"
  size: 20Gi
  authSecretName: "llm-d-hf-token"

routing:
  servicePort: 8000
  proxy:
    image: ghcr.io/llm-d/llm-d-routing-sidecar:v0.4.0-rc.1
    connector: nixlv2
    secure: false

decode:
  create: true
  replicas: 1
  extraConfig:
    runtimeClassName: nvidia  # Required for k3d GPU access
    nodeSelector:
      kubernetes.io/hostname: k3d-llmd-cluster-agent-0  # Schedule to agent-0
    securityContext:
      fsGroup: 0  # Set group to root so pod can write to /models/hub (llm-d runs as uid=2000, needs fsGroup to write)
  containers:
  - name: "vllm"
    image: ghcr.io/llm-d/llm-d-cuda:v0.4.0  # Updated to v0.4.0
    modelCommand: vllmServe
    args:
      - "--disable-uvicorn-access-log"
    env:
      - name: CUDA_VISIBLE_DEVICES
        value: "0"  # Use GPU 0 on agent-0
    resources:
      limits:
        nvidia.com/gpu: "1"
        memory: 8Gi
      requests:
        nvidia.com/gpu: "1"
        memory: 6Gi
    mountModelVolume: false  # Using hostPath volume from k3d mount (/raid/models on host → /models in container)
    volumeMounts:
    - name: models
      mountPath: /models
      readOnly: false  # Allow vLLM to write cache to /models/hub
  volumes:
  - name: models
    hostPath:
      path: /models  # k3d maps /raid/models (host) to /models (container)
      type: Directory

prefill:
  create: false
```

**Deploy:**

```bash
cd ${LLMD_HOME}/guides/inference-scheduling

# Create directory and copy custom values file
# Note: helmfile expects ms-inference-scheduling/values.yaml, so we'll use that path
mkdir -p ms-inference-scheduling
cp ${LLMD_CONFIG_DIR}/llama-3.2-1b-values.yaml ms-inference-scheduling/values.yaml

# Disable Prometheus monitoring to avoid ServiceMonitor CRD requirement
# Backup original file and disable Prometheus monitoring
cp gaie-inference-scheduling/values.yaml gaie-inference-scheduling/values.yaml.bak
# Use yq to disable Prometheus (yq should be installed from prerequisites)
yq eval '.inferenceExtension.monitoring.prometheus.enabled = false' -i gaie-inference-scheduling/values.yaml

# Deploy first model with custom release name
# Note: RELEASE_NAME_POSTFIX cannot contain dots (.) - use hyphens instead
RELEASE_NAME_POSTFIX=llama-32-1b \
helmfile apply -n ${NAMESPACE}

# Wait for pods to be ready (this may take several minutes for model download)
kubectl wait --for=condition=ready pod -l llm-d.ai/role=decode -n ${NAMESPACE} --timeout=600s
```

### Step 2: Deploy Second Model (Qwen2.5-0.5B-Instruct)

**Important:** The second model should use the **same** Gateway and InferencePool as the first model. Only deploy the ModelService for the second model, not a new Gateway/InferencePool.

Create a custom values file for the second model. The file should be created in your config directory:

**File:** `${LLMD_CONFIG_DIR}/qwen2.5-0.5b-values.yaml`

```yaml
multinode: false

modelArtifacts:
  uri: "hf://Qwen/Qwen2.5-0.5B-Instruct"  # HuggingFace model path
  name: "Qwen/Qwen2.5-0.5B-Instruct"
  size: 2Gi
  authSecretName: ""  # Qwen2.5-0.5B-Instruct is not gated, no token needed

routing:
  servicePort: 8000
  proxy:
    image: ghcr.io/llm-d/llm-d-routing-sidecar:v0.4.0-rc.1
    connector: nixlv2
    secure: false

decode:
  create: true
  replicas: 1
  extraConfig:
    runtimeClassName: nvidia  # Required for k3d GPU access
    nodeSelector:
      kubernetes.io/hostname: k3d-llmd-cluster-agent-1  # Schedule to agent-1
    securityContext:
      fsGroup: 0  # Set group to root so pod can write to /models/hub (llm-d runs as uid=2000, needs fsGroup to write)
    volumes:
    - name: models
      hostPath:
        path: /models  # k3d maps /raid/models (host) to /models (container)
        type: Directory
  containers:
  - name: "vllm"
    image: ghcr.io/llm-d/llm-d-cuda:v0.4.0  # Updated to v0.4.0
    modelCommand: vllmServe
    args:
      - "--disable-uvicorn-access-log"
      # Note: Qwen2.5-0.5B-Instruct doesn't require --trust-remote-code
      - "--gpu-memory-utilization"
      - "0.9"
    env:
      - name: CUDA_DEVICE_ORDER
        value: "PCI_BUS_ID"  # Ensure consistent GPU ordering
      # Note: Removed CUDA_VISIBLE_DEVICES to let Kubernetes GPU scheduling handle it
      # The nodeSelector ensures pod runs on agent-1, and nvidia.com/gpu resource request
      # will allocate one GPU. vLLM will use the allocated GPU automatically.
      - name: HF_HOME
        value: "/models/hub"  # Now writable after permission fix
      - name: TRANSFORMERS_CACHE
        value: "/models/hub"
      - name: HF_HUB_CACHE
        value: "/models/hub"
    resources:
      limits:
        nvidia.com/gpu: "1"
        memory: 8Gi
      requests:
        nvidia.com/gpu: "1"
        memory: 4Gi
    mountModelVolume: true  # Use default model volume mount for HuggingFace models

prefill:
  create: false
```

**Deploy:**

```bash
# Update values file for the second model
# Note: helmfile expects ms-inference-scheduling/values.yaml
cp ${LLMD_CONFIG_DIR}/qwen2.5-0.5b-values.yaml ms-inference-scheduling/values.yaml

# Prometheus should already be disabled from previous step
# Deploy second model with different release name
# Note: RELEASE_NAME_POSTFIX cannot contain dots (.) - use hyphens instead
# IMPORTANT: This will deploy infra, gaie, and ms, but we only need ms.
# The existing Gateway and InferencePool will work with both models.
RELEASE_NAME_POSTFIX=qwen2-5-0-5b \
helmfile -e default -f helmfile.yaml.gotmpl --namespace ${NAMESPACE} apply

# Clean up duplicate Gateway and InferencePool (keep only one shared instance)
helm uninstall infra-qwen2-5-0-5b -n ${NAMESPACE} 2>/dev/null || true
helm uninstall gaie-qwen2-5-0-5b -n ${NAMESPACE} 2>/dev/null || true

# Wait for pods to be ready (this may take several minutes for model download)
kubectl wait --for=condition=ready pod -l llm-d.ai/role=decode -n ${NAMESPACE} --timeout=600s
```

### Step 3: Configure InferencePool for Multi-Model Routing

The InferencePool automatically discovers ModelService instances and routes requests based on the `model` field. 

**Note:** llm-d's InferencePool supports plugin-based customization through `pluginsCustomConfig`. The default configuration routes by `model` field only. To route by both `model` and `owned_by` (inference engine type), you would need to:

1. **Use custom plugins** - Create a custom filter plugin that reads the `owned_by` field from:
   - **Request body** (JSON field) - requires parsing request body
   - **Request header** (e.g., `x-owned-by` or `X-Owned-By`) - requires header reading capability
   - Then filter endpoints based on labels matching the `owned_by` value
   - This requires modifying the `llm-d-inference-scheduler` source code

2. **Use a custom API Gateway** - Deploy a custom API Gateway (like the one in `09/code/k3d/ref.md`) that routes based on both `model` and `owned_by` fields (from body or header) before requests reach the InferencePool.

**Available Plugin Types:**
- `prefill-header-handler` - Handles prefill-specific headers (adds `X-Prefiller-Host-Port` for P/D disaggregation)
- `prefill-filter` / `decode-filter` - Filters endpoints based on request type
- `queue-scorer` - Scores endpoints based on queue depth
- `kv-cache-utilization-scorer` - Scores based on KV cache usage
- `prefix-cache-scorer` - Scores based on prefix cache hit rate
- `max-score-picker` / `random-picker` - Picks endpoint based on scores
- `slo-aware-profile-handler` - Switches profile based on SLO headers (e.g., `x-prediction-based-scheduling: true`)

**Using `owned_by` in Request Headers:**

While `prefill-header-handler` is designed for P/D disaggregation (adding `X-Prefiller-Host-Port`), the plugin system does support reading custom headers. However:

1. **Default plugins don't support `owned_by` header routing** - The existing filter plugins (`prefill-filter`, `decode-filter`) filter by request type, not by custom headers like `x-owned-by`.

2. **Possible approach**: You could potentially create a custom filter plugin that:
   - Reads `x-owned-by` or `X-Owned-By` header from the request
   - Filters endpoints based on labels matching the `owned_by` value
   - This would require modifying `llm-d-inference-scheduler` source code

3. **✅ Custom API Gateway (Implemented)**: A custom API Gateway has been created at `09/code/llmd/llmd-api-gateway.yaml` that:
   - Reads `owned_by` from request body, HTTP header (`x-owned-by`), or query parameters
   - Automatically discovers ModelService instances from Kubernetes
   - Routes to different ModelServices based on `model` and `owned_by`
   - Works without modifying llm-d source code
   - See **Step 5: Deploy Custom API Gateway** below for deployment instructions

**Example**: See `llm-d/guides/predicted-latency-based-scheduling/README.md` for how custom headers (`x-prediction-based-scheduling`, `x-slo-ttft-ms`) are used with `slo-aware-profile-handler`.

See `llm-d/guides/wide-ep-lws/manifests/inferencepool.values.yaml` for a custom plugin configuration example.

Create or update the InferencePool configuration:

**File:** `inferencepool-multi-model-values.yaml`

```yaml
provider:
  name: istio  # or kgateway, gke, etc.

inferencePool:
  apiVersion: inference.networking.k8s.io/v1
  metadata:
    name: multi-model-pool
  spec:
    # InferencePool automatically discovers ModelService instances
    # and routes based on model field in requests
```

**Deploy InferencePool:**

```bash
helm install multi-model-pool \
  -n ${NAMESPACE} \
  -f inferencepool-multi-model-values.yaml \
  --set "provider.name=istio" \
  --set "inferenceExtension.monitoring.prometheus.enable=true" \
  oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/inferencepool \
  --version v1.2.0-rc.1
```

### Step 4: Deploy HTTPRoute

Create an HTTPRoute to expose the InferencePool:

**File:** `httproute-multi-model.yaml`

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: multi-model-route
  namespace: ${NAMESPACE}
spec:
  parentRefs:
  - name: gateway  # Your Gateway name
    namespace: istio-system
  rules:
  - matches:
    - path:
        type: PathPrefix
        value: /
    backendRefs:
    - name: multi-model-pool-epp
      port: 8000
```

**Deploy:**

```bash
kubectl apply -f httproute-multi-model.yaml -n ${NAMESPACE}
```

### Step 5: Deploy Custom API Gateway (Optional - for owned_by routing)

**Note:** This step is optional. If you only need routing by `model` field, the InferencePool (deployed in Step 1) is sufficient. Deploy this custom API Gateway only if you need to route by both `model` and `owned_by` fields.

The custom API Gateway provides:
- ✅ Routing by both `model` and `owned_by` fields
- ✅ Automatic service discovery from Kubernetes
- ✅ Support for `owned_by` in request body, HTTP header (`x-owned-by`), or query parameters
- ✅ Unified interface for all models

**Deploy the Custom API Gateway:**

**Option A: Using deployment script (Recommended)**

```bash
export NAMESPACE=llm-d-multi-model
cd ${LLMD_CONFIG_DIR}

# Deploy using the script
./deploy-gateway.sh
```

**Option B: Manual deployment**

```bash
export NAMESPACE=llm-d-multi-model

# Deploy the Gateway (includes ConfigMap, Deployment, ServiceAccount, Role, RoleBinding, and Service)
kubectl apply -f ${LLMD_CONFIG_DIR}/llmd-api-gateway.yaml

# Wait for Gateway to be ready
kubectl wait --for=condition=ready pod -l app=llmd-gateway -n ${NAMESPACE} --timeout=60s

# Verify Gateway is running
kubectl get pod,svc -l app=llmd-gateway -n ${NAMESPACE}
```

**Trigger Service Discovery:**

The Gateway automatically discovers services on startup, but you can manually trigger discovery:

```bash
# Port-forward to Gateway
kubectl port-forward -n ${NAMESPACE} svc/llmd-api-gateway 8001:8000

# In another terminal, trigger service discovery
curl -X POST http://localhost:8001/admin/discover

# Check routing configuration
curl http://localhost:8001/admin/api/routing
```

**Test the Gateway:**

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

**Gateway Features:**

1. **Automatic Service Discovery**: Discovers ModelService instances from Kubernetes using label selector `llm-d.ai/role=decode`
2. **Multiple Input Sources**: Reads `owned_by` from:
   - Request body JSON field: `{"model": "...", "owned_by": "vllm"}`
   - HTTP header: `x-owned-by: vllm` or `X-Owned-By: vllm`
   - Query parameter: `?owned_by=vllm`
3. **Backward Compatibility**: Also supports `engine` and `inference_server` fields
4. **Admin API**: 
   - `GET /admin/api/routing` - List all routing mappings
   - `POST /admin/api/routing` - Add a routing mapping
   - `DELETE /admin/api/routing?model=...&owned_by=...` - Delete a routing mapping
   - `POST /admin/discover` - Manually trigger service discovery

**Integration with InferencePool:**

You can use the custom API Gateway **instead of** or **in front of** the InferencePool:
- **Option A**: Use Gateway only (bypass InferencePool) - Gateway routes directly to ModelServices
- **Option B**: Use Gateway → InferencePool → ModelServices - Gateway routes to InferencePool, which then routes to ModelServices (if InferencePool supports owned_by routing in the future)

### Step 6: Test Multi-Model Routing

**For k3d clusters, use port-forward to access the service:**

```bash
export NAMESPACE=llm-d-multi-model

# Option 1: Port-forward to Custom API Gateway (recommended if deployed)
kubectl port-forward -n ${NAMESPACE} svc/llmd-api-gateway 8001:8000

# In another terminal, test the API
curl -X GET http://localhost:8001/v1/models

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

# Option 2: Port-forward directly to ModelService pod (for direct testing)
POD_NAME=$(kubectl get pods -n ${NAMESPACE} -l llm-d.ai/role=decode | grep llama-32-1b | awk '{print $1}')
kubectl port-forward -n ${NAMESPACE} pod/${POD_NAME} 8002:8000

# In another terminal, test the API
curl -X GET http://localhost:8002/v1/models

# Option 3: Port-forward to InferencePool Gateway (through InferencePool routing)
kubectl port-forward -n ${NAMESPACE} svc/infra-llama-32-1b-inference-gateway-istio 8080:80
curl -X GET http://localhost:8080/v1/models
```

**Or use the Gateway service directly (if accessible):**

```bash
# Get Gateway service ClusterIP
export NAMESPACE=llm-d-multi-model
GATEWAY_IP=$(kubectl get svc -n ${NAMESPACE} infra-llama-32-1b-inference-gateway-istio -o jsonpath='{.spec.clusterIP}')
echo "Gateway IP: ${GATEWAY_IP}"

# List Available Models (from within cluster)
curl http://${GATEWAY_IP}/v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "meta-llama/Llama-3.2-1B-Instruct",
      "object": "model",
      "created": 0,
      "owned_by": "vllm"
    },
    {
      "id": "Qwen/Qwen2.5-0.5B-Instruct",
      "object": "model",
      "created": 0,
      "owned_by": "vllm"
    }
  ]
}
```

**3. Request to Llama Model:**

```bash
curl http://${GATEWAY_IP}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 100
  }'
```

**4. Request to Qwen2.5-0.5B-Instruct Model:**

```bash
curl http://${GATEWAY_IP}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 100
  }'
```

### Key Advantages of llm-d Approach

1. **Production-Ready**: Tested and benchmarked configurations
2. **Intelligent Routing**: Prefix-cache aware and load-aware balancing
3. **Automatic Discovery**: InferencePool automatically discovers ModelService instances
4. **Monitoring**: Built-in Prometheus metrics and Grafana dashboards
5. **Scalability**: Easy to add more models or scale replicas
6. **Multi-Hardware Support**: Works with NVIDIA, AMD, Intel XPU, Google TPU
7. **Advanced Features**: Supports prefill/decode disaggregation, expert parallelism, etc.

**Customization Options:**
- **Plugin System**: llm-d's InferencePool supports plugin-based customization via `pluginsCustomConfig`. You can configure custom filters, scorers, and pickers to customize routing behavior.
- **Available Plugins**: Filters (`prefill-filter`, `decode-filter`), Scorers (`queue-scorer`, `kv-cache-utilization-scorer`, `prefix-cache-scorer`), Pickers (`max-score-picker`, `random-picker`), and Handlers (`prefill-header-handler`, `pd-profile-handler`).
- **Example**: See `llm-d/guides/wide-ep-lws/manifests/inferencepool.values.yaml` for custom plugin configuration.

**Limitations:**
- **Default Routing**: The default InferencePool configuration routes based on `model` field only, not by inference engine type (`owned_by` field).
- **Multi-Engine Routing**: To route by both `model` and `owned_by` (e.g., same model with vLLM vs SGLang), you have three options:
  1. **✅ Custom API Gateway (Recommended)**: Use the custom API Gateway deployed in Step 5 (`09/code/llmd/llmd-api-gateway.yaml`) that automatically discovers services and routes based on both `model` and `owned_by` fields. This works without modifying llm-d source code.
  2. **Custom Plugin**: Create a custom filter plugin that reads `owned_by` from request body or header (requires modifying `llm-d-inference-scheduler` source code).
  3. **Reference Implementation**: See `09/code/k3d/ref.md` Example 2 for another custom API Gateway implementation.

## Summary

**Current Status:**
- ✅ Cluster `llmd-cluster` is running with 3 nodes (1 server + 2 agents)
- ✅ All pods are deployed in `llm-d-multi-model` namespace
- ✅ **Correct Architecture**: 1 Gateway + 1 InferencePool + 2 ModelService pods
- ✅ First model (llama-32-1b) is running (2/2 Ready) and **accepting curl requests** ✓
- ✅ Second model (qwen2-5-0-5b) is running (2/2 Ready) and **accepting curl requests** ✓
- ✅ Gateway and HTTPRoute are configured
- ✅ InferencePool automatically discovers ModelService instances via label selector `llm-d.ai/inferenceServing=true`
- ✅ Models accessible: `/raid/models` (host) → `/models` (k3d container) → `/models` (pod via hostPath)
- ✅ **API Testing**: First model successfully accepts curl requests via port-forward

**Model Storage Configuration:**
- **Host path**: `/raid/models` (where models are stored on host machine)
- **k3d mount**: `/raid/models:/models` (mounted when creating cluster with `--volume /raid/models:/models`)
- **Pod access**: Use HuggingFace path `hf://Qwen/Qwen2.5-0.5B-Instruct` or local path `/models/<model-name>` for local models
- **Volume config**: Add `hostPath` volume in `decode.volumes` section of values.yaml to mount `/models` from k3d container
- **Volume mount**: Set `readOnly: false` to allow vLLM to write cache files to `/models/hub`

**Testing the API:**
- ✅ First model (llama-32-1b) successfully accepts curl requests
- ✅ Tested endpoints: `/v1/models` and `/v1/completions`
- ✅ Direct pod port-forward works: `kubectl port-forward pod/<pod-name> 8002:8000`

**Reference:** 
- See `/home/fuhwu/workspace/coderepo/09/code/k3d/ref.md` for manual deployment examples using kubectl (without llm-d)
- See `/home/fuhwu/workspace/coderepo/09/code/vllm/` for manual vLLM deployment examples
- See `/home/fuhwu/workspace/coderepo/09/code/vllm/llama-3.2-1b.yaml` for reference configurations

**To check pods:**
```bash
export NAMESPACE=llm-d-multi-model
kubectl get pods -n ${NAMESPACE}  # NOT just "kubectl get pods" (which queries default namespace)
```

**To test the API:**
```bash
export NAMESPACE=llm-d-multi-model
# Get pod name
POD_NAME=$(kubectl get pods -n ${NAMESPACE} -l llm-d.ai/role=decode | grep llama-32-1b | awk '{print $1}')

# Port-forward to pod
kubectl port-forward -n ${NAMESPACE} pod/${POD_NAME} 8002:8000

# In another terminal, test
curl -X GET http://localhost:8002/v1/models
curl -X POST http://localhost:8002/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.2-1B-Instruct", "prompt": "Hello", "max_tokens": 10}'
```

**Key Fixes Applied:**
1. ✅ Added `runtimeClassName: nvidia` to `extraConfig` for GPU access in k3d
2. ✅ Changed `RELEASE_NAME_POSTFIX` from `llama-3.2-1b` to `llama-32-1b` (no dots allowed in DNS names)
3. ✅ Moved `nodeSelector` to `extraConfig.nodeSelector` (correct ModelService chart schema)
4. ✅ Disabled Prometheus monitoring to avoid ServiceMonitor CRD requirement
5. ✅ Configured `hostPath` volume for local model access (`/raid/models` on host → `/models` in pods)
6. ✅ Set `readOnly: false` on volume mount to allow vLLM cache writes
7. ✅ Cleaned up duplicate Gateway/InferencePool deployments (keep only one shared instance)

**Successfully Tested:**
- ✅ First model (llama-32-1b) accepts curl requests
- ✅ `/v1/models` endpoint returns model list
- ✅ `/v1/completions` endpoint returns inference results
- ✅ Direct pod port-forward works: `kubectl port-forward pod/<pod-name> 8002:8000`

### Comparison: k3d vs llm-d

| Feature | k3d (Manual) | llm-d (Production) |
|---------|--------------|---------------------|
| **Setup Complexity** | Manual YAML files | Helm charts (automated) |
| **Routing** | Custom API Gateway | Inference Gateway (K8s native) |
| **Load Balancing** | Basic round-robin | Intelligent (prefix-cache aware) |
| **Monitoring** | Manual setup | Built-in Prometheus/Grafana |
| **Scaling** | Manual pod management | HPA-ready, autoscaling support |
| **Multi-Model** | Manual service mapping | Automatic discovery |
| **Production Features** | Limited | Full production stack |


