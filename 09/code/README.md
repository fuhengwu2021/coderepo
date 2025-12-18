# Production LLM Serving Stack

This directory contains code examples for building a complete production LLM serving stack, including Kubernetes deployment configurations for vLLM.

## Directory Structure

```
09/code/
├── vllm/                    # vLLM server deployment configurations
│   ├── llama-3.2-1b.yaml    # Llama-3.2-1B-Instruct model deployment (deployed)
│   ├── phi-tiny-moe.yaml    # Phi-tiny-MoE-instruct model deployment
│   ├── api-gateway.yaml     # API Gateway deployment (auto-routing)
│   ├── api-gateway.py       # Gateway Python code (stored in ConfigMap)
│   ├── deploy-llama-3.2-1b.sh
│   ├── deploy-gateway.sh    # Gateway deployment script
│   ├── test-api.sh          # API testing script
│   └── README.md            # vLLM deployment documentation
└── *.py                     # Python 代码示例
```

## Complete Setup Guide: From k3d to vLLM Deployment

### 1. Install k3d

```bash
# Install k3d
curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash

# Verify installation
k3d --version
```

### 2. Build k3s-cuda Image (for GPU Support)

```bash
# Build custom k3s image with CUDA support
docker build -t k3s-cuda:v1.33.6-cuda-12.2.0 \
  -f - <<EOF
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y curl
RUN curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION=v1.33.6 sh -
EOF

# Or use pre-built image if available
# docker pull k3s-cuda:v1.33.6-cuda-12.2.0
```

### 3. Create k3d Cluster with GPU Support

```bash
# Create cluster with GPU support and volume mounts
k3d cluster create mycluster-gpu \
  --image k3s-cuda:v1.33.6-cuda-12.2.0-working \
  --gpus=all \
  --servers 1 \
  --agents 1 \
  --volume /raid/models:/models

# Merge kubeconfig
k3d kubeconfig merge mycluster-gpu --kubeconfig-merge-default

# Verify cluster
kubectl get nodes
```

### 4. Add Additional Agent Node (Optional)

If you need to run multiple models on different nodes for isolation, you can add more agent nodes:

```bash
# Add a new agent node (k3d node create doesn't support --gpus directly)
# First create the node
k3d node create agent-1 \
  --cluster mycluster-gpu \
  --role agent

# Then manually reconfigure the container to add GPU support
# Stop and remove the container
docker stop k3d-agent-1-0
docker rm k3d-agent-1-0

# Recreate with GPU and volume support
# Get the image ID first
IMAGE_ID=$(docker images k3s-cuda:v1.33.6-cuda-12.2.0-working --format "{{.ID}}")

docker run -d \
  --name k3d-agent-1-0 \
  --hostname k3d-agent-1-0 \
  --network k3d-mycluster-gpu \
  --privileged \
  --tmpfs /run \
  --tmpfs /var/run \
  -e K3S_TOKEN=LmYHFPGciNataclGfjAI \
  -e K3S_URL=https://k3d-mycluster-gpu-server-0:6443 \
  -e K3S_KUBECONFIG_OUTPUT=/output/kubeconfig.yaml \
  -v /raid/models:/models \
  -v /home/fuhwu/workspace/distributedai/resources/vllm:/vllm \
  --label k3d.cluster=mycluster-gpu \
  --label k3d.role=agent \
  --gpus all \
  --restart unless-stopped \
  $IMAGE_ID \
  agent --with-node-id

# Wait for node to be ready and GPU to be detected
kubectl get nodes -w
# Wait ~20-30 seconds for device plugin to detect GPU
# Note: Node name will have a suffix (e.g., k3d-agent-1-0-56098497) due to --with-node-id
NEW_NODE=$(kubectl get nodes -o json | jq -r '.items[] | select(.metadata.name | startswith("k3d-agent-1-0")) | .metadata.name' | head -1)
kubectl get nodes $NEW_NODE -o json | jq -r '.status.capacity."nvidia.com/gpu"'
```

```
 🚀 $kubectl get nodes
NAME                         STATUS   ROLES                  AGE     VERSION
k3d-agent-1-0-56098497       Ready    <none>                 40s     v1.33.6+k3s1
k3d-mycluster-gpu-agent-0    Ready    <none>                 7d21h   v1.33.6+k3s1
k3d-mycluster-gpu-server-0   Ready    control-plane,master   7d21h   v1.33.6+k3s1
```

**Note:** 
- Replace the `K3S_TOKEN` with your actual cluster token (get it from existing agent node: `docker inspect k3d-mycluster-gpu-agent-0 | jq -r '.[0].Config.Env[]' | grep K3S_TOKEN`).
- Use the correct image ID if the image name doesn't work: `docker images k3s-cuda:v1.33.6-cuda-12.2.0-working --format "{{.ID}}"`.
- **Important:** Add `--with-node-id` flag to the agent command to avoid "duplicate hostname" error.
- After creating the container, wait ~20-30 seconds for the device plugin to detect and report GPU resources.
- If GPU doesn't appear, check device plugin pod: `kubectl get pods -n kube-system | grep nvidia-device-plugin`.

### 5. Configure Storage for k3d

```bash
# Configure local-path-provisioner to use /raid/tmpdata
kubectl patch configmap local-path-config -n kube-system --type merge -p '{
  "data": {
    "config.json": "{\"nodePathMap\":[{\"node\":\"DEFAULT_PATH_FOR_NON_LISTED_NODES\",\"paths\":[\"/raid/tmpdata/k3s-storage\"]}]}"
  }
}'

# Add hostPath volume mount to local-path-provisioner
kubectl patch deployment local-path-provisioner -n kube-system --type json -p '[
  {
    "op": "add",
    "path": "/spec/template/spec/containers/0/volumeMounts/-",
    "value": {
      "name": "raid-storage",
      "mountPath": "/raid/tmpdata"
    }
  },
  {
    "op": "add",
    "path": "/spec/template/spec/volumes/-",
    "value": {
      "name": "raid-storage",
      "hostPath": {
        "path": "/raid/tmpdata",
        "type": "DirectoryOrCreate"
      }
    }
  }
]'

# Restart local-path-provisioner
kubectl rollout restart deployment local-path-provisioner -n kube-system
```

### 6. Set Up HuggingFace Token (for Gated Models)

```bash
# Export HF_TOKEN environment variable
export HF_TOKEN='your_huggingface_token_here'

# Verify it's set
echo $HF_TOKEN
```

### 7. Deploy vLLM Models

#### 7.1 Deploy Llama-3.2-1B-Instruct

##### Option A: Using Deployment Script (Recommended)

```bash
cd vllm/

# Deploy Llama-3.2-1B-Instruct
./deploy-llama-3.2-1b.sh
```

##### Option B: Manual Deployment

```bash
# Create HF token secret (if not already created)
kubectl create secret generic hf-token-secret \
  --from-literal=token="$HF_TOKEN"

# Deploy model
kubectl apply -f vllm/llama-3.2-1b.yaml

# Check status
kubectl get pod vllm-llama-32-1b -w
```

#### 7.2 Deploy Phi-tiny-MoE-instruct

Phi-tiny-MoE-instruct is deployed to a separate agent node for isolation. The model should be available at `/raid/models/Phi-tiny-MoE-instruct` on the host.

##### Prerequisites

- Model directory exists: `/raid/models/Phi-tiny-MoE-instruct`
- Additional agent node created (see step 4)
- GPU available on the target node

##### Option A: Using Deployment Script (Recommended)

```bash
cd vllm/

# Deploy Phi-tiny-MoE-instruct
./deploy-phi-tiny-moe.sh
```

The script will:
- Check if the cluster exists
- Verify the model directory exists
- Deploy the model to the new agent node (`k3d-agent-1-0-56098497`)
- Wait for the pod to be ready

##### Option B: Manual Deployment

```bash
# Deploy model (will be scheduled to new agent node via nodeSelector)
kubectl apply -f vllm/phi-tiny-moe.yaml

# Wait for pod to be ready (model loading may take several minutes)
kubectl wait --for=condition=Ready pod -l app=vllm,model=phi-tiny-moe --timeout=600s

# Check status
kubectl get pod -l app=vllm,model=phi-tiny-moe -o wide
```

##### Configuration Details

The `phi-tiny-moe.yaml` includes:
- **Node Selection**: Deploys to `k3d-agent-1-0-56098497` via `nodeSelector`
- **Image**: Uses `vllm/vllm-openai:v0.12.0` (same as llama)
- **Model Path**: `/models/Phi-tiny-MoE-instruct` (mapped from `/raid/models/Phi-tiny-MoE-instruct`)
- **GPU**: Requires 1 GPU with sufficient memory
- **HF Cache**: Configured to use `/models/hub` for HuggingFace cache

##### Troubleshooting

If the pod fails to start:

1. **Check GPU memory availability**:
   ```bash
   # Check GPU memory on the target node
   kubectl describe node k3d-agent-1-0-56098497 | grep -A 5 nvidia.com/gpu
   ```

2. **Check model path**:
   ```bash
   # Verify model exists in k3d container
   docker exec k3d-agent-1-0-56098497 ls -la /models/Phi-tiny-MoE-instruct
   ```

3. **Check pod logs**:
   ```bash
   kubectl logs -l app=vllm,model=phi-tiny-moe --tail=50
   ```

4. **Common issues**:
   - **GPU memory insufficient**: Reduce `--gpu-memory-utilization` in the YAML (default is 0.9)
   - **Model path not found**: Ensure model is at `/raid/models/Phi-tiny-MoE-instruct` on host
   - **Node not ready**: Check if the agent node has GPU support and is ready

### 8. Test the Deployments

#### 8.1 Test Llama-3.2-1B-Instruct

```bash
# Wait for pod to be ready
kubectl wait --for=condition=Ready pod/vllm-llama-32-1b --timeout=300s

# Port forward
kubectl port-forward svc/vllm-llama-32-1b 8000:8000

# In another terminal, test health endpoint
curl http://localhost:8000/health

# Test models endpoint
curl http://localhost:8000/v1/models

# Test chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "Which is more beautiful: Gaussian Integral and Euler Formula, and why in one sentence?"}]
  }'
```

#### 8.2 Test Phi-tiny-MoE-instruct

```bash
# Wait for pod to be ready
kubectl wait --for=condition=Ready pod -l app=vllm,model=phi-tiny-moe --timeout=600s

# Port forward (use different local port to avoid conflict with llama on 8000)
kubectl port-forward svc/vllm-phi-tiny-moe-service 9876:8000

# In another terminal, test health endpoint
curl http://localhost:9876/health

# Test models endpoint
curl http://localhost:9876/v1/models

# Test chat completion
curl http://localhost:9876/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Phi-tiny-MoE-instruct",
    "messages": [{"role": "user", "content": "Which is more beautiful: Gaussian Integral and Euler Formula, and why in one sentence?"}],
    "max_tokens": 100
  }'
```

**Note:** Both services use port 8000 internally, but we use different local ports (8000 for llama, 9876 for phi-tiny-moe) when port-forwarding to avoid conflicts.

#### 8.3 Test via API Gateway (Recommended)

If you've deployed the API Gateway (see step 9), you can test both models through a single endpoint:

```bash
# Port forward Gateway
kubectl port-forward svc/vllm-api-gateway 8000:8000

# Test llama (automatically routed)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "Which is more beautiful: Gaussian Integral and Euler Formula, and why in one sentence?"}]
  }'

# Test phi-tiny-moe (automatically routed)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Phi-tiny-MoE-instruct",
    "messages": [{"role": "user", "content": "Which is more beautiful: Gaussian Integral and Euler Formula, and why in one sentence?"}]
  }'
```

The Gateway automatically routes requests to the correct service based on the `model` field in the request body.

### 9. Deploy API Gateway (Optional but Recommended)

The API Gateway provides a unified entry point that automatically routes requests to the correct vLLM service based on the `model` field in the request body. This eliminates the need to know which service to call for each model.

#### Why Use API Gateway?

- **Unified Entry Point**: Single endpoint for all models
- **Automatic Routing**: Routes requests based on `model` field in request body
- **No GPU Required**: Gateway is lightweight and can run on any agent node
- **Easy to Scale**: Add new models by updating the routing configuration

#### Deployment Steps

```bash
cd vllm/

# Option A: Using deployment script (Recommended)
./deploy-gateway.sh

# Option B: Manual deployment
kubectl apply -f api-gateway.yaml

# Wait for Gateway to be ready
kubectl wait --for=condition=Ready pod/vllm-api-gateway --timeout=60s

# Check Gateway status
kubectl get pod,svc -l app=vllm-gateway
```

#### Gateway Configuration

The Gateway automatically routes requests based on the `model` field. Current model mappings are defined in `api-gateway.py`:

```python
MODEL_TO_SERVICE = {
    "meta-llama/Llama-3.2-1B-Instruct": "vllm-llama-32-1b",
    "/models/Phi-tiny-MoE-instruct": "vllm-phi-tiny-moe-service",
    "Phi-tiny-MoE-instruct": "vllm-phi-tiny-moe-service",
}
```

To add a new model, simply update this mapping in `api-gateway.py` and redeploy.

#### Production Deployment: Exposing Gateway to External Access

Expose the Gateway service using Ingress with TLS for production-grade access.

**Step 1: Create TLS Certificate**

Create a self-signed certificate for `localhost` (or your domain):

```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /tmp/tls.key \
  -out /tmp/tls.crt \
  -subj "/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,DNS:*.localhost,IP:127.0.0.1,IP:::1"

kubectl create secret tls vllm-api-tls \
  --cert=/tmp/tls.crt \
  --key=/tmp/tls.key
```

**Step 2: Deploy Ingress with Traefik (Recommended)**

k3d comes with Traefik by default, and the loadbalancer forwards to Traefik on ports 80/443. Use Traefik for the simplest setup:

```bash
# Apply Traefik Ingress configuration
kubectl apply -f vllm/ingress-tls-traefik.yaml

# Test access via HTTPS
curl -k https://localhost/health
curl -k https://localhost/v1/models
curl -k https://localhost/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Why Traefik:**
- ✅ No cluster recreation needed
- ✅ k3d loadbalancer already configured
- ✅ Works on standard HTTPS port 443
- ✅ Default k3d installation

**Alternative: Port-Forward for Testing**

For quick testing without Ingress:

```bash
# Port-forward Gateway directly
kubectl port-forward svc/vllm-api-gateway 8000:8000

# Test
curl http://localhost:8000/health
```

**Note:** For production with a real domain, replace `localhost` in the Ingress configuration with your domain (e.g., `api.example.com`) and configure DNS accordingly.


#### Testing Gateway

```bash
# Port forward Gateway
kubectl port-forward svc/vllm-api-gateway 8000:8000

# Test health endpoint
curl http://localhost:8000/health

# Test with automatic routing (Gateway will route to llama service)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Test with different model (Gateway will route to phi-tiny-moe service)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Phi-tiny-MoE-instruct",
    "messages": [{"role": "user", "content": "What is a mixture of experts?"}]
  }'

# Or use the test script
./test-api.sh http://localhost:8000
```

#### Gateway Architecture

```
Client Request
    ↓
API Gateway (Unified Entry Point)
    ↓
Parse 'model' field from request body
    ↓
Route to corresponding Service
    - "meta-llama/Llama-3.2-1B-Instruct" → vllm-llama-32-1b:8000
    - "/models/Phi-tiny-MoE-instruct" → vllm-phi-tiny-moe-service:8000
    ↓
转发请求到对应的 vLLM Pod
    ↓
返回响应
```

#### Node Placement

The Gateway is configured to run on **Agent nodes** (work nodes), not the control-plane node:

- **Deployed on**: Agent nodes (via `nodeAffinity` excluding control-plane)
- **Resource requirements**: Low (256Mi memory, 100m CPU)
- **No GPU needed**: Gateway only forwards requests, doesn't run models
- **Can share nodes**: Gateway can run alongside model pods on the same node

Current deployment:
- Gateway runs on `k3d-mycluster-gpu-agent-0` (same node as llama pod)
- No need to create a dedicated node for Gateway

### 10. Monitor and Debug

```bash
# View pod logs
kubectl logs -f vllm-llama-32-1b

# Describe pod for events
kubectl describe pod vllm-llama-32-1b

# Check resource usage
kubectl top pod vllm-llama-32-1b

# Check service
kubectl get svc vllm-llama-32-1b
```

## Common Issues and Solutions

### Issue 1: Disk Pressure Preventing Pod Scheduling

**Symptoms:**
```bash
kubectl get nodes
# 节点显示 DiskPressure=True
kubectl describe pod <pod-name>
# 显示: 0/1 nodes are available: 1 node(s) had disk-pressure
```

**Cause:**
- `/dev/sda1` disk usage exceeds 85% (kubelet default threshold)
- Docker images and containers consume significant space
- k3d's overlay filesystem is also on `/dev/sda1`

**Solutions:**

1. **Clean Docker resources (Recommended, quick and effective)**
```bash
# Check disk usage
df -h /dev/sda1

# Check Docker usage
docker system df

# Clean unused resources (can free ~200-300GB)
docker image prune -a -f      # Clean unused images
docker container prune -f      # Clean stopped containers
docker volume prune -f        # Clean unused volumes

# Or use script
./cleanup-docker.sh
```

2. **Configure local-path-provisioner to use /raid**
```bash
# Change storage path to /raid/tmpdata (see step 4)
# This stores PVC data in /raid instead of /dev/sda1
```

3. **Add toleration (Temporary solution, not recommended for production)**
```yaml
spec:
  tolerations:
  - key: node.kubernetes.io/disk-pressure
    operator: Exists
    effect: NoSchedule
```

### Issue 2: vLLM Pod Error `libcuda.so.1: cannot open shared object file`

**Symptoms:**
```bash
kubectl logs vllm-pod
# ImportError: libcuda.so.1: cannot open shared object file: No such file or directory
```

**Cause:**
- Pod missing GPU runtime configuration
- `runtimeClassName: nvidia` not specified

**Solution:**
```yaml
spec:
  runtimeClassName: nvidia  # 必须添加
  containers:
  - name: vllm-server
    resources:
      limits:
        nvidia.com/gpu: 1   # 必须添加
      requests:
        nvidia.com/gpu: 1   # 必须添加
```

### Issue 3: Gated Model Access Denied

**Symptoms:**
```bash
kubectl logs vllm-pod
# OSError: You are trying to access a gated repo. 
# Access to model ... is restricted. You must have access to it and be authenticated.
```

**Cause:**
- Model is a gated repo (e.g., meta-llama/Llama-3.2-1B-Instruct)
- HuggingFace token required but not configured

**Solutions:**

1. **Create Secret (Do not hardcode token)**
```bash
# 从环境变量创建
kubectl create secret generic hf-token-secret \
  --from-literal=token="$HF_TOKEN"
```

2. **在 Pod 中引用 Secret**
```yaml
env:
- name: HF_TOKEN
  valueFrom:
    secretKeyRef:
      name: hf-token-secret
      key: token
```

3. **Use deployment script (handles automatically)**
```bash
./vllm/deploy-llama-3.2-1b.sh  # Script automatically creates Secret
```

### Issue 4: Service Name Contains Dot Causing Creation Failure

**Symptoms:**
```bash
kubectl apply -f vllm-llama-3.2-1b.yaml
# Error: metadata.name: Invalid value: "vllm-llama-3.2-1b": 
# a DNS-1035 label must consist of lower case alphanumeric characters or '-'
```

**Cause:**
- Kubernetes Service names must comply with DNS-1035 specification
- Cannot contain dots (`.`), only lowercase letters, numbers, and hyphens allowed

**Solution:**
```yaml
# ❌ Wrong
metadata:
  name: vllm-llama-3.2-1b

# ✅ Correct
metadata:
  name: vllm-llama-32-1b  # Replace dot with hyphen
```

### Issue 5: Pod Evicted Even After Adding Toleration

**Symptoms:**
```bash
kubectl get pod
# NAME               STATUS    RESTARTS   AGE
# test-pod           Evicted   0          5m
```

**Cause:**
- `toleration` only affects **scheduling**, not **eviction**
- kubelet actively evicts Pods when disk pressure persists
- kubelet will evict Pods even if they're already running

**Solutions:**
1. **Fix root cause: Free disk space** (see Issue 1)
2. **Wait for kubelet to update status** (1-2 minutes after cleanup)
```bash
# Wait after cleanup
sleep 120

# Check node status
kubectl describe node | grep -A 2 DiskPressure
# Should show: DiskPressure=False
```

### Issue 6: Model Cache Path Causing Disk Pressure

**Symptoms:**
- Models downloaded to `/root/.cache/huggingface` (default path)
- This path is in container overlay filesystem, consuming `/dev/sda1`

**Solution:**
```yaml
env:
- name: HF_HOME
  value: "/models/hub"
- name: TRANSFORMERS_CACHE
  value: "/models/hub"
- name: HF_HUB_CACHE
  value: "/models/hub"

volumeMounts:
- name: models
  mountPath: /models  # Maps to /raid/models (hostPath)

volumes:
- name: models
  hostPath:
    path: /models  # Path in k3d container, corresponds to /raid/models on host
```

### Issue 7: k3d Cluster Creation Failed

**Symptoms:**
```bash
k3d cluster create mycluster-gpu
# Error: failed to prepare cluster
```

**Possible Causes and Solutions:**

1. **Image Issue**
```bash
# Ensure using correct image
docker images | grep k3s-cuda
# If image has issues, rebuild (see step 2)
```

2. **GPU Driver Issue**
```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

3. **Port Conflict**
```bash
# Check port usage
netstat -tuln | grep -E '6443|8080'
# Delete old cluster
k3d cluster delete mycluster-gpu
```

## Python Code Examples

### Prerequisites

```bash
pip install fastapi uvicorn transformers vllm httpx opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
```

### Running Examples

#### Tokenizer Service
```bash
python tokenizer_service.py
# Service runs on http://localhost:8001
```

#### API Gateway
```bash
python ch09_api_gateway.py
# Gateway runs on http://localhost:8000
```

#### Canary Deployment
```bash
python ch09_canary_deployment.py
```

#### Tracing
```bash
# Requires OpenTelemetry collector running
python ch09_tracing.py
```

## Troubleshooting Commands

```bash
# Check cluster status
kubectl get nodes
kubectl get pods --all-namespaces

# Check disk pressure
kubectl describe nodes | grep -A 2 DiskPressure
df -h /dev/sda1

# Check GPU availability
kubectl get nodes -o json | jq '.items[].status.capacity."nvidia.com/gpu"'

# Check storage
kubectl get storageclass
kubectl get pvc

# Check service endpoints
kubectl get svc
kubectl get endpoints

# Debug pod
kubectl describe pod <pod-name>
kubectl logs <pod-name>
kubectl exec -it <pod-name> -- /bin/sh
```

## Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [k3d Documentation](https://k3d.io/)
- [Kubernetes GPU Support](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
