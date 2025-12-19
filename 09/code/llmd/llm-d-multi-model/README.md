# LLM-d Cluster Deployment Guide

This directory contains the deployment configuration for an llm-d cluster to host `meta-llama/Llama-3.2-1B-Instruct` using two inference servers:
- **vLLM** (`vllm/vllm-openai:v0.12.0`)
- **SGLang** (`lmsysorg/sglang:v0.5.6.post2-runtime`)

## Design Overview

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    llm-d Cluster                        │
│                  (Separate k3d cluster)                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐         ┌──────────────┐            │
│  │  vLLM Server │         │ SGLang Server │            │
│  │  (Port 8000) │         │  (Port 8000)  │            │
│  └──────┬───────┘         └──────┬───────┘            │
│         │                        │                     │
│         └────────┬───────────────┘                     │
│                  │                                     │
│         ┌────────▼─────────┐                          │
│         │  llm-d Gateway   │                          │
│         │  (Routing Layer) │                          │
│         └──────────────────┘                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Key Components

1. **Separate k3d Cluster**: `llmd-cluster` (isolated from existing `mycluster-gpu`)
2. **vLLM Server**: OpenAI-compatible API server for Llama-3.2-1B-Instruct
3. **SGLang Server**: High-performance inference server for Llama-3.2-1B-Instruct
4. **llm-d Framework**: Kubernetes-native framework for distributed LLM serving
   - Uses Custom Resource Definitions (CRDs) for `LLMInferenceService`
   - Provides intelligent routing and load balancing
   - Supports disaggregated serving (prefill/decode separation)

## Prerequisites

1. **k3d installed** (already installed for existing cluster)
2. **Helm 3.x** installed
3. **kubectl** configured
4. **Docker** with GPU support
5. **HuggingFace Token** (for gated model access)
6. **Model storage**: `/raid/models` directory (shared with existing cluster)

## Step-by-Step Deployment

### Step 1: Verify Prerequisites

```bash
# Check k3d installation
k3d --version

# Check Helm installation
helm version

# Check kubectl
kubectl version --client

# Verify GPU support
nvidia-smi

# Set HuggingFace token
export HF_TOKEN='your_huggingface_token_here'
echo $HF_TOKEN
```

### Step 2: Create Separate k3d Cluster for llm-d

**Important**: This creates a NEW cluster separate from `mycluster-gpu` to avoid conflicts.

```bash
# Create llm-d cluster with GPU support
k3d cluster create llmd-cluster \
  --image k3s-cuda:v1.33.6-cuda-12.2.0-working \
  --gpus=all \
  --servers 1 \
  --agents 1 \
  --volume /raid/models:/models \
  --k3s-arg '--disable=traefik@server:0' \
  --port "8080:80@loadbalancer" \
  --port "8443:443@loadbalancer"

# Merge kubeconfig (will create separate context)
k3d kubeconfig merge llmd-cluster --kubeconfig-merge-default

# Switch to llmd-cluster context
kubectl config use-context k3d-llmd-cluster

# Verify cluster
kubectl get nodes
```

**Expected Output:**
```
NAME                      STATUS   ROLES                  AGE   VERSION
k3d-llmd-cluster-server-0   Ready    control-plane,master   30s   v1.33.6+k3s1
k3d-llmd-cluster-agent-0    Ready    <none>                 25s   v1.33.6+k3s1
```

### Step 3: Install NVIDIA Device Plugin

```bash
# Apply NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml

# Wait for device plugin to be ready
kubectl wait --for=condition=ready pod \
  -l name=nvidia-device-plugin-ds \
  -n kube-system \
  --timeout=60s

# Verify GPU detection (may take 20-30 seconds)
kubectl get nodes -o json | jq -r '.items[] | {name: .metadata.name, gpu: .status.allocatable."nvidia.com/gpu"}'
```

### Step 4: Install llm-d Framework

```bash
# Add llm-d Helm repository
helm repo add llm-d https://llm-d.ai/charts
helm repo update

# Install llm-d (this installs CRDs and controller)
helm install llm-d llm-d/llm-d \
  --namespace llm-d \
  --create-namespace \
  --wait \
  --timeout 5m

# Verify installation
kubectl get pods -n llm-d
kubectl get crd | grep llm
```

**Expected CRDs:**
- `llminferenceservices.llm-d.ai`

### Step 5: Create HuggingFace Token Secret

```bash
# Create secret in default namespace (for model servers)
kubectl create secret generic hf-token-secret \
  --from-literal=token="$HF_TOKEN" \
  --namespace default

# Verify secret
kubectl get secret hf-token-secret
```

### Step 6: Deploy vLLM Server

#### Option A: Using llm-d LLMInferenceService (Recommended)

```bash
# Deploy vLLM using llm-d CRD
kubectl apply -f vllm-llminference.yaml

# Check status
kubectl get llminferenceservice vllm-llama-32-1b
kubectl describe llminferenceservice vllm-llama-32-1b

# Wait for pods to be ready
kubectl wait --for=condition=ready pod \
  -l llm-d.ai/inference-service=vllm-llama-32-1b \
  --timeout=600s

# Check pods
kubectl get pods -l llm-d.ai/inference-service=vllm-llama-32-1b
```

#### Option B: Direct Pod Deployment (Alternative)

If llm-d CRD approach doesn't work, use direct pod deployment:

```bash
# Deploy vLLM pod directly
kubectl apply -f vllm-pod.yaml

# Wait for pod to be ready
kubectl wait --for=condition=ready pod/vllm-llama-32-1b --timeout=600s

# Check status
kubectl get pod vllm-llama-32-1b -o wide
```

### Step 7: Deploy SGLang Server

#### Option A: Using llm-d LLMInferenceService (Recommended)

```bash
# Deploy SGLang using llm-d CRD
kubectl apply -f sglang-llminference.yaml

# Check status
kubectl get llminferenceservice sglang-llama-32-1b
kubectl describe llminferenceservice sglang-llama-32-1b

# Wait for pods to be ready
kubectl wait --for=condition=ready pod \
  -l llm-d.ai/inference-service=sglang-llama-32-1b \
  --timeout=600s

# Check pods
kubectl get pods -l llm-d.ai/inference-service=sglang-llama-32-1b
```

#### Option B: Direct Pod Deployment (Alternative)

```bash
# Deploy SGLang pod directly
kubectl apply -f sglang-pod.yaml

# Wait for pod to be ready
kubectl wait --for=condition=ready pod/sglang-llama-32-1b --timeout=600s

# Check status
kubectl get pod sglang-llama-32-1b -o wide
```

### Step 8: Verify Deployments

```bash
# Check all pods
kubectl get pods -o wide

# Check services
kubectl get svc

# Check llm-d services
kubectl get llminferenceservice

# View logs (vLLM)
kubectl logs -f vllm-llama-32-1b --tail=50

# View logs (SGLang)
kubectl logs -f sglang-llama-32-1b --tail=50
```

### Step 9: Test Deployments

#### Test vLLM Server

```bash
# Port forward to vLLM service
kubectl port-forward svc/vllm-llama-32-1b 8001:8000

# In another terminal, test health
curl http://localhost:8001/health

# Test chat completion
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100
  }'
```

#### Test SGLang Server

```bash
# Port forward to SGLang service
kubectl port-forward svc/sglang-llama-32-1b 8002:8000

# In another terminal, test health
curl http://localhost:8002/health

# Test chat completion
curl http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100
  }'
```

## Management Commands

### Switch Between Clusters

```bash
# Switch to llm-d cluster
kubectl config use-context k3d-llmd-cluster

# Switch back to original cluster
kubectl config use-context k3d-mycluster-gpu

# List all contexts
kubectl config get-contexts
```

### Cluster Management

```bash
# Stop llm-d cluster
k3d cluster stop llmd-cluster

# Start llm-d cluster
k3d cluster start llmd-cluster

# Delete llm-d cluster (⚠️ WARNING: This deletes everything)
k3d cluster delete llmd-cluster

# List clusters
k3d cluster list
```

### Pod Management

```bash
# Restart vLLM pod
kubectl delete pod vllm-llama-32-1b
# Pod will be recreated automatically

# Restart SGLang pod
kubectl delete pod sglang-llama-32-1b
# Pod will be recreated automatically

# Scale (if using Deployment instead of Pod)
kubectl scale deployment vllm-llama-32-1b --replicas=2
```

### Logs and Debugging

```bash
# View vLLM logs
kubectl logs -f vllm-llama-32-1b

# View SGLang logs
kubectl logs -f sglang-llama-32-1b

# View llm-d controller logs
kubectl logs -f -n llm-d -l app.kubernetes.io/name=llm-d

# Describe pod for events
kubectl describe pod vllm-llama-32-1b
kubectl describe pod sglang-llama-32-1b
```

## Troubleshooting

### Issue: GPU not detected

```bash
# Check NVIDIA device plugin
kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds

# Check node GPU capacity
kubectl describe node k3d-llmd-cluster-agent-0 | grep -i gpu

# Restart device plugin
kubectl rollout restart daemonset nvidia-device-plugin-daemonset -n kube-system
```

### Issue: Pod stuck in Pending

```bash
# Check pod events
kubectl describe pod vllm-llama-32-1b

# Check resource availability
kubectl top nodes
kubectl top pods

# Check if GPU is available
kubectl get nodes -o json | jq '.items[] | {name: .metadata.name, gpu: .status.allocatable."nvidia.com/gpu"}'
```

### Issue: Model download fails

```bash
# Verify HF_TOKEN secret
kubectl get secret hf-token-secret -o jsonpath='{.data.token}' | base64 -d

# Check pod logs for download errors
kubectl logs vllm-llama-32-1b | grep -i "huggingface\|token\|download"

# Recreate secret if needed
kubectl delete secret hf-token-secret
kubectl create secret generic hf-token-secret --from-literal=token="$HF_TOKEN"
```

### Issue: Port conflicts

If ports 8001 or 8002 are already in use:

```bash
# Use different ports for port-forward
kubectl port-forward svc/vllm-llama-32-1b 9001:8000
kubectl port-forward svc/sglang-llama-32-1b 9002:8000
```

## Cleanup

### Remove Deployments

```bash
# Delete vLLM service
kubectl delete -f vllm-llminference.yaml
# or
kubectl delete -f vllm-pod.yaml

# Delete SGLang service
kubectl delete -f sglang-llminference.yaml
# or
kubectl delete -f sglang-pod.yaml

# Delete secrets
kubectl delete secret hf-token-secret
```

### Uninstall llm-d

```bash
# Uninstall llm-d Helm chart
helm uninstall llm-d -n llm-d

# Delete namespace
kubectl delete namespace llm-d
```

### Delete Cluster

```bash
# ⚠️ WARNING: This deletes the entire cluster and all data
k3d cluster delete llmd-cluster
```

## Files Overview

- `README.md` - This file (design doc and commands)
- `vllm-llminference.yaml` - vLLM LLMInferenceService CRD (llm-d framework)
- `sglang-llminference.yaml` - SGLang LLMInferenceService CRD (llm-d framework)
- `vllm-pod.yaml` - Direct vLLM Pod deployment (fallback)
- `sglang-pod.yaml` - Direct SGLang Pod deployment (fallback)
- `deploy.sh` - Automated deployment script
- `test.sh` - Test script for both servers

## Notes

1. **Cluster Isolation**: The `llmd-cluster` is completely separate from `mycluster-gpu`, so there's no risk of conflicts.

2. **Resource Sharing**: Both clusters share the `/raid/models` volume mount, so model files are accessible from both.

3. **GPU Allocation**: Ensure you have sufficient GPUs. Each server requires 1 GPU.

4. **Model Caching**: Models are cached in `/models/hub` (mapped from `/raid/models/hub`), so subsequent deployments will be faster.

5. **llm-d Framework**: The llm-d framework provides advanced features like:
   - Disaggregated serving (prefill/decode separation)
   - KV-cache aware routing
   - Automatic scaling
   - Load balancing

6. **Fallback Option**: If llm-d CRDs don't work as expected, use the direct Pod deployment files (`vllm-pod.yaml`, `sglang-pod.yaml`).
