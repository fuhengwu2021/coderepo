# Production LLM Serving Stack

This directory contains code examples for building a complete production LLM serving stack, including Kubernetes deployment configurations for vLLM.

## Directory Structure

```
09/code/
├── vllm/                    # vLLM 服务器部署配置
│   ├── llama-3.2-1b.yaml    # Llama-3.2-1B-Instruct 模型部署（已部署）
│   ├── phi-tiny-moe.yaml    # Phi-tiny-MoE-instruct 模型部署
│   ├── api-gateway.yaml     # API Gateway 部署（自动路由）
│   ├── api-gateway.py       # Gateway Python 代码（存储在 ConfigMap）
│   ├── deploy-llama-3.2-1b.sh
│   ├── deploy-gateway.sh    # Gateway 部署脚本
│   ├── test-api.sh          # API 测试脚本
│   └── README.md            # vLLM 部署文档
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

For production environments, expose the Gateway service using Ingress for production-grade access with TLS and domain names.

**Prerequisites:**
```bash
# Install Nginx Ingress Controller (if not already installed)
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml

# For k3d, use the k3d-specific ingress
kubectl apply -f https://raw.githubusercontent.com/k3d-io/k3d/main/docs/usage/examples/ingress/ingress.yaml
```

**Create Ingress Resource:**

Create `vllm/ingress.yaml`:
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vllm-api-gateway-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "false"  # Set to true with TLS
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
spec:
  ingressClassName: nginx
  rules:
  - host: localhost
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: vllm-api-gateway
            port:
              number: 8000
```

**Deploy:**
```bash
kubectl apply -f vllm/ingress.yaml

# Access via localhost (HTTP)
curl http://localhost/v1/models
```

**With TLS (Production):**

For production environments with TLS enabled, use HTTPS:
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vllm-api-gateway-ingress
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"  # If using cert-manager
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - localhost
    secretName: vllm-api-tls
  rules:
  - host: localhost
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: vllm-api-gateway
            port:
              number: 8000
```

**Deploy with TLS:**
```bash
kubectl apply -f vllm/ingress.yaml

# Access via HTTPS (if TLS is configured)
curl https://localhost/v1/models --insecure  # Use --insecure for self-signed certs
# Or with proper certificate validation:
curl https://localhost/v1/models --cacert /path/to/ca.crt
```

**Note:** For localhost, TLS is typically not necessary. Use HTTP for local development. TLS is recommended for production environments with real domains.

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
API Gateway (统一入口)
    ↓
解析请求体中的 'model' 字段
    ↓
路由到对应的 Service
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

## 踩过的坑和解决方案 (Common Issues and Solutions)

### 坑 1: Disk Pressure 导致 Pod 无法调度

**问题现象：**
```bash
kubectl get nodes
# 节点显示 DiskPressure=True
kubectl describe pod <pod-name>
# 显示: 0/1 nodes are available: 1 node(s) had disk-pressure
```

**原因：**
- `/dev/sda1` 磁盘使用率超过 85%（kubelet 默认阈值）
- Docker 镜像和容器占用大量空间
- k3d 的 overlay 文件系统也在 `/dev/sda1` 上

**解决方案：**

1. **清理 Docker 资源（推荐，快速有效）**
```bash
# 查看磁盘使用
df -h /dev/sda1

# 查看 Docker 占用
docker system df

# 清理未使用的资源（可释放约 200-300GB）
docker image prune -a -f      # 清理未使用的镜像
docker container prune -f      # 清理已停止的容器
docker volume prune -f        # 清理未使用的卷

# 或使用脚本
./cleanup-docker.sh
```

2. **配置 local-path-provisioner 使用 /raid**
```bash
# 将存储路径改为 /raid/tmpdata（见步骤 4）
# 这样 PVC 数据会存储在 /raid 而不是 /dev/sda1
```

3. **添加 toleration（临时方案，不推荐生产环境）**
```yaml
spec:
  tolerations:
  - key: node.kubernetes.io/disk-pressure
    operator: Exists
    effect: NoSchedule
```

### 坑 2: vLLM Pod 报错 `libcuda.so.1: cannot open shared object file`

**问题现象：**
```bash
kubectl logs vllm-pod
# ImportError: libcuda.so.1: cannot open shared object file: No such file or directory
```

**原因：**
- Pod 缺少 GPU runtime 配置
- 没有指定 `runtimeClassName: nvidia`

**解决方案：**
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

### 坑 3: Gated Model 访问被拒绝

**问题现象：**
```bash
kubectl logs vllm-pod
# OSError: You are trying to access a gated repo. 
# Access to model ... is restricted. You must have access to it and be authenticated.
```

**原因：**
- 模型是 gated repo（如 meta-llama/Llama-3.2-1B-Instruct）
- 需要 HuggingFace token 但未配置

**解决方案：**

1. **创建 Secret（不要硬编码 token）**
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

3. **使用部署脚本（自动处理）**
```bash
./vllm/deploy-llama-3.2-1b.sh  # 脚本会自动创建 Secret
```

### 坑 4: Service 名称包含点号导致创建失败

**问题现象：**
```bash
kubectl apply -f vllm-llama-3.2-1b.yaml
# Error: metadata.name: Invalid value: "vllm-llama-3.2-1b": 
# a DNS-1035 label must consist of lower case alphanumeric characters or '-'
```

**原因：**
- Kubernetes Service 名称必须符合 DNS-1035 规范
- 不能包含点号（`.`），只能使用小写字母、数字和连字符

**解决方案：**
```yaml
# ❌ 错误
metadata:
  name: vllm-llama-3.2-1b

# ✅ 正确
metadata:
  name: vllm-llama-32-1b  # 用连字符替代点号
```

### 坑 5: Pod 被 Evicted 即使添加了 Toleration

**问题现象：**
```bash
kubectl get pod
# NAME               STATUS    RESTARTS   AGE
# test-pod           Evicted   0          5m
```

**原因：**
- `toleration` 只影响**调度**（scheduling），不影响**驱逐**（eviction）
- kubelet 会主动驱逐 Pod 当磁盘压力持续存在
- 即使 Pod 已经运行，kubelet 也会驱逐它

**解决方案：**
1. **解决根本问题：释放磁盘空间**（见坑 1）
2. **等待 kubelet 更新状态**（清理后需要 1-2 分钟）
```bash
# 清理后等待
sleep 120

# 检查节点状态
kubectl describe node | grep -A 2 DiskPressure
# 应该显示: DiskPressure=False
```

### 坑 6: 模型缓存路径导致磁盘压力

**问题现象：**
- 模型下载到 `/root/.cache/huggingface`（默认路径）
- 这个路径在容器 overlay 文件系统中，占用 `/dev/sda1`

**解决方案：**
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
  mountPath: /models  # 映射到 /raid/models（hostPath）

volumes:
- name: models
  hostPath:
    path: /models  # k3d 容器内路径，对应主机的 /raid/models
```

### 坑 7: k3d 集群创建失败

**问题现象：**
```bash
k3d cluster create mycluster-gpu
# Error: failed to prepare cluster
```

**可能原因和解决方案：**

1. **镜像问题**
```bash
# 确保使用正确的镜像
docker images | grep k3s-cuda
# 如果镜像有问题，重新构建（见步骤 2）
```

2. **GPU 驱动问题**
```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 Docker GPU 支持
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

3. **端口冲突**
```bash
# 检查端口占用
netstat -tuln | grep -E '6443|8080'
# 删除旧集群
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
