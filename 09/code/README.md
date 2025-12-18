# Production LLM Serving Stack

This directory contains code examples for building a complete production LLM serving stack, including Kubernetes deployment configurations for vLLM.

## Directory Structure

```
09/code/
├── vllm/                    # vLLM 服务器部署配置
│   ├── llama-3.2-1b.yaml    # Llama-3.2-1B-Instruct 模型部署（已部署）
│   ├── deploy-llama-3.2-1b.sh
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

### 4. Configure Storage for k3d

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

### 5. Set Up HuggingFace Token (for Gated Models)

```bash
# Export HF_TOKEN environment variable
export HF_TOKEN='your_huggingface_token_here'

# Verify it's set
echo $HF_TOKEN
```

### 6. Deploy vLLM Model

#### Option A: Using Deployment Script (Recommended)

```bash
cd vllm/

# Deploy Llama-3.2-1B-Instruct
./deploy-llama-3.2-1b.sh
```

#### Option B: Manual Deployment

```bash
# Create HF token secret
kubectl create secret generic hf-token-secret \
  --from-literal=token="$HF_TOKEN"

# Deploy model
kubectl apply -f vllm/llama-3.2-1b.yaml

# Check status
kubectl get pod vllm-llama-32-1b -w
```

### 7. Test the Deployment

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
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 8. Monitor and Debug

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
