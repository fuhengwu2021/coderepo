#!/bin/bash
# Script to recreate k3d cluster with port mapping for Ingress
# This allows direct access via https://localhost without port-forward
#
# ⚠️  WARNING: This will delete the existing cluster and all deployments!
# Make sure to backup important configurations before running this script.

set -e

CLUSTER_NAME="mycluster-gpu"
MODELS_PATH="/raid/models"
VLLM_SOURCE_PATH="/home/fuhwu/workspace/distributedai/resources/vllm"

echo "=========================================="
echo "重新创建 k3d 集群（带端口映射）"
echo "=========================================="
echo ""
echo "⚠️  警告：这将删除现有集群和所有部署！"
echo "   请确保已备份重要配置"
echo ""
echo "端口映射："
echo "  - 443:443@loadbalancer (HTTPS)"
echo "  - 80:80@loadbalancer (HTTP)"
echo ""

# 备份当前部署
echo ""
echo "💾 备份当前 Kubernetes 部署..."
if [ -f "./backup-deployments.sh" ]; then
    ./backup-deployments.sh
else
    echo "⚠️  备份脚本不存在，跳过部署备份"
fi

# 备份 kubeconfig
echo ""
echo "💾 备份 kubeconfig..."
if [ -f "$HOME/.kube/config" ]; then
    cp "$HOME/.kube/config" "$HOME/.kube/config.backup.$(date +%Y%m%d_%H%M%S)"
    echo "✅ 已备份"
fi

# 确认操作
echo ""
read -p "⚠️  确定要删除集群并重新创建吗？(y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 操作已取消"
    exit 1
fi

# 删除现有集群
echo ""
echo "🗑️  删除现有集群..."
k3d cluster delete "$CLUSTER_NAME" 2>/dev/null || echo "  集群不存在或已删除"

# 等待清理
sleep 3

# 创建新集群（带端口映射）
echo ""
echo "🚀 创建新集群（带端口映射）..."
echo "   - 集群名称: $CLUSTER_NAME"
echo "   - 模型路径: $MODELS_PATH -> /models"
if [ -d "$VLLM_SOURCE_PATH" ]; then
    echo "   - vLLM 源码: $VLLM_SOURCE_PATH -> /vllm"
fi
echo "   - 端口映射: 443:443@loadbalancer, 80:80@loadbalancer"
echo ""

K3D_CMD="k3d cluster create $CLUSTER_NAME \
  --image k3s-cuda:v1.33.6-cuda-12.2.0-working \
  --gpus=all \
  --servers 1 \
  --agents 1 \
  --volume $MODELS_PATH:/models \
  --port '443:443@loadbalancer' \
  --port '80:80@loadbalancer'"

if [ -d "$VLLM_SOURCE_PATH" ]; then
    K3D_CMD="$K3D_CMD --volume $VLLM_SOURCE_PATH:/vllm"
fi

echo "执行命令："
echo "$K3D_CMD"
echo ""

eval $K3D_CMD

# 等待集群就绪
echo ""
echo "⏳ 等待集群就绪..."
sleep 10

# 合并 kubeconfig
echo ""
echo "🔗 合并 kubeconfig..."
k3d kubeconfig merge "$CLUSTER_NAME" --kubeconfig-merge-default

# 修复 kubeconfig
export KUBECONFIG=$HOME/.kube/config
KUBE_SERVER=$(kubectl config view -o jsonpath='{.clusters[?(@.name=="k3d-'$CLUSTER_NAME'")].cluster.server}' 2>/dev/null || echo "")
if [[ "$KUBE_SERVER" == *"0.0.0.0"* ]]; then
    echo "🔧 修复 kubeconfig server 地址..."
    kubectl config set-cluster "k3d-$CLUSTER_NAME" --server=$(echo $KUBE_SERVER | sed 's/0.0.0.0/127.0.0.1/')
fi

# 验证集群
echo ""
echo "=========================================="
echo "✅ 集群创建完成！"
echo "=========================================="
echo ""
echo "📊 集群状态："
kubectl get nodes
echo ""

# 验证端口映射
echo "🔍 验证端口映射："
LB_CONTAINER=$(docker ps -q --filter "name=k3d.*loadbalancer")
if [ -n "$LB_CONTAINER" ]; then
    echo "Loadbalancer 容器: $LB_CONTAINER"
    docker port $LB_CONTAINER 2>/dev/null || echo "  检查端口映射..."
    echo ""
    echo "端口绑定："
    docker inspect $LB_CONTAINER --format='{{json .HostConfig.PortBindings}}' | jq '.' 2>/dev/null || echo "  无法检查端口绑定"
else
    echo "⚠️  未找到 loadbalancer 容器"
fi

echo ""
echo "📝 下一步："
echo "   1. 安装 Ingress Controller:"
echo "      kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml"
echo ""
echo "   2. 部署 Gateway 和 Ingress:"
echo "      kubectl apply -f vllm/api-gateway.yaml"
echo "      kubectl apply -f vllm/ingress-tls.yaml"
echo ""
echo "   3. 测试访问（应该可以直接使用 https://localhost）:"
echo "      curl -k https://localhost/v1/models"
echo ""
