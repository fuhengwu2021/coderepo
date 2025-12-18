#!/bin/bash
# Script to recreate k3d cluster without Traefik, using NGINX Ingress Controller
# This ensures k3d loadbalancer forwards correctly to NGINX Ingress Controller

set -e

CLUSTER_NAME="mycluster-gpu"
MODELS_PATH="/raid/models"
VLLM_SOURCE_PATH="/home/fuhwu/workspace/distributedai/resources/vllm"

echo "=========================================="
echo "重新创建 k3d 集群（禁用 Traefik，使用 NGINX Ingress）"
echo "=========================================="
echo ""
echo "⚠️  警告：这将删除现有集群和所有部署！"
echo ""
echo "关键配置："
echo "  - 禁用 Traefik: --k3s-arg '--disable=traefik@server:0'"
echo "  - 端口映射: 443:443@loadbalancer, 80:80@loadbalancer"
echo ""

# Backup deployments
if [ -f "./backup-deployments.sh" ]; then
    ./backup-deployments.sh
fi

# Confirm
read -p "确定要继续吗？(y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 操作已取消"
    exit 1
fi

# Delete existing cluster
echo ""
echo "🗑️  删除现有集群..."
k3d cluster delete "$CLUSTER_NAME" 2>/dev/null || echo "  集群不存在或已删除"
sleep 3

# Create cluster without Traefik
echo ""
echo "🚀 创建新集群（禁用 Traefik）..."
K3D_CMD="k3d cluster create $CLUSTER_NAME \
  --image k3s-cuda:v1.33.6-cuda-12.2.0-working \
  --gpus=all \
  --servers 1 \
  --agents 1 \
  --volume $MODELS_PATH:/models \
  --k3s-arg '--disable=traefik@server:0' \
  --port '443:443@loadbalancer' \
  --port '80:80@loadbalancer'"

if [ -d "$VLLM_SOURCE_PATH" ]; then
    K3D_CMD="$K3D_CMD --volume $VLLM_SOURCE_PATH:/vllm"
fi

echo "执行命令："
echo "$K3D_CMD"
echo ""

eval $K3D_CMD

# Wait for cluster
echo ""
echo "⏳ 等待集群就绪..."
sleep 10

# Merge kubeconfig
echo ""
echo "🔗 合并 kubeconfig..."
k3d kubeconfig merge "$CLUSTER_NAME" --kubeconfig-merge-default

# Fix kubeconfig
export KUBECONFIG=$HOME/.kube/config
KUBE_SERVER=$(kubectl config view -o jsonpath='{.clusters[?(@.name=="k3d-'$CLUSTER_NAME'")].cluster.server}' 2>/dev/null || echo "")
if [[ "$KUBE_SERVER" == *"0.0.0.0"* ]]; then
    echo "🔧 修复 kubeconfig server 地址..."
    kubectl config set-cluster "k3d-$CLUSTER_NAME" --server=$(echo $KUBE_SERVER | sed 's/0.0.0.0/127.0.0.1/')
fi

# Verify
echo ""
echo "=========================================="
echo "✅ 集群创建完成！"
echo "=========================================="
echo ""
echo "📊 集群状态："
kubectl get nodes
echo ""
echo "验证 Traefik 已禁用："
kubectl get pods -n kube-system | grep traefik && echo "⚠️  Traefik 仍在运行" || echo "✅ Traefik 已禁用"
echo ""
echo "📝 下一步："
echo "  1. 安装 NGINX Ingress Controller:"
echo "     kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml"
echo ""
echo "  2. 等待 Ingress Controller 就绪:"
echo "     kubectl wait --namespace ingress-nginx --for=condition=ready pod --selector=app.kubernetes.io/component=controller --timeout=90s"
echo ""
echo "  3. 部署 Gateway 和 Ingress:"
echo "     kubectl apply -f vllm/api-gateway.yaml"
echo "     kubectl apply -f vllm/ingress-tls.yaml"
echo ""
echo "  4. 测试访问:"
echo "     curl -k https://localhost/health"
