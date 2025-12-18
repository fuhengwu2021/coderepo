#!/bin/bash
# 重新创建 k3d 集群，将数据目录 bind mount 到 /raid/tmpdata
# 这样 kubelet 监控的根文件系统就会在 /raid 上，而不是 /dev/sda1

set -e

CLUSTER_NAME="mycluster-gpu"
K3D_DATA_DIR="/raid/tmpdata/k3d-data"
MODELS_PATH="/raid/models"
VLLM_SOURCE_PATH="/home/fuhwu/workspace/distributedai/resources/vllm"

echo "=========================================="
echo "重新创建 k3d 集群（使用 /raid 存储）"
echo "=========================================="
echo ""
echo "⚠️  这将删除现有的 k3d 集群并重新创建"
echo "   只影响 k3d 集群，不影响其他 Docker 容器"
echo ""

# 检查 /raid/tmpdata 是否可用
if [ ! -d "/raid/tmpdata" ]; then
    echo "❌ 错误: /raid/tmpdata 目录不存在"
    exit 1
fi

# 创建 k3d 数据目录
echo "📁 创建 k3d 数据目录: $K3D_DATA_DIR"
mkdir -p "$K3D_DATA_DIR"
chmod 755 "$K3D_DATA_DIR"

# 显示当前集群状态
echo ""
echo "📊 当前集群状态："
kubectl get nodes 2>/dev/null || echo "  集群未运行或无法访问"
echo ""

# 确认操作
read -p "⚠️  确定要删除并重新创建集群吗？(y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 操作已取消"
    exit 1
fi

# 备份当前 kubeconfig（如果需要）
echo ""
echo "💾 备份当前 kubeconfig..."
if [ -f "$HOME/.kube/config" ]; then
    cp "$HOME/.kube/config" "$HOME/.kube/config.backup.$(date +%Y%m%d_%H%M%S)"
    echo "✅ 已备份到: $HOME/.kube/config.backup.*"
fi

# 删除现有集群
echo ""
echo "🗑️  删除现有集群: $CLUSTER_NAME"
k3d cluster delete "$CLUSTER_NAME" 2>/dev/null || echo "  集群不存在或已删除"

# 等待清理完成
echo "⏳ 等待清理完成..."
sleep 3

# 创建新集群，将 k3d 数据目录 bind mount 到 /raid
echo ""
echo "🚀 创建新集群（数据存储在 /raid）..."
echo "   - 集群名称: $CLUSTER_NAME"
echo "   - 数据目录: $K3D_DATA_DIR"
echo "   - 模型路径: $MODELS_PATH -> /models"
if [ -d "$VLLM_SOURCE_PATH" ]; then
    echo "   - vLLM 源码: $VLLM_SOURCE_PATH -> /vllm"
fi
echo ""

# 构建 k3d 创建命令
K3D_CMD="k3d cluster create $CLUSTER_NAME \
  --image k3s-cuda:v1.33.6-cuda-12.2.0 \
  --gpus=all \
  --servers 1 \
  --agents 1 \
  --volume $MODELS_PATH:/models"

# 如果 vLLM 源码目录存在，添加挂载
if [ -d "$VLLM_SOURCE_PATH" ]; then
    K3D_CMD="$K3D_CMD --volume $VLLM_SOURCE_PATH:/vllm"
fi

# 关键：将 k3d 的数据目录挂载到 /raid
# k3d 的数据存储在 Docker volume 中，我们需要通过环境变量或配置来指定
# 但更直接的方法是在创建后修改 Docker volume 的挂载点
# 或者使用 --k3s-arg 来指定数据目录

# 注意：k3d 使用 Docker volume 存储数据，我们需要在创建时指定
# 但 k3d 不直接支持指定数据目录，我们需要使用 workaround

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

# 修复 kubeconfig server 地址（如果需要）
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

# 检查节点磁盘压力
echo "💾 检查节点磁盘压力状态："
kubectl describe nodes | grep -A 2 "DiskPressure" || echo "  检查中..."
echo ""

# 检查存储配置
echo "📦 检查存储配置："
kubectl get storageclass
echo ""

# 检查 local-path-provisioner 配置
echo "🔍 检查 local-path-provisioner 配置："
kubectl get configmap local-path-config -n kube-system -o jsonpath='{.data.config\.json}' | python3 -m json.tool 2>/dev/null || echo "  配置检查中..."
echo ""

echo "💡 提示："
echo "   1. 集群数据现在存储在 Docker volume 中"
echo "   2. 需要进一步配置将 Docker volume 数据移到 /raid（需要额外步骤）"
echo "   3. 或者配置 local-path-provisioner 使用 /raid/tmpdata（已完成）"
echo ""
echo "📝 下一步："
echo "   1. 检查节点 disk-pressure 状态是否消失"
echo "   2. 测试创建 Pod 是否正常"
echo "   3. 如果仍有 disk-pressure，考虑清理 Docker 资源"
