#!/bin/bash
# Script to add 443 port mapping to existing k3d cluster
# This uses k3d node edit command (available in k3d 5.0.0+)
# No need to delete and recreate the cluster!

set -e

CLUSTER_NAME="mycluster-gpu"

echo "=========================================="
echo "为现有 k3d 集群添加 443 端口映射"
echo "=========================================="
echo ""
echo "✅ 好消息：不需要删除集群！"
echo "   使用 k3d node edit 命令动态添加端口映射"
echo ""

# Check k3d version
K3D_VERSION=$(k3d --version 2>&1 | grep -oP 'v\d+\.\d+' | head -1)
echo "k3d 版本: $K3D_VERSION"
echo ""

# Check if cluster exists
if ! k3d cluster list | grep -q "$CLUSTER_NAME"; then
    echo "❌ 错误: 集群 $CLUSTER_NAME 不存在"
    exit 1
fi

# Find loadbalancer node
LB_NAME=$(k3d node list 2>/dev/null | grep loadbalancer | awk '{print $1}' | head -1)

if [ -z "$LB_NAME" ]; then
    echo "❌ 错误: 未找到 loadbalancer 节点"
    echo "   请检查集群状态: k3d cluster list"
    exit 1
fi

echo "找到 loadbalancer: $LB_NAME"
echo ""

# Check current port mappings
echo "当前端口映射："
k3d node list | grep "$LB_NAME" || echo "  无端口映射"
echo ""

# Check if port 443 is already mapped
if docker port $(docker ps -q --filter "name=$LB_NAME") 2>/dev/null | grep -q "443"; then
    echo "⚠️  警告: 端口 443 可能已经映射"
    echo "   继续操作可能会失败或产生冲突"
    read -p "是否继续？(y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ 操作已取消"
        exit 1
    fi
fi

# Add port mapping
echo ""
echo "🚀 添加端口映射 443:443..."
echo "命令: k3d node edit $LB_NAME --port-add 443:443"
echo ""

if k3d node edit "$LB_NAME" --port-add 443:443; then
    echo ""
    echo "✅ 端口映射添加成功！"
    echo ""
    echo "验证端口映射："
    sleep 2
    docker port $(docker ps -q --filter "name=$LB_NAME") 2>/dev/null | grep 443 || echo "  检查中..."
    echo ""
    echo "📝 下一步："
    echo "   1. 确保 Ingress Controller 已安装"
    echo "   2. 确保 Ingress 和 Gateway 已部署"
    echo "   3. 测试访问: curl -k https://localhost/v1/models"
else
    echo ""
    echo "❌ 端口映射添加失败"
    echo ""
    echo "可能的原因："
    echo "  - k3d 版本过低（需要 5.0.0+）"
    echo "  - 端口 443 已被占用"
    echo "  - 权限不足"
    echo ""
    echo "如果失败，可以尝试重新创建集群："
    echo "  ./recreate-cluster-with-ports.sh"
    exit 1
fi

