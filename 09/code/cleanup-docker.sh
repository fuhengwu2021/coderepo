#!/bin/bash
# 清理 Docker 资源脚本
# 可以释放约 309GB 空间，不影响正在运行的服务

set -e

echo "=========================================="
echo "Docker 资源清理脚本"
echo "=========================================="
echo ""

# 显示清理前的状态
echo "📊 清理前的状态："
echo "----------------------------------------"
df -h /dev/sda1 | tail -1
echo ""
docker system df
echo ""

# 确认操作
read -p "⚠️  这将删除未使用的镜像、已停止的容器和未使用的卷。继续吗？(y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 操作已取消"
    exit 1
fi

echo ""
echo "🧹 开始清理..."
echo ""

# 1. 清理未使用的镜像（不包括正在使用的）
echo "1️⃣  清理未使用的镜像..."
docker image prune -a -f
echo "✅ 镜像清理完成"
echo ""

# 2. 清理已停止的容器
echo "2️⃣  清理已停止的容器..."
docker container prune -f
echo "✅ 容器清理完成"
echo ""

# 3. 清理未使用的卷
echo "3️⃣  清理未使用的卷..."
docker volume prune -f
echo "✅ 卷清理完成"
echo ""

# 4. 显示清理后的状态
echo "📊 清理后的状态："
echo "----------------------------------------"
df -h /dev/sda1 | tail -1
echo ""
docker system df
echo ""

echo "✅ 清理完成！"
echo ""
echo "💡 提示：如果磁盘使用率仍然高于 85%，可以考虑："
echo "   1. 调整 kubelet eviction 阈值"
echo "   2. 清理其他大文件（如 /mnt/co-research）"
echo "   3. 移动 Docker 数据目录到 /raid（高风险）"
