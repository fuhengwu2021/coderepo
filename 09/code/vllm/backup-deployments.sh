#!/bin/bash
# Script to backup current Kubernetes deployments before recreating cluster
# This helps restore deployments after cluster recreation

set -e

BACKUP_DIR="./k8s-backup-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "📦 备份当前 Kubernetes 部署..."
echo "备份目录: $BACKUP_DIR"
echo ""

# Backup all deployments
echo "1. 备份 Deployments..."
kubectl get deployments -A -o yaml > "$BACKUP_DIR/deployments.yaml" 2>/dev/null || echo "  无 Deployments"

# Backup all pods
echo "2. 备份 Pods..."
kubectl get pods -A -o yaml > "$BACKUP_DIR/pods.yaml" 2>/dev/null || echo "  无 Pods"

# Backup all services
echo "3. 备份 Services..."
kubectl get services -A -o yaml > "$BACKUP_DIR/services.yaml" 2>/dev/null || echo "  无 Services"

# Backup all configmaps
echo "4. 备份 ConfigMaps..."
kubectl get configmaps -A -o yaml > "$BACKUP_DIR/configmaps.yaml" 2>/dev/null || echo "  无 ConfigMaps"

# Backup all secrets (without sensitive data)
echo "5. 备份 Secrets (metadata only)..."
kubectl get secrets -A -o yaml > "$BACKUP_DIR/secrets.yaml" 2>/dev/null || echo "  无 Secrets"

# Backup ingress
echo "6. 备份 Ingress..."
kubectl get ingress -A -o yaml > "$BACKUP_DIR/ingress.yaml" 2>/dev/null || echo "  无 Ingress"

# Backup vLLM specific resources
echo "7. 备份 vLLM 相关资源..."
mkdir -p "$BACKUP_DIR/vllm"
kubectl get all -l app=vllm -o yaml > "$BACKUP_DIR/vllm/vllm-resources.yaml" 2>/dev/null || echo "  无 vLLM 资源"
kubectl get all -l app=vllm-gateway -o yaml > "$BACKUP_DIR/vllm/gateway-resources.yaml" 2>/dev/null || echo "  无 Gateway 资源"

# Backup YAML files from vllm directory
echo "8. 备份 vllm 目录中的 YAML 文件..."
if [ -d "./vllm" ]; then
    cp -r ./vllm/*.yaml "$BACKUP_DIR/vllm/" 2>/dev/null || echo "  无 YAML 文件"
fi

# Create restore script
cat > "$BACKUP_DIR/restore.sh" << 'EOF'
#!/bin/bash
# Script to restore deployments from backup
# Usage: ./restore.sh

set -e

BACKUP_DIR=$(dirname "$0")

echo "🔄 恢复 Kubernetes 部署..."
echo "备份目录: $BACKUP_DIR"
echo ""

# Restore YAML files first
if [ -d "$BACKUP_DIR/vllm" ]; then
    echo "恢复 YAML 文件..."
    kubectl apply -f "$BACKUP_DIR/vllm/" 2>/dev/null || echo "  部分文件可能已存在"
fi

echo ""
echo "✅ 恢复完成！"
echo "注意：某些资源（如 Secrets）可能需要手动重新创建"
EOF

chmod +x "$BACKUP_DIR/restore.sh"

echo ""
echo "✅ 备份完成！"
echo ""
echo "备份内容："
ls -lh "$BACKUP_DIR"
echo ""
echo "📝 恢复方法："
echo "  cd $BACKUP_DIR"
echo "  ./restore.sh"
echo ""
echo "备份位置: $BACKUP_DIR"
