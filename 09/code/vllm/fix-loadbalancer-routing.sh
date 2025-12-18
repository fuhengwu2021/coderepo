#!/bin/bash
# Script to fix k3d loadbalancer routing to NGINX Ingress Controller
# This script addresses the issue where k3d loadbalancer forwards to Traefik by default

set -e

echo "=========================================="
echo "修复 k3d loadbalancer 路由到 NGINX Ingress Controller"
echo "=========================================="
echo ""

# Check if Traefik is running
echo "1. 检查 Traefik 状态..."
TRAEFIK_PODS=$(kubectl get pods -A | grep traefik | wc -l)
if [ "$TRAEFIK_PODS" -gt 0 ]; then
    echo "⚠️  发现 Traefik 正在运行（$TRAEFIK_PODS 个 Pod）"
    echo "   这是问题的根源：k3d loadbalancer 默认转发到 Traefik"
    echo ""
    echo "选项："
    echo "  A. 禁用 Traefik（推荐，但需要重新创建集群）"
    echo "  B. 配置 Traefik 转发到 NGINX Ingress Controller"
    echo "  C. 使用 Traefik 而不是 NGINX Ingress Controller"
    echo ""
    read -p "选择方案 (A/B/C，默认 A): " choice
    choice=${choice:-A}
    
    case $choice in
        A)
            echo ""
            echo "方案 A: 禁用 Traefik（需要重新创建集群）"
            echo ""
            echo "⚠️  这需要重新创建集群，会删除所有部署"
            echo "   创建集群时使用："
            echo "   k3d cluster create ... --k3s-arg '--disable=traefik@server:0'"
            echo ""
            echo "已创建脚本: recreate-cluster-without-traefik.sh"
            exit 0
            ;;
        B)
            echo ""
            echo "方案 B: 配置 Traefik 转发到 NGINX Ingress Controller"
            echo "   这需要修改 Traefik 配置，比较复杂"
            echo "   不推荐此方案"
            exit 1
            ;;
        C)
            echo ""
            echo "方案 C: 使用 Traefik 而不是 NGINX Ingress Controller"
            echo "   删除 NGINX Ingress Controller，使用 Traefik"
            read -p "确定要删除 NGINX Ingress Controller 吗？(y/N) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo "删除 NGINX Ingress Controller..."
                kubectl delete namespace ingress-nginx
                echo "✅ 已删除，现在使用 Traefik"
                echo ""
                echo "需要更新 Ingress 配置使用 Traefik class"
                exit 0
            else
                echo "操作已取消"
                exit 1
            fi
            ;;
    esac
else
    echo "✅ Traefik 未运行"
fi

echo ""
echo "2. 检查 NGINX Ingress Controller 状态..."
kubectl get pods -n ingress-nginx | grep controller || echo "  NGINX Ingress Controller 未运行"

echo ""
echo "3. 检查 Ingress Controller Service 类型..."
INGRESS_TYPE=$(kubectl get svc -n ingress-nginx ingress-nginx-controller -o jsonpath='{.spec.type}' 2>/dev/null || echo "未找到")
echo "Service 类型: $INGRESS_TYPE"

if [ "$INGRESS_TYPE" = "LoadBalancer" ]; then
    echo ""
    echo "4. 检查 LoadBalancer IP..."
    LB_IP=$(kubectl get svc -n ingress-nginx ingress-nginx-controller -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    if [ -z "$LB_IP" ]; then
        echo "⚠️  LoadBalancer IP 未分配"
        echo "   在 k3d 中，LoadBalancer 类型需要 MetalLB 或类似工具"
        echo "   或者使用 NodePort 类型"
        echo ""
        echo "建议：将 Service 改回 NodePort 类型"
        kubectl patch svc -n ingress-nginx ingress-nginx-controller -p '{"spec":{"type":"NodePort"}}'
        echo "✅ 已改回 NodePort"
    else
        echo "LoadBalancer IP: $LB_IP"
    fi
fi

echo ""
echo "5. 测试访问..."
echo ""
echo "如果 Traefik 存在，k3d loadbalancer 会转发到 Traefik"
echo "需要确保 Traefik 能正确路由到 NGINX Ingress Controller"
echo ""
echo "或者，最简单的方法："
echo "  使用 NodePort 访问: curl -k -H 'Host: localhost' https://localhost:31965/health"
