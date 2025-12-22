#!/bin/bash
# Script to manage k3d cluster (stop/start/status)
# 
# Usage:
#   ./manage-cluster.sh [stop|start|restart|status]
#
# Examples:
#   ./manage-cluster.sh stop      # Stop the cluster
#   ./manage-cluster.sh start     # Start the cluster
#   ./manage-cluster.sh restart   # Restart the cluster
#   ./manage-cluster.sh status    # Show cluster status

set -e

CLUSTER_NAME="mycluster-gpu"

# Function to show usage
show_usage() {
    echo "Usage: $0 [stop|start|restart|status]"
    echo ""
    echo "Commands:"
    echo "  stop     - Stop the k3d cluster"
    echo "  start    - Start the k3d cluster"
    echo "  restart  - Restart the k3d cluster"
    echo "  status   - Show cluster status"
    exit 1
}

# Function to stop cluster
stop_cluster() {
    echo "=========================================="
    echo "Stopping k3d Cluster: $CLUSTER_NAME"
    echo "=========================================="
    echo ""
    
    if ! k3d cluster list | grep -q "$CLUSTER_NAME"; then
        echo "⚠️  Cluster $CLUSTER_NAME not found"
        exit 1
    fi
    
    # Stop port-forward processes before stopping cluster
    # This prevents orphaned processes when cluster stops
    echo "Checking for active port-forward processes..."
    PF_PIDS=$(pgrep -f "kubectl port-forward" 2>/dev/null || true)
    if [ -n "$PF_PIDS" ]; then
        echo "  Found port-forward processes, stopping them..."
        for pid in $PF_PIDS; do
            # Get the command line to show what we're stopping
            PF_CMD=$(ps -p $pid -o args= 2>/dev/null | head -1 || echo "unknown")
            echo "    Stopping port-forward (PID: $pid): $PF_CMD"
            kill $pid 2>/dev/null || true
        done
        sleep 1
        # Force kill if still running
        for pid in $PF_PIDS; do
            if ps -p $pid > /dev/null 2>&1; then
                echo "    Force killing port-forward (PID: $pid)..."
                kill -9 $pid 2>/dev/null || true
            fi
        done
        echo "  ✅ Port-forward processes stopped"
    else
        echo "  No port-forward processes found"
    fi
    echo ""
    
    echo "Stopping cluster..."
    k3d cluster stop "$CLUSTER_NAME"
    
    # Also stop manually created agent nodes (created via docker run)
    echo ""
    echo "Stopping manually created agent nodes..."
    MANUAL_AGENTS=$(docker ps -a --filter "label=k3d.cluster=$CLUSTER_NAME" --filter "label=k3d.role=agent" --format "{{.Names}}" | grep -v "^k3d-$CLUSTER_NAME" || true)
    if [ -n "$MANUAL_AGENTS" ]; then
        for agent in $MANUAL_AGENTS; do
            if docker ps --format "{{.Names}}" | grep -q "^${agent}$"; then
                echo "  Stopping $agent..."
                docker stop "$agent" 2>/dev/null || true
            fi
        done
    fi
    
    echo ""
    echo "✅ Cluster stopped"
    echo ""
    echo "📊 Cluster status:"
    k3d cluster list
    echo ""
    echo "📊 Docker containers status:"
    docker ps -a --filter "name=k3d-$CLUSTER_NAME" --filter "label=k3d.cluster=$CLUSTER_NAME" --format "table {{.Names}}\t{{.Status}}"
}

# Function to start cluster
start_cluster() {
    echo "=========================================="
    echo "Starting k3d Cluster: $CLUSTER_NAME"
    echo "=========================================="
    echo ""
    
    if ! k3d cluster list | grep -q "$CLUSTER_NAME"; then
        echo "❌ Error: Cluster $CLUSTER_NAME not found"
        echo "   Please create the cluster first (see README)"
        exit 1
    fi
    
    echo "Starting cluster..."
    k3d cluster start "$CLUSTER_NAME"
    
    # Also start manually created agent nodes if they exist
    echo ""
    echo "Starting manually created agent nodes..."
    MANUAL_AGENTS=$(docker ps -a --filter "label=k3d.cluster=$CLUSTER_NAME" --filter "label=k3d.role=agent" --format "{{.Names}}" | grep -v "^k3d-$CLUSTER_NAME" || true)
    if [ -n "$MANUAL_AGENTS" ]; then
        for agent in $MANUAL_AGENTS; do
            if ! docker ps --format "{{.Names}}" | grep -q "^${agent}$"; then
                echo "  Starting $agent..."
                docker start "$agent" 2>/dev/null || true
            fi
        done
    fi
    
    echo ""
    echo "⏳ Waiting for cluster to be ready..."
    sleep 5
    
    # Wait for NVIDIA device plugin to be ready (important for GPU workloads)
    echo ""
    echo "⏳ Waiting for NVIDIA device plugin to register GPUs..."
    echo "   (This may take 20-30 seconds after cluster start)"
    for i in {1..30}; do
        if kubectl get nodes -o json | python3 -c "import sys, json; d=json.load(sys.stdin); nodes=[n for n in d['items'] if n['status'].get('allocatable', {}).get('nvidia.com/gpu')]; exit(0 if nodes else 1)" 2>/dev/null; then
            echo "   ✅ GPUs detected on nodes"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "   ⚠️  GPUs not yet detected (device plugin may need more time)"
        else
            echo -n "."
            sleep 1
        fi
    done
    echo ""
    
    # Merge kubeconfig
    echo "🔗 Merging kubeconfig..."
    k3d kubeconfig merge "$CLUSTER_NAME" --kubeconfig-merge-default
    
    # Fix kubeconfig server address if needed
    export KUBECONFIG=$HOME/.kube/config
    KUBE_SERVER=$(kubectl config view -o jsonpath='{.clusters[?(@.name=="k3d-'$CLUSTER_NAME'")].cluster.server}' 2>/dev/null || echo "")
    if [[ "$KUBE_SERVER" == *"0.0.0.0"* ]]; then
        echo "🔧 Fixing kubeconfig server address..."
        kubectl config set-cluster "k3d-$CLUSTER_NAME" --server=$(echo $KUBE_SERVER | sed 's/0.0.0.0/127.0.0.1/')
    fi
    
    # Auto-deploy LLM serving pods if they don't exist
    echo ""
    echo "🚀 Checking and deploying LLM serving pods..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # Deploy vLLM Llama-3.2-1B pod if it doesn't exist
    if ! kubectl get pod vllm-llama-32-1b &>/dev/null; then
        echo "  📦 Deploying vLLM Llama-3.2-1B pod..."
        if [ -f "$SCRIPT_DIR/vllm/llama-3.2-1b.yaml" ]; then
            kubectl apply -f "$SCRIPT_DIR/vllm/llama-3.2-1b.yaml" 2>/dev/null || echo "    ⚠️  Failed to deploy vLLM pod (may need HF_TOKEN secret)"
        else
            echo "    ⚠️  vLLM YAML file not found: $SCRIPT_DIR/vllm/llama-3.2-1b.yaml"
        fi
    else
        echo "  ✅ vLLM Llama-3.2-1B pod already exists"
    fi
    
    # Deploy SGLang Llama-3.2-1B pod if it doesn't exist
    if ! kubectl get pod sglang-llama-32-1b &>/dev/null; then
        echo "  📦 Deploying SGLang Llama-3.2-1B pod..."
        if [ -f "$SCRIPT_DIR/sglang/llama-3.2-1b.yaml" ]; then
            kubectl apply -f "$SCRIPT_DIR/sglang/llama-3.2-1b.yaml" 2>/dev/null || echo "    ⚠️  Failed to deploy SGLang pod (may need HF_TOKEN secret)"
        else
            echo "    ⚠️  SGLang YAML file not found: $SCRIPT_DIR/sglang/llama-3.2-1b.yaml"
        fi
    else
        echo "  ✅ SGLang Llama-3.2-1B pod already exists"
    fi
    
    echo ""
    echo "✅ Cluster started"
    echo ""
    echo "📊 Cluster status:"
    k3d cluster list
    echo ""
    echo "📊 Node status:"
    kubectl get nodes
    echo ""
    echo "📊 Pod status:"
    kubectl get pods
}

# Function to restart cluster
restart_cluster() {
    echo "=========================================="
    echo "Restarting k3d Cluster: $CLUSTER_NAME"
    echo "=========================================="
    echo ""
    stop_cluster
    echo ""
    sleep 2
    start_cluster
}

# Function to show status
show_status() {
    echo "=========================================="
    echo "k3d Cluster Status: $CLUSTER_NAME"
    echo "=========================================="
    echo ""
    
    echo "📊 Cluster list:"
    k3d cluster list
    echo ""
    
    # Check if cluster is running by checking if servers/agents are running (not 0/1)
    CLUSTER_STATUS=$(k3d cluster list --no-headers | grep "^$CLUSTER_NAME" | awk '{print $2, $3}' || echo "")
    if echo "$CLUSTER_STATUS" | grep -qE "[1-9]/[0-9]+.*[1-9]/[0-9]+"; then
        # Switch to cluster context
        kubectl config use-context "k3d-$CLUSTER_NAME" 2>/dev/null || true
        
        echo "📊 Kubernetes nodes:"
        kubectl get nodes 2>/dev/null || echo "  Unable to connect to cluster"
        echo ""
        
        echo "📊 Pods:"
        kubectl get pods 2>/dev/null || echo "  Unable to list pods"
        echo ""
        
        echo "📊 Services:"
        kubectl get svc 2>/dev/null | head -10 || echo "  Unable to list services"
    else
        echo "⚠️  Cluster is not running (servers/agents: $CLUSTER_STATUS)"
    fi
}

# Main script logic
case "${1:-}" in
    stop)
        stop_cluster
        ;;
    start)
        start_cluster
        ;;
    restart)
        restart_cluster
        ;;
    status)
        show_status
        ;;
    *)
        show_usage
        ;;
esac
