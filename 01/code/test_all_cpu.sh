#!/bin/bash
# Test all demo scripts on CPU

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECTIVE_DIR="$BASE_DIR/collective-operation"
P2P_DIR="$BASE_DIR/p2p"

# Function to run a demo and check if it succeeds
run_demo() {
    local demo_file=$1
    local demo_name=$(basename $demo_file .py)
    local port=$2
    
    echo "=========================================="
    echo "Testing: $demo_name"
    echo "=========================================="
    
    MASTER_PORT=$port OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 "$demo_file" --use_cpu > "/tmp/${demo_name}_output.log" 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ $demo_name: SUCCESS"
        return 0
    else
        echo "✗ $demo_name: FAILED (exit code: $exit_code)"
        echo "Last 10 lines of output:"
        tail -10 "/tmp/${demo_name}_output.log"
        return 1
    fi
}

# Start with port 29510 to avoid conflicts
PORT=29510

# Test collective operations
echo "Testing Collective Operations..."
echo ""

for demo in "$COLLECTIVE_DIR"/demo_*.py; do
    run_demo "$demo" $PORT
    PORT=$((PORT + 1))
    sleep 1  # Small delay between tests
done

echo ""
echo "Testing Point-to-Point Operations..."
echo ""

# Test p2p operations
for demo in "$P2P_DIR"/demo_*.py; do
    run_demo "$demo" $PORT
    PORT=$((PORT + 1))
    sleep 1  # Small delay between tests
done

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="
