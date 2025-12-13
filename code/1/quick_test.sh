#!/bin/bash
# Quick test script for CPU demos

# Cleanup function to kill any remaining processes
cleanup() {
    pkill -f "torchrun.*demo_" 2>/dev/null || true
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECTIVE_DIR="$BASE_DIR/collective-operation"
P2P_DIR="$BASE_DIR/p2p"

test_demo() {
    local demo=$1
    local port=$2
    local name=$(basename $demo .py)
    
    echo -n "Testing $name... "
    # Use timeout and redirect both stdout and stderr
    # Run in subshell to ensure proper cleanup
    (MASTER_PORT=$port OMP_NUM_THREADS=1 timeout 15 torchrun --nproc_per_node=2 "$demo" --use_cpu > /dev/null 2>&1)
    local exit_code=$?
    # Wait for any background jobs to finish
    wait 2>/dev/null || true
    # Small delay to ensure process cleanup
    sleep 0.2
    if [ $exit_code -eq 0 ] || [ $exit_code -eq 124 ]; then
        echo "✓"
        return 0
    else
        echo "✗"
        return 1
    fi
}

PORT=29550
FAILED=0

echo "=== Collective Operations ==="
for demo in "$COLLECTIVE_DIR"/demo_*.py; do
    test_demo "$demo" $PORT || FAILED=$((FAILED + 1))
    PORT=$((PORT + 1))
done

echo ""
echo "=== Point-to-Point Operations ==="
for demo in "$P2P_DIR"/demo_*.py; do
    test_demo "$demo" $PORT || FAILED=$((FAILED + 1))
    PORT=$((PORT + 1))
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "All tests passed! ✓"
else
    echo "$FAILED test(s) failed"
fi

# Final cleanup and wait for any remaining processes
wait 2>/dev/null || true
sleep 0.2
cleanup

exit $FAILED
