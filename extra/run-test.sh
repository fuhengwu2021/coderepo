#!/bin/bash
# Wrapper script to run test_llama4_scout.py with conda environment "research"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="$SCRIPT_DIR/test_llama4_scout.py"

# Activate conda environment
echo "🔧 Activating conda environment: research"
eval "$(conda shell.bash hook)"
conda activate research

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to activate conda environment 'research'"
    echo "   Please ensure the environment exists: conda create -n research"
    exit 1
fi

echo "✅ Conda environment activated"
echo ""

# Run the test script with all arguments
exec python3 "$TEST_SCRIPT" "$@"
