#!/bin/bash
# Complete Slurm setup script for single-node multi-GPU configuration
# This script sets up a single-node multi-GPU cluster
# Replace SLURM_HOME with your Slurm installation path (e.g., /opt/slurm or $HOME/slurm)
# Replace HOSTNAME with your actual hostname (e.g., $(hostname))
# Replace USERNAME with your actual username (e.g., $USER)

set -e

# Detect values automatically, but allow override via environment variables
SLURM_HOME=${SLURM_HOME:-${HOME}/slurm}
HOSTNAME=${SLURM_HOSTNAME:-$(hostname -s)}
USERNAME=${SLURM_USER:-$(whoami)}

echo "=========================================="
echo "Slurm Single-Node Setup Script"
echo "=========================================="

echo ""
echo "=== Step 0: Setup environment and verify prerequisites ==="

# CRITICAL: Set PATH to use compiled Slurm, not system version
export PATH=$SLURM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SLURM_HOME/lib:$LD_LIBRARY_PATH

# Verify Slurm binaries exist
if [ ! -f "$SLURM_HOME/sbin/slurmctld" ]; then
    echo "❌ ERROR: slurmctld not found at $SLURM_HOME/sbin/slurmctld"
    echo "   Please ensure Slurm is compiled and installed at $SLURM_HOME"
    exit 1
fi

if [ ! -f "$SLURM_HOME/sbin/slurmd" ]; then
    echo "❌ ERROR: slurmd not found at $SLURM_HOME/sbin/slurmd"
    exit 1
fi

# Verify GPU devices exist
for gpu in 6 7; do
    if [ ! -e "/dev/nvidia${gpu}" ]; then
        echo "❌ ERROR: GPU device /dev/nvidia${gpu} not found"
        echo "   Available GPUs:"
        ls -la /dev/nvidia* 2>/dev/null || echo "   No NVIDIA devices found"
        exit 1
    fi
done

# Verify Munge is running (required for authentication)
if ! systemctl is-active --quiet munge 2>/dev/null && ! pgrep -x munged > /dev/null; then
    echo "⚠️  WARNING: Munge daemon not running"
    echo "   Slurm authentication may fail. Start with: sudo systemctl start munge"
    if [ -t 0 ]; then
        # Interactive mode - ask user
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        # Non-interactive mode - continue with warning
        echo "   Continuing in non-interactive mode..."
    fi
fi

# Enable systemd lingering for Slurm user (fixes systemd scope permission issues)
echo ""
echo "=== Step 0.5: Enable systemd lingering (fixes cgroup v2 systemd scope permissions) ==="
SLURM_USER=$(whoami)
if ! loginctl show-user "$SLURM_USER" 2>/dev/null | grep -q "Linger=yes"; then
    echo "⚠️  systemd lingering not enabled for user $SLURM_USER"
    echo "   This is required for slurmd to create systemd scopes with cgroup v2"
    if [ -t 0 ]; then
        read -p "Enable lingering now? (requires sudo) [Y/n] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            if sudo loginctl enable-linger "$SLURM_USER" 2>/dev/null; then
                echo "✅ systemd lingering enabled for $SLURM_USER"
            else
                echo "⚠️  Failed to enable lingering (may need to run manually: sudo loginctl enable-linger $SLURM_USER)"
            fi
        else
            echo "⚠️  Skipping lingering setup - slurmd may fail to start with cgroup v2"
        fi
    else
        echo "⚠️  Non-interactive mode - cannot enable lingering automatically"
        echo "   Run manually: sudo loginctl enable-linger $SLURM_USER"
    fi
else
    echo "✅ systemd lingering already enabled for $SLURM_USER"
fi

# Verify we can write to directories
if [ ! -w "$SLURM_HOME" ]; then
    echo "❌ ERROR: Cannot write to $SLURM_HOME"
    exit 1
fi

# Verify cgroup v2 plugin exists (if using cgroup)
if [ -f "$SLURM_HOME/etc/cgroup.conf" ] && grep -q "CgroupPlugin=cgroup/v2" "$SLURM_HOME/etc/cgroup.conf" 2>/dev/null; then
    if [ ! -f "$SLURM_HOME/lib/slurm/cgroup_v2.so" ]; then
        echo "⚠️  WARNING: cgroup.conf specifies cgroup/v2, but cgroup_v2.so plugin not found"
        echo "   Slurm may need to be recompiled with --enable-cgroupv2"
        echo "   See: code/chapter8/recompile_with_cgroupv2.sh"
    else
        echo "✅ cgroup v2 plugin found"
    fi
fi

echo "✅ Prerequisites verified"

echo ""
echo "=== Step 1: Create directory structure ==="
mkdir -p $SLURM_HOME/var/{log,state,spool}
# Only create directories for node6 and node7 (GPU6 and GPU7)
for i in 6 7; do
    mkdir -p $SLURM_HOME/var/spool/node${i}
done
echo "✅ Directories created for node6 and node7"

echo ""
echo "=== Step 2: Create gres.conf ==="
cat > $SLURM_HOME/etc/gres.conf << 'GRESEOF'
# GPU to node mapping - Using GPU 6 and GPU 7
NodeName=node6 Name=gpu File=/dev/nvidia6
NodeName=node7 Name=gpu File=/dev/nvidia7
GRESEOF
echo "✅ gres.conf created for GPU6 and GPU7"

echo ""
echo "=== Step 3: Create cgroup.conf ==="
# Since we're using proctrack/linux (not proctrack/cgroup), cgroup is OPTIONAL
# Cgroup is only used for resource constraints (CPU/memory limits), not process tracking
# Disabling cgroup avoids systemd scope permission issues
# If you want cgroup constraints and can use sudo, you can enable it later

# Check if systemd lingering is enabled (needed for cgroup v2 systemd scopes)
SLURM_USER=$(whoami)
LINGERING_ENABLED=false
if loginctl show-user "$SLURM_USER" 2>/dev/null | grep -q "Linger=yes"; then
    LINGERING_ENABLED=true
fi

if [ "$LINGERING_ENABLED" = true ]; then
    # User has lingering enabled - can use cgroup v2
    # The plugin was compiled with --enable-cgroupv2, so it's available
    cat > $SLURM_HOME/etc/cgroup.conf << 'CGROUPEOF'
# Cgroup v2 configuration
# Slurm was compiled with --enable-cgroupv2, so the plugin is available
# systemd lingering is enabled, so systemd scopes should work
CgroupPlugin=cgroup/v2
ConstrainCores=yes
ConstrainDevices=yes
ConstrainRAMSpace=yes
ConstrainSwapSpace=yes
# Note: CgroupAutomount is defunct in Slurm 25.11.0, removed
CGROUPEOF
    echo "✅ cgroup.conf created - cgroup v2 ENABLED (plugin compiled, lingering enabled)"
else
    # No lingering - explicitly DISABLE cgroup plugin to prevent auto-detection
    # This prevents Slurm from trying to use systemd scopes
    cat > $SLURM_HOME/etc/cgroup.conf << 'CGROUPEOF'
# Cgroup plugin explicitly DISABLED
# Slurm was compiled with --enable-cgroupv2, but we disable it here to avoid
# systemd scope permission issues. Slurm will work perfectly without cgroup.
CgroupPlugin=disabled
#
# To enable cgroup v2 later (requires systemd lingering):
# 1. Enable lingering: sudo loginctl enable-linger $(whoami)
# 2. Change CgroupPlugin=disabled to CgroupPlugin=cgroup/v2
# 3. Add constraint lines below
# 4. Restart slurmd daemons
#
# CgroupPlugin=cgroup/v2
# ConstrainCores=yes
# ConstrainDevices=yes
# ConstrainRAMSpace=yes
# ConstrainSwapSpace=yes
#
# Note: Slurm works perfectly without cgroup for:
#   - Job scheduling and allocation
#   - GPU allocation (via GRES)
#   - Multi-node jobs
#   - Process tracking (via proctrack/linux)
# Cgroup only adds resource limit enforcement (optional)
CGROUPEOF
    echo "✅ cgroup.conf created - cgroup v2 EXPLICITLY DISABLED (no systemd scopes needed)"
fi

echo ""
echo "=== Step 4: Backup old slurm.conf ==="
if [ -f $SLURM_HOME/etc/slurm.conf ]; then
    cp $SLURM_HOME/etc/slurm.conf $SLURM_HOME/etc/slurm.conf.backup.$(date +%Y%m%d_%H%M%S)
    echo "✅ Backed up old slurm.conf"
fi

echo ""
echo "=== Step 5: Create new slurm.conf for 2 nodes ==="
cat > $SLURM_HOME/etc/slurm.conf << CONFEOF
# Slurm configuration for single-node setup with 2 GPUs
# Generated by slurm_setup.sh
# Hostname: $HOSTNAME
# Username: $USERNAME
# Slurm Home: $SLURM_HOME
ClusterName=distributed-ai
SlurmctldHost=$HOSTNAME

# Authentication
AuthType=auth/munge
CryptoType=crypto/munge

# Paths and files - using %n for per-node substitution
SlurmctldPidFile=$SLURM_HOME/var/state/slurmctld.pid
SlurmctldPort=6817
SlurmdPidFile=$SLURM_HOME/var/spool/%n/slurmd.pid
SlurmdPort=6818
SlurmdSpoolDir=$SLURM_HOME/var/spool/%n
StateSaveLocation=$SLURM_HOME/var/state

# Logging
SlurmctldDebug=info
SlurmctldLogFile=$SLURM_HOME/var/log/slurmctld.log
SlurmdDebug=info
SlurmdLogFile=$SLURM_HOME/var/log/slurmd.%n.log

# Process tracking and task binding
# Using proctrack/pgid for virtual nodes (doesn't require matching hardware topology)
# proctrack/linux requires exact hardware match, which fails with virtual nodes
ProctrackType=proctrack/pgid
TaskPlugin=task/affinity

# Scheduling
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory

# Resource limits
DefMemPerCPU=4000
MaxMemPerCPU=0

# Timeouts
SlurmctldTimeout=120
SlurmdTimeout=300
InactiveLimit=0
MinJobAge=300
KillWait=30
Waittime=0

# User
SlurmUser=$USERNAME
SlurmdUser=$USERNAME

# MPI support for distributed training
# Using pmi2 - PMIx plugin requires Slurm recompile with --with-pmix
# To use PMIx: recompile Slurm with: ./configure ... --with-pmix
MpiDefault=pmi2

# GRES
GresTypes=gpu

# 2 virtual nodes
# Each node uses different port for multiple slurmd support
# Adjust CPUs and RealMemory based on your hardware
# For virtual nodes, let Slurm auto-detect hardware topology
# Each virtual node will share the same physical hardware
NodeName=node6 NodeHostname=$HOSTNAME Port=17016 \
    CPUs=112 RealMemory=240000 Gres=gpu:1 State=UNKNOWN

NodeName=node7 NodeHostname=$HOSTNAME Port=17017 \
    CPUs=112 RealMemory=240000 Gres=gpu:1 State=UNKNOWN

# GPU partition with 2 nodes
PartitionName=gpu Nodes=node6,node7 Default=YES MaxTime=INFINITE State=UP OverSubscribe=NO
CONFEOF
echo "✅ slurm.conf created with 2 nodes (node6, node7)"

echo ""
echo "=== Step 6: Stop existing Slurm daemons and check ports ==="
pkill slurmctld 2>/dev/null || true
pkill slurmd 2>/dev/null || true
sleep 3
echo "✅ Old daemons stopped"

# Check if ports are available
check_port() {
    local port=$1
    if command -v netstat >/dev/null 2>&1; then
        if netstat -tuln 2>/dev/null | grep -q ":${port} "; then
            return 1
        fi
    elif command -v ss >/dev/null 2>&1; then
        if ss -tuln 2>/dev/null | grep -q ":${port} "; then
            return 1
        fi
    fi
    return 0
}

if ! check_port 6817; then
    echo "⚠️  WARNING: Port 6817 (slurmctld) is in use"
fi

for port in 17016 17017; do
    if ! check_port $port; then
        echo "⚠️  WARNING: Port $port is in use - slurmd may fail to start"
    fi
done

echo ""
echo "=== Step 7: Validate slurm.conf ==="
# Check if config file exists and is readable
if [ ! -f "$SLURM_HOME/etc/slurm.conf" ]; then
    echo "❌ ERROR: slurm.conf not found at $SLURM_HOME/etc/slurm.conf"
    exit 1
fi

# Basic syntax check - look for common issues
if ! grep -q "ClusterName=" "$SLURM_HOME/etc/slurm.conf"; then
    echo "⚠️  WARNING: ClusterName not found in slurm.conf"
fi

if ! grep -q "NodeName=node6" "$SLURM_HOME/etc/slurm.conf"; then
    echo "⚠️  WARNING: node6 not found in slurm.conf"
fi

if ! grep -q "NodeName=node7" "$SLURM_HOME/etc/slurm.conf"; then
    echo "⚠️  WARNING: node7 not found in slurm.conf"
fi

echo "✅ slurm.conf basic checks passed (full validation will occur when daemons start)"

echo ""
echo "=== Step 8: Start slurmctld (controller) ==="
$SLURM_HOME/sbin/slurmctld
sleep 3
if pgrep -f "slurmctld" > /dev/null; then
    echo "✅ slurmctld started successfully (PID: $(pgrep -f slurmctld))"
    # Check if it's actually responding
    sleep 2
    if [ -f "$SLURM_HOME/bin/sinfo" ]; then
        if timeout 5 $SLURM_HOME/bin/sinfo >/dev/null 2>&1; then
            echo "✅ slurmctld is responding to queries"
        else
            echo "⚠️  WARNING: slurmctld started but not responding. Check logs:"
            echo "   tail -20 $SLURM_HOME/var/log/slurmctld.log"
        fi
    fi
else
    echo "❌ slurmctld failed to start. Check logs:"
    echo "   tail -20 $SLURM_HOME/var/log/slurmctld.log"
    exit 1
fi

echo ""
echo "=== Step 9: Start slurmd daemons (compute nodes) ==="
# Only start slurmd for node6 and node7
failed_nodes=()
for i in 6 7; do
    if $SLURM_HOME/sbin/slurmd -N node${i}; then
        sleep 2
        if pgrep -f "slurmd.*node${i}" > /dev/null; then
            echo "✅ slurmd started for node${i} (port 1701${i}, GPU${i}, PID: $(pgrep -f "slurmd.*node${i}"))"
        else
            echo "❌ slurmd for node${i} failed to start"
            failed_nodes+=("node${i}")
        fi
    else
        echo "❌ Failed to start slurmd for node${i}"
        failed_nodes+=("node${i}")
    fi
done

if [ ${#failed_nodes[@]} -gt 0 ]; then
    echo "⚠️  WARNING: Failed to start slurmd for: ${failed_nodes[*]}"
    echo "   Check logs:"
    for node in "${failed_nodes[@]}"; do
        echo "   tail -20 $SLURM_HOME/var/log/slurmd.${node}.log"
    done
else
    echo "✅ 2 slurmd daemons started for GPU6 and GPU7"
fi

echo ""
echo "=== Step 10: Verify daemons and cluster status ==="
echo "Controller:"
if pgrep -f slurmctld > /dev/null; then
    pgrep -a -f slurmctld
else
    echo "❌ slurmctld not running"
fi
echo ""
echo "Compute nodes:"
if pgrep -f slurmd > /dev/null; then
    pgrep -a -f slurmd
else
    echo "❌ No slurmd running"
fi

echo ""
echo "Cluster status:"
if [ -f "$SLURM_HOME/bin/sinfo" ]; then
    if timeout 5 $SLURM_HOME/bin/sinfo 2>/dev/null; then
        echo ""
        echo "Node details:"
        timeout 5 $SLURM_HOME/bin/scontrol show nodes 2>/dev/null | head -20 || echo "   (Could not retrieve node details)"
    else
        echo "⚠️  Could not query cluster status (sinfo failed)"
    fi
else
    echo "⚠️  sinfo not found at $SLURM_HOME/bin/sinfo"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Cluster configured with:"
echo "  - 2 virtual nodes: node6, node7"
echo "  - 2 GPUs: GPU6, GPU7"
echo "  - Ports: 17016, 17017"
echo ""
echo "Next steps:"
echo "1. PATH is already set in this session, but add to ~/.bashrc for persistence:"
echo "   echo 'export PATH=$SLURM_HOME/bin:\$PATH' >> ~/.bashrc"
echo "   echo 'export LD_LIBRARY_PATH=$SLURM_HOME/lib:\$LD_LIBRARY_PATH' >> ~/.bashrc"
echo ""
echo "2. Check cluster status:"
echo "   sinfo"
echo "   Expected: 2 nodes (node6, node7) in idle state"
echo ""
echo "3. Test simple job:"
echo "   srun -N 1 hostname"
echo "   srun -N 2 hostname"
echo ""
echo "4. Test GPU allocation:"
echo "   srun -N 1 --gres=gpu:1 nvidia-smi -L"
echo ""
echo "5. View logs if issues:"
echo "   tail -f $SLURM_HOME/var/log/slurmctld.log"
echo "   tail -f $SLURM_HOME/var/log/slurmd.node6.log"
echo "   tail -f $SLURM_HOME/var/log/slurmd.node7.log"
echo ""

