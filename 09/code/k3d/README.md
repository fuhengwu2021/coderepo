# k3d GPU Cluster Setup

This directory contains all the code and scripts needed to set up a GPU-enabled Kubernetes cluster using k3d (k3s in Docker).

## Overview

k3d provides a lightweight, production-like Kubernetes environment that runs entirely in Docker containers. This setup creates a custom k3s image with NVIDIA GPU support, allowing you to run GPU workloads locally for development and testing.

## Prerequisites

Before starting, ensure you have:

1. **NVIDIA Drivers** installed and working (`nvidia-smi` should work)
2. **Docker** installed and running
3. **k3d** installed (will be installed by the setup script)
4. **NVIDIA Container Toolkit** (will be installed by the setup script)

## Quick Start

### 1. Install Prerequisites

```bash
chmod +x install-prerequisites.sh
./install-prerequisites.sh
```

This script will:
- Verify NVIDIA drivers
- Install NVIDIA Container Toolkit
- Install k3d
- Configure Docker runtime

### 2. Build Custom k3s-cuda Image

The default k3s image doesn't include NVIDIA Container Toolkit support. We need to build a custom image:

```bash
chmod +x build.sh
./build.sh
```

**Options:**
- `K3S_TAG`: k3s version (default: `v1.33.6-k3s1`)
- `CUDA_TAG`: CUDA base image tag (default: `12.2.0-base-ubuntu22.04`)
- `IMAGE_NAME`: Output image name (default: `k3s-cuda:v1.33.6-cuda-12.2.0`)

**Example with custom versions:**
```bash
K3S_TAG=v1.33.6-k3s1 CUDA_TAG=12.4.1-base-ubuntu22.04 ./build.sh
```

**Note:** CUDA 12.2.0 and 12.4.1 are both tested and working. 12.2.0 is recommended.

### 3. Create the Cluster

```bash
chmod +x create-cluster.sh
./create-cluster.sh
```

**Options:**
- `CLUSTER_NAME`: Cluster name (default: `mycluster-gpu`)
- `IMAGE_NAME`: k3s-cuda image to use (default: `k3s-cuda:v1.33.6-cuda-12.2.0`)
- `GPUS`: GPU allocation (default: `all`)
  - `all`: Use all available GPUs
  - `device=4,5`: Use specific GPUs (e.g., GPU 4 and 5)
- `MODEL_PATH`: Optional path to mount model directory (e.g., `/raid/models`)

**Examples:**

```bash
# Use all GPUs
./create-cluster.sh

# Use specific GPUs
GPUS="device=4,5" ./create-cluster.sh

# Mount model directory
MODEL_PATH=/raid/models ./create-cluster.sh

# Custom cluster name
CLUSTER_NAME=my-gpu-cluster ./create-cluster.sh
```

### 4. Configure kubectl

```bash
chmod +x setup-kubectl.sh
./setup-kubectl.sh
```

This will:
- Merge k3d kubeconfig with your default config
- Fix any server address issues (0.0.0.0 -> 127.0.0.1)
- Verify cluster access

### 5. Verify GPU Access

```bash
chmod +x verify-gpu.sh
./verify-gpu.sh
```

This will:
- Check if NVIDIA device plugin is running
- Verify GPU resources on nodes
- Test GPU access with a test pod

## Files

- **Dockerfile**: Custom k3s-cuda image definition
- **device-plugin-daemonset.yaml**: NVIDIA device plugin manifest (automatically deployed)
- **gpu-test.yaml**: Simple GPU test pod
- **install-prerequisites.sh**: Install all prerequisites
- **build.sh**: Build the custom k3s-cuda image
- **create-cluster.sh**: Create the k3d cluster
- **setup-kubectl.sh**: Configure kubectl access
- **verify-gpu.sh**: Verify GPU visibility and access
- **cleanup.sh**: Delete cluster and optionally remove image

## GPU Allocation Options

k3d supports flexible GPU allocation:

```bash
# Use all GPUs
--gpus=all

# Use specific GPU (single)
--gpus "device=4"

# Use multiple specific GPUs
--gpus "device=4,5"

# Use GPU range (if supported)
--gpus "device=4-7"  # GPUs 4, 5, 6, 7
```

To see available GPUs:
```bash
nvidia-smi --query-gpu=index,gpu_name --format=csv
```

## Troubleshooting

### Issue: Device plugin reports "No devices found"

- Ensure the custom k3s image was built correctly
- Verify NVIDIA Container Toolkit is installed on the host
- Check that `--gpus` flag was used when creating the cluster
- Review device plugin logs: `kubectl logs -n kube-system -l name=nvidia-device-plugin-ds`

### Issue: Pod cannot access GPU

- Verify `runtimeClassName: nvidia` is set in pod spec
- Check GPU resource requests/limits are specified
- Ensure device plugin DaemonSet is running: `kubectl get ds -n kube-system nvidia-device-plugin-daemonset`

### Issue: kubectl connection refused (localhost:8080)

The kubeconfig may have an incorrect server address. Fix it:

```bash
KUBE_SERVER=$(kubectl config view -o jsonpath='{.clusters[?(@.name=="k3d-mycluster-gpu")].cluster.server}')
kubectl config set-cluster k3d-mycluster-gpu --server=$(echo $KUBE_SERVER | sed 's/0.0.0.0/127.0.0.1/')
```

### Issue: Build fails with "unknown flag: --exclude"

- Enable Docker BuildKit: `export DOCKER_BUILDKIT=1`
- Install buildx: `docker buildx version` or `docker plugin install --grant-all-permissions moby/buildx`
- Use `docker buildx build` instead of `docker build` (handled automatically by build.sh)

### Issue: Custom k3s image fails during cluster creation

If you see "exec /usr/bin/sh: no such file or directory", the custom image is missing `/usr/bin/sh`:

- **Root cause:** The custom k3s-cuda image was built from CUDA base and may not have included all necessary shell binaries
- **Solutions:**
  1. Fix the custom image: Ensure `/usr/bin/sh` (or symlink to `/bin/sh`) exists
  2. Use standard k3s: Use `rancher/k3s:v1.33.6-k3s1` and manually configure NVIDIA runtime (complex)
  3. Use production K8s: Use kubeadm, k0s, or managed Kubernetes for full GPU support

## Cleaning Up

To remove the cluster and optionally the custom image:

```bash
chmod +x cleanup.sh
./cleanup.sh
```

Or manually:

```bash
# Delete cluster
k3d cluster delete mycluster-gpu

# Remove custom image (optional)
docker rmi k3s-cuda:v1.33.6-cuda-12.2.0
```

## Next Steps

After setting up the cluster, you can:

1. **Deploy vLLM**: See the vLLM deployment examples in the chapter
2. **Deploy GPU workloads**: Use the cluster to test GPU-accelerated applications
3. **Test multi-model serving**: Deploy multiple models and use API gateway routing
4. **Practice production patterns**: Test canary deployments, A/B testing, and observability

## Reference

- k3d documentation: https://k3d.io/
- NVIDIA Container Toolkit: https://github.com/NVIDIA/nvidia-container-toolkit
- k3s documentation: https://k3s.io/
- NVIDIA Device Plugin: https://github.com/NVIDIA/k8s-device-plugin
