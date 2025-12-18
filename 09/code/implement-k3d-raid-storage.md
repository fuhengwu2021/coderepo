# 方案 3 实现指南：将 k3d 数据存储到 /raid

## 问题分析

k3d 的数据存储结构：
- k3d 容器内的 overlay 文件系统（根文件系统）存储在 `/var/lib/docker/overlay2`
- k3d 的 k3s 数据存储在 Docker volume 中（如 `k3d-mycluster-gpu-images`）
- 这些都在主机的 `/dev/sda1` 上

## 实现方案

由于 k3d 容器内的 overlay 文件系统由 Docker 管理，我们有两个选择：

### 方案 A：调整 kubelet eviction 阈值（推荐，只影响 k3d）

通过 k3d 的 k3s 配置调整 kubelet 的磁盘压力阈值，让它容忍更高的磁盘使用率。

### 方案 B：移动 Docker volume 到 /raid（更彻底，但需要额外步骤）

将 k3d 相关的 Docker volume 数据移到 /raid。

## 步骤 1：备份当前配置

```bash
# 备份 kubeconfig
cp ~/.kube/config ~/.kube/config.backup.$(date +%Y%m%d_%H%M%S)

# 检查当前集群状态
kubectl get nodes
kubectl describe nodes | grep -A 2 "DiskPressure"
```

## 步骤 2：选择实现方案

### 选择方案 A（推荐）

继续执行 `step-2a-adjust-kubelet-threshold.sh`

### 选择方案 B（更彻底）

继续执行 `step-2b-move-docker-volumes.sh`
