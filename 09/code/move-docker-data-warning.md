# 移动 Docker 数据目录的影响分析

## ⚠️ 重要警告

**移动 Docker 数据目录会影响整个系统上的所有 Docker 用户和容器！**

## 影响范围

### 1. 系统级影响
- Docker 是系统级服务，所有用户共享同一个 Docker daemon
- 移动数据目录会影响**所有用户**的容器和镜像
- 需要停止 Docker 服务，所有容器都会停止

### 2. 当前运行的服务
- k3d 集群（k3d-mycluster-gpu）: **会停止，需要重新启动**
- vllm-serve: **会停止**
- exciting_mayer (VSCode dev container): **会停止**
- buildx_buildkit_builder0: **会停止**

### 3. 数据迁移
- 需要移动所有 Docker 数据（镜像、容器、卷等）
- 迁移时间取决于数据量大小
- 迁移过程中 Docker 服务不可用

## 更安全的替代方案

### 方案 1: 调整 kubelet eviction 阈值（推荐）
只修改 kubelet 的磁盘压力阈值，不移动 Docker 数据：
- 优点：不影响其他容器和服务
- 缺点：只是临时解决，不解决根本问题

### 方案 2: 清理 Docker 资源
清理未使用的镜像和容器，释放空间：
- 可以释放约 309GB 空间
- 不影响正在运行的服务
- 降低 /dev/sda1 使用率

**执行清理：**
```bash
# 使用提供的脚本（推荐）
./09/code/cleanup-docker.sh

# 或手动执行
docker image prune -a -f      # 清理未使用的镜像
docker container prune -f      # 清理已停止的容器
docker volume prune -f        # 清理未使用的卷
```

### 方案 3: 使用 bind mount（仅 k3d）
重新创建 k3d 集群时，将 Docker 数据目录 bind mount 到 /raid：
- 只影响 k3d 集群
- 不影响其他 Docker 容器
- 需要重新创建 k3d 集群

### 方案 4: 移动整个 Docker 数据目录
**⚠️ 高风险操作**
- 停止所有 Docker 容器和服务
- 移动 /var/lib/docker 到 /raid/tmpdata/docker
- 修改 Docker daemon 配置
- 重启 Docker 服务
- 影响所有用户和所有容器

## 建议

**优先考虑方案 2（清理 Docker 资源）**，如果还不够，再考虑方案 1（调整阈值）。
