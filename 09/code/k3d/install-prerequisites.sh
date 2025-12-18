#!/bin/bash
set -e

echo "Installing prerequisites for k3d GPU cluster setup..."

# 1. Check NVIDIA Drivers
echo "Checking NVIDIA drivers..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi
nvidia-smi
echo "✓ NVIDIA drivers installed"

# 2. Install NVIDIA Container Toolkit
echo "Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
echo "✓ NVIDIA Container Toolkit installed"

# 3. Install k3d
echo "Installing k3d..."
if ! command -v k3d &> /dev/null; then
    curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
else
    echo "k3d already installed"
fi

# Verify installation
k3d --version
echo "✓ k3d installed"

echo ""
echo "Prerequisites installation complete!"
echo ""
echo "Next steps:"
echo "1. Build the custom k3s-cuda image: ./build.sh"
echo "2. Create the cluster: ./create-cluster.sh"
