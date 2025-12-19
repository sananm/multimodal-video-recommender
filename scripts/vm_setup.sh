#!/bin/bash
# Run this ON the GCP VM after connecting via SSH
# Usage: bash vm_setup.sh

set -e

echo "=== Setting up GCP VM for Training ==="

# Update system
sudo apt-get update

# Install system dependencies
sudo apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0

# Check NVIDIA GPU
echo ""
echo "=== GPU Status ==="
nvidia-smi

# Check PyTorch CUDA
echo ""
echo "=== PyTorch CUDA Status ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Install Python dependencies
echo ""
echo "=== Installing Python Dependencies ==="
pip3 install --upgrade pip
pip3 install \
    transformers \
    librosa \
    opencv-python \
    pandas \
    tqdm \
    tensorboard

# Create project directory
mkdir -p ~/project
cd ~/project

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Upload your code:"
echo "   gcloud compute scp --recurse ./src ./scripts ./requirements.txt VM_NAME:~/project/ --zone=ZONE"
echo ""
echo "2. Download dataset and train:"
echo "   cd ~/project"
echo "   python3 scripts/train_cloud.py --download --epochs 10 --max_videos 500"
echo ""
echo "3. Monitor training:"
echo "   tensorboard --logdir=logs --port=6006 --bind_all"
echo ""
echo "4. Download trained model:"
echo "   gcloud compute scp VM_NAME:~/project/checkpoints/best.pt ./checkpoints/ --zone=ZONE"
