#!/bin/bash
# GCP Setup Script for Multimodal Video Recommender Training
#
# Prerequisites:
# 1. Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install
# 2. Run: gcloud auth login
# 3. Run: gcloud config set project YOUR_PROJECT_ID

set -e

# ============ Configuration ============
PROJECT_ID=$(gcloud config get-value project)
ZONE="us-central1-a"  # Has good GPU availability
INSTANCE_NAME="video-recommender-training"
MACHINE_TYPE="n1-standard-4"  # 4 vCPUs, 15GB RAM
GPU_TYPE="nvidia-tesla-t4"    # Good balance of cost/performance
GPU_COUNT=1

echo "=== GCP Training Setup ==="
echo "Project: $PROJECT_ID"
echo "Zone: $ZONE"
echo "Instance: $INSTANCE_NAME"
echo ""

# ============ Check GPU Quota ============
echo "Checking GPU quota..."
echo "Note: You may need to request GPU quota increase at:"
echo "https://console.cloud.google.com/iam-admin/quotas"
echo ""

# ============ Create VM with GPU ============
echo "Creating VM with GPU..."
gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --accelerator=type=$GPU_TYPE,count=$GPU_COUNT \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True"

echo ""
echo "=== VM Created Successfully ==="
echo ""
echo "To connect:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "To copy files:"
echo "  gcloud compute scp --recurse ./src $INSTANCE_NAME:~/project/ --zone=$ZONE"
echo ""
echo "To stop (save money when not training):"
echo "  gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "To delete:"
echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "Estimated cost: ~$0.35/hour for T4 GPU"
echo "Your $300 credits = ~850 hours of training!"
