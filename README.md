# Multimodal Short Video Recommender System

A deep learning-based video recommendation system that uses multimodal embeddings (video, audio, and text) to recommend similar short videos. Built with PyTorch, HuggingFace Transformers, and FastAPI.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Two-Tower Architecture                        │
├─────────────────────────────────┬───────────────────────────────────┤
│          Query Tower            │         Candidate Tower           │
│                                 │                                   │
│  ┌─────────┐ ┌─────────┐ ┌────┐│  ┌─────────┐ ┌─────────┐ ┌────┐  │
│  │ ResNet  │ │Wav2Vec2 │ │BERT││  │ ResNet  │ │Wav2Vec2 │ │BERT│  │
│  │ (Video) │ │ (Audio) │ │Text││  │ (Video) │ │ (Audio) │ │Text│  │
│  └────┬────┘ └────┬────┘ └──┬─┘│  └────┬────┘ └────┬────┘ └──┬─┘  │
│       │           │         │  │       │           │         │     │
│       └───────────┼─────────┘  │       └───────────┼─────────┘     │
│                   │            │                   │               │
│           ┌───────▼───────┐    │           ┌───────▼───────┐       │
│           │  GatedFusion  │    │           │  GatedFusion  │       │
│           └───────┬───────┘    │           └───────┬───────┘       │
│                   │            │                   │               │
│              512-d embed       │              512-d embed          │
└──────────────────┬─────────────┴───────────────────┬───────────────┘
                   │                                 │
                   └──────── Cosine Similarity ──────┘
                                    │
                            Recommendation Score
```

## Features

- **Multimodal Understanding**: Combines visual (CNN), audio (Wav2Vec2), and text (BERT) features
- **Transfer Learning**: Leverages pretrained models from ImageNet, LibriSpeech, and Wikipedia
- **Gated Fusion**: Learns optimal modality weights for each video
- **Contrastive Learning**: InfoNCE loss with in-batch negatives
- **GPU Support**: CUDA (NVIDIA), MPS (Apple Silicon), and CPU
- **REST API**: FastAPI endpoints for indexing and recommendations
- **Vector Database**: PostgreSQL with pgvector for similarity search
- **Docker Support**: Containerized deployment with docker-compose

## Models Used

| Component | Model | Pretrained On | Parameters |
|-----------|-------|---------------|------------|
| Video Encoder | ResNet-50 | ImageNet (1.4M images) | 23M |
| Audio Encoder | Wav2Vec2 | LibriSpeech (960h audio) | 95M |
| Text Encoder | BERT-base | Wikipedia + BookCorpus | 110M |
| Fusion | GatedFusion | Trained from scratch | 3K |
| **Total** | | | **~231M** (3M trainable) |

## Project Structure

```
multimodal-video-recommender/
├── src/
│   ├── features/           # Feature extractors
│   │   ├── video_encoder.py    # ResNet-50 CNN for frames
│   │   ├── audio_encoder.py    # Wav2Vec2 for audio
│   │   └── text_encoder.py     # BERT for text
│   │
│   ├── models/             # Model architectures
│   │   ├── fusion.py           # Concat, Gated, Attention fusion
│   │   └── two_tower.py        # Two-tower recommendation model
│   │
│   ├── data/               # Data loading
│   │   ├── processors.py       # Video/Audio preprocessing
│   │   ├── dataset.py          # PyTorch Dataset class
│   │   └── microlens_dataset.py # MicroLens dataset loader
│   │
│   ├── training/           # Training utilities
│   │   ├── losses.py           # InfoNCE, Triplet, Contrastive
│   │   └── trainer.py          # Training loop with checkpoints
│   │
│   ├── api/                # REST API
│   │   ├── app.py              # FastAPI app (text-only demo)
│   │   ├── app_v2.py           # FastAPI app (real video processing)
│   │   └── schemas.py          # Pydantic schemas
│   │
│   ├── db/                 # Database layer
│   │   ├── config.py           # Database configuration
│   │   ├── models.py           # SQLAlchemy models
│   │   ├── database.py         # Connection management
│   │   └── repository.py       # CRUD operations
│   │
│   └── utils/              # Utilities
│       └── device.py           # CUDA/MPS/CPU device selection
│
├── scripts/
│   ├── train_cloud.py      # Cloud training script
│   ├── gcp_setup.sh        # GCP VM creation
│   └── vm_setup.sh         # VM environment setup
│
├── data/
│   └── raw/microlens/      # MicroLens dataset
│       ├── MicroLens-50k_titles.csv
│       ├── MicroLens-50k_pairs.csv
│       └── videos/
│
├── checkpoints/            # Saved model weights
├── logs/                   # TensorBoard logs
├── docker-compose.yml      # Docker services
├── Dockerfile              # Container build
└── requirements.txt        # Python dependencies
```

## Installation

### Local Development

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/multimodal-video-recommender.git
cd multimodal-video-recommender

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install system dependencies (Mac)
brew install ffmpeg

# Install system dependencies (Ubuntu)
sudo apt-get install ffmpeg libsndfile1 libgl1-mesa-glx
```

### Docker Deployment

```bash
# Build and start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

## Usage

### Quick Start - API Server

```bash
# Start the API server
cd src/api
python -m uvicorn app_v2:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and model status |
| `/index/microlens` | POST | Index videos from MicroLens dataset |
| `/index` | POST | Index a single video |
| `/recommend` | POST | Get recommendations for a video |
| `/index/list` | GET | List all indexed videos |

### Example API Calls

```bash
# Health check
curl http://localhost:8000/health

# Index 10 videos from MicroLens
curl -X POST http://localhost:8000/index/microlens \
  -H "Content-Type: application/json" \
  -d '{"max_videos": 10}'

# Get recommendations
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"video_id": "1002", "top_k": 5}'

# List indexed videos
curl http://localhost:8000/index/list
```

### Python SDK Usage

```python
from src.models.two_tower import TwoTowerModel
from src.data.processors import VideoProcessor, AudioProcessor
from src.utils.device import get_device
import torch

# Initialize
device = get_device()  # Auto-selects CUDA/MPS/CPU
model = TwoTowerModel(feature_dim=512).to(device)
video_proc = VideoProcessor(num_frames=8)
audio_proc = AudioProcessor(duration=5.0)

# Process a video
frames = video_proc.extract_frames("video.mp4")  # (8, 3, 224, 224)
audio = audio_proc.extract_audio("video.mp4")    # (80000,)

# Get embedding
with torch.no_grad():
    embedding = model.encode_video(
        frames.unsqueeze(0).to(device),
        audio.unsqueeze(0).to(device),
        ["Video title"]
    )  # (1, 512)
```

## Training

### Local Training (CPU/MPS)

```bash
python scripts/train_cloud.py --epochs 5 --max_videos 100 --batch_size 4
```

### Cloud Training (GCP with GPU)

```bash
# 1. Create GCP VM with T4 GPU
bash scripts/gcp_setup.sh

# 2. SSH into VM
gcloud compute ssh video-recommender-training --zone=us-east1-b

# 3. Setup VM environment
bash scripts/vm_setup.sh

# 4. Run training
python3 scripts/train_cloud.py --download --epochs 10 --max_videos 500

# 5. Monitor with TensorBoard
tensorboard --logdir=logs --port=6006 --bind_all
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 10 | Number of training epochs |
| `--batch_size` | 8 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--max_videos` | 500 | Max videos to use |
| `--download` | False | Download dataset first |
| `--data_dir` | data/raw/microlens | Dataset directory |

## Dataset

This project uses the [MicroLens](https://recsys.westlake.edu.cn/MicroLens-Dataset/) dataset:

- **50K version**: 50,000 short videos with titles
- **Source**: Short video platform (TikTok-like)
- **Contents**: Video files, titles, user interaction pairs

### Dataset Structure

```
data/raw/microlens/
├── MicroLens-50k_titles.csv    # video_id, title
├── MicroLens-50k_pairs.csv     # user_id, video_id (interactions)
└── videos/
    ├── 1001.mp4
    ├── 1002.mp4
    └── ...
```

## Technical Details

### Video Processing Pipeline

1. **Frame Extraction**: Sample 8 frames uniformly from video
2. **Frame Resize**: Resize to 224x224 RGB
3. **CNN Encoding**: ResNet-50 → 2048-d per frame
4. **Temporal Pooling**: Mean pool across frames → 2048-d
5. **Projection**: Linear layer → 512-d

### Audio Processing Pipeline

1. **Audio Extraction**: Extract audio track from video
2. **Resampling**: Resample to 16kHz mono
3. **Duration**: Take first 5 seconds (pad if shorter)
4. **Wav2Vec2 Encoding**: → 768-d per timestep
5. **Temporal Pooling**: Mean pool → 768-d
6. **Projection**: Linear layer → 512-d

### Text Processing Pipeline

1. **Tokenization**: BERT WordPiece tokenizer
2. **Encoding**: BERT → 768-d [CLS] embedding
3. **Projection**: Linear layer → 512-d

### Gated Fusion

```python
# Learnable gate weights
gate = softmax([w_video, w_audio, w_text])

# Weighted combination
fused = gate[0] * video_emb + gate[1] * audio_emb + gate[2] * text_emb
```

### Contrastive Learning (InfoNCE)

```python
# Similarity matrix
sim = query_emb @ candidate_emb.T / temperature

# InfoNCE loss (in-batch negatives)
loss = CrossEntropy(sim, labels)  # labels = [0, 1, 2, ..., B-1]
```

## Device Support

The system automatically selects the best available device:

```python
from src.utils.device import get_device

device = get_device()
# Returns: cuda (NVIDIA GPU) > mps (Apple Silicon) > cpu
```

## Performance

| Hardware | Batch Size | Videos/sec | Training Time (500 videos, 10 epochs) |
|----------|------------|------------|--------------------------------------|
| NVIDIA T4 | 8 | ~15 | ~1-2 hours |
| Apple M1 Pro | 4 | ~5 | ~4-5 hours |
| CPU (8 cores) | 2 | ~1 | ~15-20 hours |

## License

MIT License

## Acknowledgments

- [MicroLens Dataset](https://recsys.westlake.edu.cn/MicroLens-Dataset/) - Westlake University
- [HuggingFace Transformers](https://huggingface.co/transformers/) - BERT, Wav2Vec2
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [FastAPI](https://fastapi.tiangolo.com/) - REST API framework
