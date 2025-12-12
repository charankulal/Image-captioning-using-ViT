# Image Captioning using Vision Transformer (ViT)

A complete deep learning system for automatically generating descriptive captions for images using Vision Transformer architecture, featuring a Flask web application with user authentication.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌟 Features

- **🎨 Web Application**: Full-featured Flask app with user authentication and beautiful UI
- **🤖 AI-Powered Captions**: Vision Transformer (ViT) + Transformer Decoder architecture
- **📊 Jupyter Notebooks**: Interactive training and inference notebooks
- **🔌 Easy-to-Use API**: Simple Python API for integration into other projects
- **⚡ GPU Acceleration**: CUDA support for fast caption generation
- **📈 Beam Search**: High-quality caption generation with beam search

## 🚀 Quick Start

### Option 1: Web Application (Recommended)

Run the Flask web app to upload images and generate captions through your browser:

**Windows:**
```cmd
start_flask_app.bat
```

**Linux/Mac/Git Bash:**
```bash
./start_flask_app.sh
```

Then open `http://localhost:5000` in your browser and start captioning!

📖 See [QUICKSTART_FLASK.md](QUICKSTART_FLASK.md) for detailed instructions.

### Option 2: Python API

```python
from src.model_manager import quick_caption

# Generate caption for an image (one-liner!)
caption = quick_caption('path/to/image.jpg')
print(caption)
```

### Option 3: Jupyter Notebooks

Open and run the interactive notebooks:
- **Training**: `notebooks/Image_Captioning_Training.ipynb`
- **Inference**: `notebooks/Image_Captioning_Inference.ipynb`

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Web Application](#web-application)
  - [Python API](#python-api)
  - [Jupyter Notebooks](#jupyter-notebooks)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a state-of-the-art image captioning system that combines:
- **Vision Transformer (ViT)** as the encoder to extract image features
- **Transformer Decoder** to generate natural language captions
- Pre-trained ViT from Google (`google/vit-base-patch16-224-in21k`)
- Trained on Flickr8k dataset with beam search decoding

### What's New

- ✨ **Flask Web Application** with user authentication and responsive UI
- 🔧 **Model Manager API** for easy integration
- 📓 **Interactive Notebooks** for training and testing
- 🐛 **Bug Fixes** for vocabulary loading and compatibility
- 📚 **Comprehensive Documentation** with quick start guides

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Image (224×224×3)                           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│            Vision Transformer (ViT) Encoder                     │
│  • Pre-trained ViT-Base (768 dim, 12 layers, 12 heads)        │
│  • Patch-based feature extraction (16×16 patches)              │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
              Image Features (197×768)
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│               Transformer Decoder                               │
│  • 6 decoder layers with 8 attention heads                     │
│  • Cross-attention to image features                           │
│  • Self-attention with causal masking                          │
│  • Positional encoding                                         │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
            Generated Caption Text
```

### Key Components

1. **Vision Transformer Encoder** (`src/model.py:34-69`)
   - Pre-trained ViT-Base from HuggingFace
   - Extracts 197 patch embeddings (196 patches + 1 CLS token)
   - Output: (batch_size, 197, 768) feature tensors

2. **Transformer Decoder** (`src/model.py:72-128`)
   - 6 decoder layers with 8 attention heads
   - Cross-attention mechanism to attend to image features
   - Self-attention with positional encoding
   - Causal masking for autoregressive generation

3. **Caption Generation** (`src/model.py:172-283`)
   - Greedy decoding: Fast, selects highest probability token
   - Beam search: Better quality, explores multiple hypothesis paths

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (optional, for GPU acceleration)
- 8GB+ VRAM recommended for training
- 4GB+ RAM for inference

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/charankulal/Image-captioning-using-ViT.git
cd Image-captioning-using-ViT

# Create virtual environment (recommended)
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate.bat

# Install all dependencies
pip install -r requirements.txt

# Install Flask dependencies (for web app)
pip install -r flask_app/requirements.txt
```

### Download Pre-trained Model

The Vision Transformer weights (~350MB) will be automatically downloaded from HuggingFace on first use.

## 📖 Usage

### Web Application

The Flask web application provides a user-friendly interface for image captioning:

**Features:**
- User registration and authentication
- Drag-and-drop image upload
- Real-time caption generation
- Caption history tracking
- Confidence scores
- Responsive design

**Start the server:**
```bash
# Windows
start_flask_app.bat

# Linux/Mac/Git Bash
./start_flask_app.sh

# Or manually
source .venv/Scripts/activate
python flask_app/run.py
```

**Access:** Open `http://localhost:5000` in your browser

📚 **Full Documentation:** [FLASK_APP_GUIDE.md](FLASK_APP_GUIDE.md)

### Python API

#### Quick Caption (One-Liner)

```python
from src.model_manager import quick_caption

caption = quick_caption('path/to/image.jpg')
print(caption)
```

#### Advanced Usage

```python
from src.model_manager import ImageCaptioningModelManager

# Initialize and load model
manager = ImageCaptioningModelManager(checkpoint_dir='models')
manager.load()

# Generate caption with beam search
caption = manager.generate_caption(
    'image.jpg',
    method='beam_search',
    beam_width=5
)
print(f"Caption: {caption}")

# Generate with confidence score
caption, confidence = manager.generate_caption(
    'image.jpg',
    method='beam_search',
    beam_width=5,
    return_confidence=True
)
print(f"Caption: {caption}")
print(f"Confidence: {confidence:.2%}")

# Batch processing
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
captions = manager.generate_captions_batch(images)
for img, cap in zip(images, captions):
    print(f"{img}: {cap}")
```

**Methods:**
- `greedy`: Fast decoding (~50ms per image on GPU)
- `beam_search`: Better quality (~200ms per image on GPU)

### Jupyter Notebooks

#### Training Notebook

Open `notebooks/Image_Captioning_Training.ipynb` to:
- Train the model from scratch
- Resume training from checkpoints
- Monitor training progress
- Visualize loss curves

#### Inference Notebook

Open `notebooks/Image_Captioning_Inference.ipynb` to:
- Load trained models
- Generate captions for images
- Compare greedy vs beam search
- Test on custom images

## 📁 Project Structure

```
Image-captioning-using-ViT/
├── 📱 flask_app/              # Flask web application
│   ├── app.py                 # Main Flask application
│   ├── run.py                 # Startup script
│   ├── database.db            # SQLite database (auto-created)
│   ├── templates/             # HTML templates
│   │   ├── base.html          # Base template
│   │   ├── signin.html        # Sign in page
│   │   ├── signup.html        # Sign up page
│   │   ├── home.html          # Home page with upload
│   │   └── history.html       # Caption history
│   ├── static/
│   │   ├── css/style.css      # Application styling
│   │   └── uploads/           # User uploaded images
│   ├── requirements.txt       # Flask dependencies
│   └── README.md              # Flask documentation
│
├── 🧠 src/                    # Source code modules
│   ├── __init__.py            # Package initializer
│   ├── model.py               # Model architecture (ViT + Transformer)
│   ├── vocabulary.py          # Vocabulary class
│   └── model_manager.py       # Model API for easy integration
│
├── 📓 notebooks/              # Jupyter notebooks
│   ├── Image_Captioning_Training.ipynb    # Training notebook
│   └── Image_Captioning_Inference.ipynb   # Inference notebook
│
├── 🤖 models/                 # Trained model checkpoints
│   ├── best_model.pth         # Best model checkpoint (~1.7GB)
│   └── vocab.pth              # Vocabulary object
│
├── 📊 data/                   # Dataset (not included)
│   ├── images/                # Training images
│   └── captions.csv           # Image-caption pairs
│
├── 🖼️ test_images/           # Sample test images
│
├── 📄 Documentation
│   ├── README.md              # This file
│   ├── CLAUDE.md              # Development guide for Claude Code
│   ├── FLASK_APP_GUIDE.md     # Complete Flask app guide
│   └── QUICKSTART_FLASK.md    # Quick start for Flask app
│
├── 🚀 Startup scripts
│   ├── start_flask_app.bat    # Windows startup script
│   └── start_flask_app.sh     # Linux/Mac startup script
│
├── requirements.txt           # Main dependencies
├── .gitignore                 # Git ignore rules
└── LICENSE                    # MIT License
```

## 🎯 Model Details

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Vision Transformer | ViT-Base | Pre-trained on ImageNet-21k |
| Model Source | `google/vit-base-patch16-224-in21k` | HuggingFace |
| Image Size | 224×224 | Standard ViT input size |
| Patch Size | 16×16 | Results in 14×14=196 patches |
| Embedding Dimension | 768 | Hidden size |
| Decoder Layers | 6 | Transformer decoder layers |
| Attention Heads | 8 | Multi-head attention |
| Feed-forward Expansion | 4× | FFN intermediate size = 3072 |
| Dropout | 0.1 | Regularization |
| Vocabulary Size | ~2,535 | Built from Flickr8k |
| Vocab Threshold | 5 occurrences | Min word frequency |
| Max Caption Length | 100 tokens | Maximum generation length |
| Beam Width | 5 | For beam search decoding |

### Training Configuration

| Setting | Value |
|---------|-------|
| Dataset | Flickr8k |
| Training Split | 72% |
| Validation Split | 8% |
| Test Split | 20% |
| Batch Size | 32 |
| Learning Rate | 3×10⁻⁴ |
| Optimizer | Adam (β₁=0.9, β₂=0.98) |
| Scheduler | ReduceLROnPlateau |
| Gradient Clipping | max_norm=1.0 |
| Epochs Trained | 3 |
| Best Val Loss | 2.69 |

### Data Augmentation

**Training:**
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation ±0.2)
- ImageNet normalization

**Validation/Testing:**
- Resize to 224×224
- ImageNet normalization only

### Special Tokens

| Token | Index | Purpose |
|-------|-------|---------|
| `<PAD>` | 0 | Padding for batch processing |
| `<SOS>` | 1 | Start of sequence |
| `<EOS>` | 2 | End of sequence |
| `<UNK>` | 3 | Unknown words (below frequency threshold) |

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | Main project documentation (this file) |
| [FLASK_APP_GUIDE.md](FLASK_APP_GUIDE.md) | Complete Flask app setup and usage guide |
| [QUICKSTART_FLASK.md](QUICKSTART_FLASK.md) | Quick start guide with troubleshooting |
| [flask_app/README.md](flask_app/README.md) | Flask app technical documentation |
| [CLAUDE.md](CLAUDE.md) | Development guide for Claude Code AI |

## 🔧 Troubleshooting

### Flask App Issues

**"Model not loaded!" Error**
- Solution: Use the provided startup scripts that activate the virtual environment
- Windows: `start_flask_app.bat`
- Linux/Mac: `./start_flask_app.sh`

**First caption takes long time**
- Expected behavior: First generation takes 10-20 seconds (model loading)
- Subsequent captions are much faster (1-2 seconds)
- The model and ViT weights are cached after first use

**Import errors (torch, flask, etc.)**
- Ensure virtual environment is activated
- Install dependencies: `pip install -r requirements.txt`
- For Flask: `pip install -r flask_app/requirements.txt`

### Training Issues

**Out of Memory (OOM)**
- Reduce batch size (try 16 or 8)
- Reduce number of decoder layers (try 4 instead of 6)
- Use smaller image size (196 instead of 224)
- Close other GPU applications

**Poor caption quality**
- Train for more epochs (50-100 recommended)
- Use beam search with larger beam width (10)
- Check vocabulary size (target: 3000-8000 words)
- Verify dataset quality and caption diversity

**Slow training**
- Use GPU if available (check with `torch.cuda.is_available()`)
- Increase `num_workers` in dataloader
- Use mixed precision training (FP16)

### Model Loading Issues

**Vocabulary unpickling error**
- Fixed in `src/model_manager.py` with automatic namespace injection
- If still issues, the vocab may need to be regenerated from training

**Model checkpoint not found**
- Ensure `models/best_model.pth` exists
- Check that `models/vocab.pth` exists
- Both files are required for inference

## 📊 Performance

### Inference Speed

| Method | Device | Time per Image | Quality |
|--------|--------|----------------|---------|
| Greedy | CPU | ~2s | Good |
| Greedy | GPU (CUDA) | ~50ms | Good |
| Beam Search (width=5) | CPU | ~8s | Better |
| Beam Search (width=5) | GPU (CUDA) | ~200ms | Better |
| Beam Search (width=10) | GPU (CUDA) | ~400ms | Best |

### Training Time

- **Hardware**: NVIDIA GPU (CUDA capable)
- **Time per epoch**: ~30-45 minutes (Flickr8k)
- **Total training**: 3 epochs completed
- **Recommended**: 30-50 epochs for production quality

### Model Size

- **Model checkpoint**: ~1.7 GB (`best_model.pth`)
- **Vocabulary**: ~68 KB (`vocab.pth`)
- **ViT weights**: ~350 MB (downloaded from HuggingFace, cached)
- **Total disk space**: ~2.1 GB

## 🚀 Future Improvements

- [ ] **Attention visualization** - Show which image regions the model focuses on
- [ ] **COCO dataset support** - Train on larger dataset for better quality
- [ ] **Additional metrics** - CIDEr, METEOR, ROUGE-L scores
- [ ] **Mixed precision training** - FP16 for faster training
- [ ] **Model quantization** - INT8 for faster inference
- [ ] **Multi-language support** - Captions in multiple languages
- [ ] **Real-time webcam** - Live caption generation from camera feed
- [ ] **Mobile app** - iOS/Android application
- [ ] **Docker container** - Easy deployment with Docker
- [ ] **REST API** - Standalone API service

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Report bugs** - Open an issue describing the problem
2. **Suggest features** - Open an issue with your idea
3. **Submit pull requests** - Fork, make changes, and submit PR
4. **Improve documentation** - Fix typos, add examples
5. **Share results** - Train on new datasets and share findings

### Development Setup

```bash
# Clone and setup
git clone https://github.com/charankulal/Image-captioning-using-ViT.git
cd Image-captioning-using-ViT

# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
pip install -r flask_app/requirements.txt

# Run tests (if available)
pytest tests/

# Start development server
python flask_app/run.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

You are free to:
- ✅ Use commercially
- ✅ Modify
- ✅ Distribute
- ✅ Private use

## 🙏 Acknowledgments

### Research Papers

- **Vision Transformer**: [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
- **Transformers**: [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- **Image Captioning**: [Show, Attend and Tell (Xu et al., 2015)](https://arxiv.org/abs/1502.03044)

### Datasets

- **Flickr8k**: [Framing Image Description as a Ranking Task (Hodosh et al., 2013)](https://www.jair.org/index.php/jair/article/view/10833)

### Tools & Libraries

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [HuggingFace Transformers](https://huggingface.co/transformers/) - Pre-trained models
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [NumPy](https://numpy.org/) - Numerical computing
- [PIL/Pillow](https://python-pillow.org/) - Image processing

## 📧 Contact

- **Author**: Charan Kulal
- **Email**: charan.kulal.02@gmail.com
- **GitHub**: [@charankulal](https://github.com/charankulal)
- **Project**: [Image-captioning-using-ViT](https://github.com/charankulal/Image-captioning-using-ViT)

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/charankulal/Image-captioning-using-ViT?style=social)
![GitHub forks](https://img.shields.io/github/forks/charankulal/Image-captioning-using-ViT?style=social)
![GitHub issues](https://img.shields.io/github/issues/charankulal/Image-captioning-using-ViT)
![GitHub pull requests](https://img.shields.io/github/issues-pr/charankulal/Image-captioning-using-ViT)

---

**Made with ❤️ using Vision Transformers and Claude Code**

*If you find this project helpful, please consider giving it a ⭐ on GitHub!*
