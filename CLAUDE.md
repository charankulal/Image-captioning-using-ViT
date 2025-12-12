# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an image captioning system that uses Vision Transformer (ViT) as encoder and Transformer decoder to generate natural language descriptions of images. The model is trained on Flickr8k dataset.

## Common Commands

### Flask Web Application
```bash
# Start the web application (Windows)
start_flask_app.bat

# Start the web application (Linux/Mac/Git Bash)
./start_flask_app.sh

# Or manually with venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate.bat
python flask_app/run.py
```

### Python API (Programmatic Usage)
```python
# Quick caption (one-liner)
from src.model_manager import quick_caption
caption = quick_caption('path/to/image.jpg')

# Advanced usage with model manager
from src.model_manager import ImageCaptioningModelManager
manager = ImageCaptioningModelManager(checkpoint_dir='models')
manager.load()
caption = manager.generate_caption('image.jpg', method='beam_search', beam_width=5)
```

### Training and Inference
```bash
# Use Jupyter notebooks for training and testing
# Training: notebooks/Image_Captioning_Training.ipynb
# Inference: notebooks/Image_Captioning_Inference.ipynb
```

## Architecture Overview

### Three-Component Design

1. **Vision Transformer Encoder** (`src/model.py:34-69`)
   - Uses pre-trained `google/vit-base-patch16-224-in21k` from HuggingFace
   - Extracts 197 patch embeddings (196 patches + 1 CLS token) of 768 dimensions
   - Input: 224x224x3 images
   - Output: (batch_size, 197, 768) feature tensors

2. **Transformer Decoder** (`src/model.py:72-128`)
   - 6 decoder layers with 8 attention heads
   - Uses causal masking for autoregressive generation
   - Cross-attention mechanism to attend to image features
   - Self-attention with positional encoding for sequence modeling

3. **Caption Generation** (`src/model.py:172-283`)
   - Greedy decoding: Fast, selects highest probability token at each step
   - Beam search: Better quality, explores multiple hypothesis paths with configurable beam width

### Data Pipeline

- **Dataset** (Defined in `notebooks/Image_Captioning_Training.ipynb`): FlickrDataset handles loading, splitting (72%/8%/20%), and tokenization
- **Vocabulary** (`src/vocabulary.py`): Builds vocabulary with frequency threshold (default: 5 occurrences)
- **Transforms**: Training uses augmentation (flip, rotation, color jitter); validation uses only normalization
- **Special Tokens**: `<PAD>` (0), `<SOS>` (1), `<EOS>` (2), `<UNK>` (3)

### Training Configuration

- **Implemented in**: `notebooks/Image_Captioning_Training.ipynb`
- **Loss**: CrossEntropyLoss with padding token ignored
- **Optimizer**: Adam with betas=(0.9, 0.98), lr=3e-4
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Gradient clipping**: max_norm=1.0 to prevent exploding gradients
- **Checkpoints**: Saved to `models/` directory
  - `best_model.pth`: Model with lowest validation loss
  - `vocab.pth`: Vocabulary object (required for inference)

## Model Manager API

The project provides a reusable API for integrating the model into other applications:

### Quick Usage
```python
from src.model_manager import quick_caption

# One-liner caption generation
caption = quick_caption('path/to/image.jpg')
```

### Advanced Usage
```python
from src.model_manager import ImageCaptioningModelManager

# Initialize and load model
manager = ImageCaptioningModelManager()
manager.load()

# Generate single caption
caption = manager.generate_caption('image.jpg', method='beam_search', beam_width=5)

# Batch processing
captions = manager.generate_captions_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])

# With confidence score
caption, confidence = manager.generate_caption('image.jpg', return_confidence=True)
```

See `FLASK_APP_GUIDE.md` and `QUICKSTART_FLASK.md` for complete usage guides.

## Key Configuration Parameters

Modify these in the training notebook (`notebooks/Image_Captioning_Training.ipynb`):

- `batch_size`: 32 (reduce to 16 if OOM on GPU)
- `learning_rate`: 3e-4
- `num_decoder_layers`: 6 (reduce to 4 if OOM)
- `num_epochs`: 30 (increase to 50-100 for better quality)
- `freq_threshold`: 5 (vocabulary word frequency threshold)
- `pretrained_vit`: True (always use pre-trained ViT for better results)
- `beam_width`: 5 (increase to 10 for better quality, decreases speed)

For Flask app configuration, modify `flask_app/app.py`:
- `SECRET_KEY`: Change for production
- `MAX_CONTENT_LENGTH`: Maximum upload size (default: 16MB)
- `ALLOWED_EXTENSIONS`: Allowed image file types

## Dataset Format

Required structure:
```
data/
├── images/          # Contains .jpg image files
└── captions.csv     # CSV with columns: image,caption
```

The CSV supports multiple captions per image (recommended: 5 per image).

## Vocabulary and Special Tokens

- `<PAD>` (idx=0): Padding token for batch processing
- `<SOS>` (idx=1): Start of sequence
- `<EOS>` (idx=2): End of sequence
- `<UNK>` (idx=3): Unknown words (below frequency threshold)

Vocabulary is built automatically during training and saved to `models/vocab.pth`.

## Checkpoint Management

Checkpoints saved to `./models/`:
- `best_model.pth`: Model with lowest validation loss (~1.7GB)
- `vocab.pth`: Vocabulary object (required for inference)

Checkpoint contents:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state (for resuming training)
- `scheduler_state_dict`: Learning rate scheduler state
- `epoch`: Current epoch number
- `config`: Training configuration used
- `train_losses` / `val_losses`: Loss history
- `best_val_loss`: Best validation loss achieved
- `config`: Training configuration used

## Important Implementation Details

### Causal Masking
The decoder uses causal masking (`src/model.py:124-128`) to prevent attention to future tokens during training. This is essential for autoregressive generation.

### Teacher Forcing
During training (`train.py:68-72`), the model receives ground truth tokens as input and predicts the next token. Input is `captions[:, :-1]`, target is `captions[:, 1:]`.

### Padding Masks
Padding tokens are masked (`train.py:75`) so they don't contribute to loss calculation or attention weights.

### Pre-trained ViT
The encoder uses Google's pre-trained ViT-Base model. On first run, it downloads ~350MB from HuggingFace. The encoder can be frozen for faster training, but is kept trainable by default for better performance.

### Beam Search vs Greedy
- Greedy: ~5x faster, selects max probability token at each step
- Beam search: Better quality, maintains top-k hypothesis paths, more computationally expensive

## Jupyter Notebooks

The project includes Jupyter notebooks for interactive development:

- `Image_Captioning_Training.ipynb`: Interactive training notebook with visualization
- `Image_Captioning_Inference.ipynb`: Interactive inference notebook for testing captions

## Common Issues

### Out of Memory
- Reduce `batch_size` to 16 or 8
- Reduce `num_decoder_layers` to 4
- Use smaller image size (196 instead of 224)

### Vocabulary Too Small/Large
- Adjust `freq_threshold` in config (lower = larger vocab)
- Target vocabulary size: 3000-8000 words

### Poor Caption Quality
- Train longer (50-100 epochs)
- Use beam search with larger beam_width (10)
- Fine-tune ViT encoder (enabled by default)
- Verify dataset quality and caption diversity

### Flask App "Model not loaded!" Error
- Use the startup scripts: `start_flask_app.bat` (Windows) or `./start_flask_app.sh` (Linux/Mac)
- These scripts automatically activate the virtual environment
- Manual fix: Activate venv before running: `source .venv/Scripts/activate`

### Vocabulary Loading Errors
- Fixed in `src/model_manager.py` with automatic namespace injection
- The vocab.pth was saved from notebooks where Vocabulary was in __main__
- The fix injects the class into __main__ before loading

## Code Style and Patterns

- PyTorch-style module definitions with `__init__` and `forward` methods
- Type hints are not extensively used (can be added)
- Configuration managed via notebooks and Flask app config
- Docstrings follow standard Python format with Args/Returns sections
- Progress tracking with tqdm for user feedback

## Files Not to Modify

- `models/vocab.pth`: Auto-generated from training, don't manually edit
- `models/best_model.pth`: Large checkpoint file (~1.7GB)
- `flask_app/database.db`: SQLite database (auto-generated)
- `flask_app/static/uploads/`: User-uploaded images

## Testing Changes

After modifying the model:
1. Test model instantiation:
   ```python
   from src.model import ImageCaptioningModel
   model = ImageCaptioningModel(vocab_size=2535)
   print("Model created successfully!")
   ```

2. Test model loading and inference:
   ```python
   from src.model_manager import ImageCaptioningModelManager
   manager = ImageCaptioningModelManager('models')
   manager.load()
   caption = manager.generate_caption('test_images/1.jpg')
   print(f"Caption: {caption}")
   ```

3. Test Flask app: `python flask_app/run.py` (after activating venv)

## Additional Resources

- `README.md`: Main project documentation with complete overview
- `FLASK_APP_GUIDE.md`: Complete Flask app setup, usage, and deployment guide
- `QUICKSTART_FLASK.md`: Quick start guide for running the web application
- `flask_app/README.md`: Technical documentation for the Flask application
- `notebooks/Image_Captioning_Training.ipynb`: Interactive training notebook
- `notebooks/Image_Captioning_Inference.ipynb`: Interactive inference notebook
