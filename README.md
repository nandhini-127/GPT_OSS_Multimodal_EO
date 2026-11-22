# GPT-OSS Multimodal Earth Observation Project

A multimodal system for Earth Observation data analysis combining vision models, text processing, and fusion techniques.

## Project Structure

```
├── config.py                 # Project configuration
├── main.py                   # Main entry point
├── requirements.txt          # Python dependencies
├── check_project_status.py   # Project status checker
├── check_embeddings.py       # Embedding validation script
├── data/                     # Data directory
│   ├── processed/           # Processed data and embeddings
│   └── train/val/test/      # Raw dataset splits
├── vision/                  # Vision model components
├── multimodal/             # Multimodal fusion components
├── fusion/                 # Fusion model components
├── llm/                    # Language model components
├── inference/              # Inference and evaluation
└── preprocessing/          # Data preprocessing
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Check Project Status
```bash
python check_project_status.py
```

### 3. Run Individual Components

#### Vision Training
```bash
cd vision
python train.py
```

#### Text Embedding Extraction
```bash
python llm/embedding_extractor.py
```

#### Check Embeddings
```bash
python check_embeddings.py
```

## Project Phases

### ✅ Phase 1: Data Preparation
- GeoTIFF loading and processing
- NDVI calculation
- Data transformations

### ✅ Phase 2: Vision Training
- Swin Transformer on EuroSAT dataset
- Feature extraction
- Model checkpoints saved

### ✅ Phase 3: LLM/Text Processing
- Text embedding generation
- Mock embeddings for demonstration
- Embeddings saved to `data/processed/text_embeddings.pt`

### 🔄 Phase 4: Multimodal Fusion (Next)
- Combine vision and text features
- Train fusion model
- Joint classification

### 🔄 Phase 5: Inference (Future)
- Deploy trained models
- Real-time EO analysis
- Web interface

## Current Status

- **Vision Model**: ✅ Trained and checkpoints available
- **Text Embeddings**: ✅ Generated and validated
- **EO Data**: ✅ Processed embeddings available
- **Dependencies**: ✅ All packages installed

## Usage Examples

### Check All Components
```bash
python check_project_status.py
```

### Validate Embeddings
```bash
python check_embeddings.py
```

### Run Vision Model Test
```python
from vision.model import build_model
model = build_model()
print("Vision model loaded successfully!")
```

## Data Files

- `data/processed/eo_embeddings.npy` - EO data embeddings
- `data/processed/text_embeddings.pt` - Text embeddings
- `checkpoints/vision_best.pth` - Trained vision model


## Requirements

- Python 3.9+
- PyTorch with CUDA support
- Transformers library
- NumPy, Pillow, OpenAI API
