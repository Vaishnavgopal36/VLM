# Vision-Language Model (VLM) integration with CLIP and Qwen

This repository contains the implementation and training pipeline for a custom Vision-Language Model (VLM). The architecture leverages **OpenAI's CLIP** as the visual encoder and **Alibaba's Qwen** as the large language model (LLM) backbone, connected via a learned projection layer.

## Repository Overview

The project is structured into multiple phases of training and inference, implemented primarily in Python and Jupyter Notebooks (comprising 88% of the codebase).

### Key Components

* **`vlm_lib.py`**: The core library containing the model definitions, dataset loaders, and utility functions required for initializing the CLIP and Qwen models, as well as the custom projector module.
* **`main.py` & `run_task.py`**: Python scripts for executing end-to-end training runs, evaluation tasks, and inference tasks from the command line.
* **`clipqwen-phase-1.ipynb`**: Notebook detailing the Phase 1 training pipeline. This phase typically involves feature alignment, training the projection layer (`projector_v3.pt` / `final_projector.pt`) while keeping the LLM and Vision encoders frozen.
* **`clipqwen_phase-2.ipynb`**: Notebook for Phase 2 training, which generally entails fine-tuning the LLM alongside the projector for complex instruction-following and visual question-answering tasks.
* **Projector Checkpoints (`*.pt`)**: Pre-trained PyTorch weights for the projection layers that map the CLIP visual embedding space into the Qwen linguistic embedding space.
  * `final_projector.pt`
  * `projector_v3.pt`
  * `vlm_projector_final (1).pt`

## Architecture

1. **Vision Encoder**: CLIP (Contrastive Language-Image Pre-training) processes input images to extract dense visual patch embeddings.
2. **Projector**: A custom multi-layer perceptron (MLP) or cross-attention module that aligns the dimensionality and semantic space of the CLIP embeddings to match the input expectations of the LLM.
3. **Language Model**: Qwen ingests the projected visual tokens prepended to text tokens to generate autoregressive text responses.

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed along with PyTorch and the required Hugging Face libraries.

```bash
pip install torch torchvision transformers accelerate
```

### Usage

**1. Loading the Model:**
You can import the core logic from `vlm_lib.py` to instantiate the pipeline.

```python
from vlm_lib import VLMModel

# Initialize the model with the trained projector weights
model = VLMModel(projector_path="final_projector.pt")
```

**2. Running Inference:**
For standard task execution, you can use the provided script:

```bash
python run_task.py --image path/to/image.jpg --prompt "Describe the contents of this image in detail."
```

## Language Composition
* Jupyter Notebook: 88%
* Python: 12%

## License
*Please specify the license for this repository.*
