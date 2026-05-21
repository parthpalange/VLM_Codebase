# VLM Pipeline — Vision-Language Model End-to-End Toolkit

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A modular, production-ready Python toolkit for training, evaluating, and deploying **Vision-Language Models (VLMs)**. Supports 35+ model checkpoints across all major VLM families, 6 task types, LoRA/QLoRA fine-tuning, and 4 inference modes — all from a single unified CLI.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Supported Models](#supported-models)
- [Supported Tasks](#supported-tasks)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Full Pipeline Walkthrough](#full-pipeline-walkthrough)
  - [1. Configuration (`config.py`)](#1-configuration-configpy)
  - [2. Preprocessing (`preprocess.py`)](#2-preprocessing-preprocesspy)
  - [3. Training (`train.py`)](#3-training-trainpy)
  - [4. Evaluation (`evaluate.py`)](#4-evaluation-evaluatepy)
  - [5. Inference (`predict.py`)](#5-inference-predictpy)
  - [6. Unified CLI (`main.py`)](#6-unified-cli-mainpy)
- [CLI Reference](#cli-reference)
- [Configuration Reference](#configuration-reference)
- [Dataset Formats](#dataset-formats)
- [Metrics Reference](#metrics-reference)
- [Advanced Usage](#advanced-usage)
- [Project Structure](#project-structure)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py  (CLI Entry Point)               │
│   train │ evaluate │ predict │ preprocess │ list-models          │
└────┬────────┬───────────┬───────────┬──────────────────────────┘
     │        │           │           │
     ▼        ▼           ▼           ▼
 train.py  evaluate.py predict.py preprocess.py
     │        │           │           │
     └────────┴───────────┴───────────┘
                          │
                          ▼
                      config.py
               (MODEL_REGISTRY · VLMConfig
                TrainingConfig · DataConfig
                PathConfig · Validators)
```

The pipeline is broken into **five focused modules**, each independently usable, all orchestrated by `main.py`:

| Module | Role |
|---|---|
| `config.py` | Single source of truth: model registry, typed config dataclasses, helpers |
| `preprocess.py` | Data loading (JSONL/CSV/HF/COCO), `VLMDataset`, `DataLoader` factory |
| `train.py` | Full/LoRA/QLoRA training loop, `Trainer` class, TensorBoard + W&B |
| `evaluate.py` | Task-specific metrics (BLEU, CIDEr, VQA-Accuracy, Recall@k, BERTScore…) |
| `predict.py` | Single / batch / interactive / zero-shot classify inference |
| `main.py` | Argparse CLI wrapping all of the above into one entry point |

---

## Supported Models

35+ model checkpoints across two families:

### Generative Models (autoregressive / seq2seq)

| Family | Models |
|---|---|
| **BLIP-2** | `blip2-opt-2.7b`, `blip2-opt-6.7b`, `blip2-flan-t5-xl`, `blip2-flan-t5-xxl` |
| **InstructBLIP** | `instructblip-vicuna-7b`, `instructblip-vicuna-13b`, `instructblip-flan-t5-xl`, `instructblip-flan-t5-xxl` |
| **LLaVA 1.5** | `llava-1.5-7b`, `llava-1.5-13b` |
| **LLaVA-NeXT (1.6)** | `llava-next-mistral-7b`, `llava-next-vicuna-7b`, `llava-next-34b` |
| **PaliGemma** | `paligemma-3b-pt-224`, `paligemma-3b-mix-224`, `paligemma-3b-mix-448` |
| **IDEFICS2** | `idefics2-8b` |
| **Florence-2** | `florence-2-base`, `florence-2-large` |
| **GIT** | `git-base`, `git-large`, `git-base-coco` |
| **BLIP** | `blip-base`, `blip-large`, `blip-vqa-base` |
| **Qwen2-VL** | `qwen2-vl-7b`, `qwen2-vl-72b` |
| **Phi-3.5** | `phi-3.5-vision` |
| **InternVL2** | `internvl2-8b` |
| **CogVLM2** | `cogvlm2` |
| **Moondream2** | `moondream2` |
| **DeepSeek-VL** | `deepseek-vl-7b` |
| **Emu3** | `emu3` |

### Contrastive Models (CLIP-style, no text generation)

| Family | Models |
|---|---|
| **CLIP** | `clip-vit-base-patch32`, `clip-vit-large-patch14`, `clip-vit-large-patch14-336` |
| **ViLT** | `vilt-vqa` |
| **FLAVA** | `flava` |

---

## Supported Tasks

| Task | Description | Key Metrics |
|---|---|---|
| `captioning` | Generate descriptive captions for images | BLEU-1/4, METEOR, ROUGE-L, CIDEr |
| `vqa` | Answer natural-language questions about images | VQA Accuracy, Exact Match, Token-F1 |
| `retrieval` | Image↔text embedding retrieval (CLIP-style) | Recall@1/5/10, mAP, MRR |
| `classification` | Zero-shot or fine-tuned image classification | Accuracy, Macro-F1, Weighted-F1 |
| `grounding` | Locate objects described in text | IoU, Pointing Accuracy, Recall@IoU |
| `chat` | Multi-turn visual dialogue | ROUGE-L, BERTScore, METEOR |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/vlm-pipeline.git
cd vlm-pipeline

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.37 datasets peft bitsandbytes accelerate
pip install pillow tqdm numpy

# Optional — for full metric support
pip install nltk rouge-score bert-score tabulate

# Optional — for experiment tracking
pip install wandb tensorboard
```

> **GPU requirements:** Models range from 4 GB VRAM (`clip-vit-base-patch32`) to 160 GB (`qwen2-vl-72b`). Use `--use_qlora` for 4-bit quantisation to reduce memory by ~4×.

---

## Quick Start

```bash
# List all available models
python main.py list-models

# List models that support VQA, formatted as JSON
python main.py list-models --task vqa --format json

# Run a single captioning prediction (no training needed)
python main.py predict \
    --model blip2-opt-2.7b \
    --mode single \
    --image ./photo.jpg \
    --task captioning

# Fine-tune with LoRA on a custom JSONL dataset
python main.py train \
    --model llava-1.5-7b \
    --task vqa \
    --dataset_type jsonl \
    --data_dir ./data \
    --train_file train.jsonl \
    --val_file val.jsonl \
    --use_lora --bf16 \
    --epochs 3

# Evaluate a saved checkpoint
python main.py evaluate \
    --model llava-1.5-7b \
    --checkpoint ./outputs/checkpoints/best \
    --task vqa \
    --data_dir ./data --test_file test.jsonl
```

---

## Full Pipeline Walkthrough

### 1. Configuration (`config.py`)

The central configuration module. Import directly or use the factory function:

```python
from config import get_config, list_models, validate_config, MODEL_REGISTRY

# Build a fully-populated VLMConfig
cfg = get_config("llava-1.5-7b", "vqa")

# Override specific fields using dotted-path syntax
cfg = get_config(
    "blip2-opt-2.7b",
    "captioning",
    **{"training.learning_rate": 5e-5,
       "training.use_lora": True,
       "data.image_size": 336,
       "paths.output_dir": "./my_runs"}
)

# Validate the config — returns a list of warnings
warnings = validate_config(cfg)
for w in warnings:
    print(w)

# List all models, or filter by task
all_models   = list_models()
vqa_models   = list_models(task="vqa")
clip_models  = list_models(task="retrieval")

# Inspect a model's full metadata
from config import get_model_info
info = get_model_info("clip-vit-large-patch14")
# info["is_generative"], info["min_gpu_gb"], info["supported_metrics"], …
```

#### Key dataclasses

| Dataclass | Purpose | Key Fields |
|---|---|---|
| `VLMConfig` | Top-level container | `model_name`, `task`, `model`, `training`, `data`, `paths` |
| `ModelConfig` | Per-model metadata | `hub_id`, `model_class`, `lora_target_modules`, `min_gpu_gb` |
| `TrainingConfig` | Hyper-parameters | `learning_rate`, `num_epochs`, `batch_size`, `use_lora`, `use_qlora`, `bf16` |
| `DataConfig` | Dataset settings | `dataset_type`, `train_file`, `hf_dataset_name`, `image_size`, `max_length` |
| `PathConfig` | File-system paths | `output_dir`, `checkpoint_dir`, `log_dir`, `cache_dir` |

---

### 2. Preprocessing (`preprocess.py`)

Loads and transforms data, building ready-to-train PyTorch `DataLoader`s.

```python
from preprocess import get_processor, get_dataloaders, preprocess_and_cache

# Load the model's processor / tokenizer
processor = get_processor(cfg.model.hub_id, cfg)

# Build train + val DataLoaders
train_loader, val_loader = get_dataloaders(cfg, processor)

# Pre-tokenise and cache to disk (fast repeated experiments)
stats = preprocess_and_cache(config=cfg, processor=processor)
```

#### Dataset formats

| Format | `dataset_type` | Details |
|---|---|---|
| JSONL | `jsonl` | One JSON object per line; keys: `image_path`, `question`, `answer`, `caption` |
| CSV | `csv` | Standard comma-separated; same column names |
| HuggingFace Hub | `hf` | Set `hf_dataset_name`; columns auto-remapped |
| COCO | `coco` | Set `coco_annotation_dir` + `coco_image_dir`; reads `captions_{split}2017.json` |

#### `VLMDataset` internals

- Loads PIL images from path, URL, or directly from HF datasets
- Applies model-appropriate image transforms (CLIP stats vs ImageNet stats)
- Builds prompt text and label text per task type
- Supports processor-native encoding with manual fallback
- Masks prompt tokens in labels with `IGNORE_INDEX = -100` for causal LM training
- Handles corrupt / missing images gracefully (grey placeholder)

---

### 3. Training (`train.py`)

Full fine-tuning or parameter-efficient adaptation via LoRA / QLoRA.

```python
from train import get_model, Trainer, freeze_backbone, apply_lora

# Load model (handles quantisation, LoRA, backbone freezing automatically)
model = get_model(cfg)

# Optionally freeze vision encoder + (optionally) LLM backbone
freeze_backbone(model, cfg)

# Wrap with LoRA adapters
model = apply_lora(model, cfg)

# Create and run the Trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=cfg,
    processor=processor,
)
metrics = trainer.train()
# returns {"train_losses": [...], "val_losses": [...], "best_val_loss": float}
```

#### Training features

| Feature | Details |
|---|---|
| **LoRA** | Configurable rank (`lora_r`), alpha, dropout, target modules per model |
| **QLoRA** | 4-bit NF4 quantisation via bitsandbytes + LoRA adapters |
| **8-bit** | 8-bit quantisation only (no LoRA) |
| **Mixed precision** | `bf16` (Ampere+) or `fp16` with automatic `GradScaler` |
| **Gradient accumulation** | `grad_accum_steps` for effective larger batch sizes |
| **Gradient clipping** | `max_grad_norm` (default 1.0) |
| **LR schedule** | Cosine decay with linear warm-up (`warmup_ratio`) |
| **Checkpointing** | Best-model + every-N-steps; configurable `save_total_limit` |
| **Resume** | `--resume <checkpoint_dir>` restores model, optimizer, and scheduler state |
| **TensorBoard** | Auto-written to `<output_dir>/tensorboard/` |
| **Weights & Biases** | Enable with `--use_wandb --wandb_project <name>` |
| **Dry run** | `--dry_run` runs exactly 1 training step for sanity checking |

---

### 4. Evaluation (`evaluate.py`)

Task-specific metrics with no mandatory heavy dependencies (all optional).

```python
from evaluate import Evaluator, compare_checkpoints

# Single-checkpoint evaluation
evaluator = Evaluator(model=model, processor=processor, config=cfg)
report = evaluator.evaluate(dataloader=test_loader, task="captioning")
# report: {"bleu1": ..., "bleu4": ..., "meteor": ..., "rouge_l": ..., "cider": ...}

# Multi-checkpoint comparison
comparison = compare_checkpoints(
    config=cfg,
    checkpoints=["./ckpt/epoch-1", "./ckpt/epoch-2", "./ckpt/epoch-3"],
    dataloader=test_loader,
    processor=processor,
)
```

#### Standalone metric functions

```python
from evaluate import (
    compute_bleu,           # BLEU-1 through BLEU-4
    compute_rouge,          # ROUGE-1, ROUGE-2, ROUGE-L
    compute_meteor,         # METEOR
    compute_cider,          # CIDEr-D (self-contained TF-IDF, no pycocoevalcap)
    compute_vqa_accuracy,   # VQA v2-style soft accuracy
    compute_exact_match,    # Fraction of exact matches (normalised)
    compute_f1_token,       # SQuAD-style token-level F1
    compute_clip_retrieval_metrics,  # Recall@1/5/10, mAP, MRR (i2t + t2i)
    compute_bertscore,      # BERTScore P/R/F1
    normalize_answer,       # VQA-style text normalisation
)
```

#### Metrics by task

| Task | Metrics |
|---|---|
| `captioning` | BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr-D |
| `vqa` | VQA Accuracy (soft), Exact Match, Token-F1 |
| `retrieval` | Recall@1/5/10, mAP, MRR (image→text + text→image + mean) |
| `classification` | Accuracy, Macro-F1, Weighted-F1, per-class P/R/F1/Support |
| `chat` | ROUGE-L, BERTScore F1, METEOR |

---

### 5. Inference (`predict.py`)

Four inference modes for generative and contrastive models.

```python
from predict import VLMPredictor

predictor = VLMPredictor("llava-hf/llava-1.5-7b-hf", device="auto")

# ── Single prediction ──────────────────────────────────────────────
caption = predictor.predict_single("photo.jpg", task="captioning")
answer  = predictor.predict_single("photo.jpg", prompt="What color is the car?", task="vqa")

# ── Batch prediction ───────────────────────────────────────────────
items = [
    {"image_path": "img1.jpg", "prompt": "Describe the scene."},
    {"image_path": "img2.jpg", "prompt": "What objects are visible?"},
]
predictions = predictor.predict_batch(items, task="captioning", batch_size=8)

# ── Zero-shot classification (CLIP) ───────────────────────────────
clip = VLMPredictor("openai/clip-vit-large-patch14")
scores = clip.zero_shot_classify("cat.jpg", class_labels=["cat", "dog", "bird"])
# {"cat": 0.92, "dog": 0.05, "bird": 0.03}

# ── Similarity score (CLIP) ───────────────────────────────────────
sim = clip.compute_similarity("cat.jpg", "a photo of a cat")

# ── Interactive chat ───────────────────────────────────────────────
predictor.predict_interactive(image_path="photo.jpg")
# Enters REPL; commands: 'quit', 'new image <path>', 'clear', 'history'
```

#### Generation parameters

Pass any `GenerationConfig` field as a keyword argument:

```python
predictor.predict_single(
    "photo.jpg",
    task="captioning",
    max_new_tokens=512,
    num_beams=4,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.2,
)
```

---

### 6. Unified CLI (`main.py`)

All pipeline stages are accessible from a single entry point:

```
python main.py <subcommand> [options]
```

Subcommands: `train`, `evaluate`, `predict`, `preprocess`, `list-models`

---

## CLI Reference

### `train`

```bash
python main.py train \
    --model <MODEL_NAME>          # Required: registry key (e.g. blip2-opt-2.7b)
    --task <TASK>                 # Required: captioning|vqa|retrieval|classification|chat
    --dataset_type jsonl          # jsonl|csv|hf|coco
    --data_dir ./data             # Root directory for images / annotation files
    --train_file train.jsonl      # Training split file
    --val_file val.jsonl          # Validation split file
    --hf_dataset <REPO/NAME>      # HuggingFace dataset (when dataset_type=hf)
    --output_dir ./outputs        # Where to save model weights and logs
    --checkpoint_dir ./checkpoints
    --epochs 3
    --batch_size 8
    --lr 1e-4
    --grad_accum 4                # Effective batch = batch_size × grad_accum
    --use_lora                    # Enable LoRA adapters
    --use_qlora                   # Enable QLoRA (4-bit + LoRA)
    --lora_r 16
    --lora_alpha 32
    --lora_dropout 0.05
    --bf16                        # bfloat16 mixed precision
    --fp16                        # float16 mixed precision
    --freeze_backbone             # Freeze vision encoder weights
    --resume <CHECKPOINT_DIR>     # Resume from checkpoint
    --use_wandb                   # Enable W&B tracking
    --wandb_project my_runs
    --max_length 128              # Max token sequence length
    --image_size 224              # Input image resolution
    --num_workers 4
    --dry_run                     # Run 1 step only (sanity check)
    --log_level INFO
    --log_file train.log
```

### `evaluate`

```bash
python main.py evaluate \
    --model <MODEL_NAME>
    --checkpoint <PATH>           # Required: path to checkpoint directory
    --task <TASK>
    --dataset_type jsonl
    --data_dir ./data
    --test_file test.jsonl
    --output_dir ./eval_outputs
    --batch_size 16
    --split test                  # test|val
    --compare "ckpt1,ckpt2,ckpt3" # Compare multiple checkpoints
    --image_size 224
    --max_length 128
    --num_workers 4
```

### `predict`

```bash
python main.py predict \
    --model <MODEL_NAME>
    --checkpoint <PATH>           # Optional: use base model if omitted
    --mode single                 # single|batch|interactive|classify
    --image <PATH_OR_URL>         # For single/classify modes
    --prompt "Describe this."     # Optional text prompt
    --task captioning
    --input_file queries.jsonl    # For batch mode
    --output_file predictions.jsonl
    --class_labels "cat,dog,bird" # For classify mode
    --max_new_tokens 256
    --num_beams 4
    --temperature 1.0
    --top_p 0.9
    --device auto                 # cpu|cuda|cuda:1|auto
```

### `preprocess`

```bash
python main.py preprocess \
    --model <MODEL_NAME>
    --dataset_type jsonl
    --data_dir ./data
    --output_dir ./data/cached
    --image_size 224
    --max_length 128
    --task captioning
```

### `list-models`

```bash
python main.py list-models                    # All models as ASCII table
python main.py list-models --task vqa         # Filter by task
python main.py list-models --format json      # JSON output
```

---

## Configuration Reference

### `VLMConfig` (top-level)

```python
from config import get_config
cfg = get_config("llava-1.5-7b", "vqa",
    **{"training.learning_rate": 1e-4,
       "training.num_epochs": 5,
       "training.batch_size": 8,
       "training.gradient_accumulation_steps": 4,
       "training.use_lora": True,
       "training.lora_r": 16,
       "training.lora_alpha": 32,
       "training.bf16": True,
       "data.dataset_type": "jsonl",
       "data.train_file": "./train.jsonl",
       "data.image_size": 336,
       "paths.output_dir": "./outputs/llava_vqa"})
```

### Override syntax

| Syntax | Effect |
|---|---|
| `"training.learning_rate"` | Sets `cfg.training.learning_rate` |
| `"data.image_size"` | Sets `cfg.data.image_size` |
| `"paths.output_dir"` | Sets `cfg.paths.output_dir` |
| `"batch_size"` (flat) | Searches training → data → paths for first match |

### Config validation

`validate_config(cfg)` checks:
- Model exists in `MODEL_REGISTRY`
- Task is in `SUPPORTED_TASKS`
- Model supports the requested task
- `bf16` and `fp16` not simultaneously enabled
- QLoRA requires `load_in_4bit=True`
- `load_in_4bit` and `load_in_8bit` are mutually exclusive
- `lora_alpha >= lora_r`
- Dataset source is valid
- GPU VRAM is sufficient (warns if not)
- Output directory is writable

---

## Dataset Formats

### JSONL (recommended)

Each line is a JSON object:

```jsonl
{"image_path": "/data/images/img001.jpg", "caption": "A dog playing fetch on a beach."}
{"image_path": "/data/images/img002.jpg", "question": "What color is the car?", "answer": "red"}
{"image_path": "/data/images/img003.jpg", "question": "Describe the scene.", "answer": "A busy street...", "conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### CSV

```csv
image_path,question,answer,caption
/data/img001.jpg,What breed is this?,labrador,A dog playing fetch
```

### HuggingFace Hub

```bash
python main.py train --dataset_type hf --hf_dataset HuggingFaceM4/VQAv2 --task vqa ...
```

Column names are auto-remapped to `image_path`, `question`, `answer`, `caption` using `DataConfig` column settings.

### COCO

Set `coco_annotation_dir` (contains `captions_train2017.json` etc.) and `coco_image_dir` (root of COCO images):

```bash
python main.py preprocess \
    --dataset_type coco \
    --data_dir ./coco/annotations \
    --output_dir ./coco/cached
```

---

## Metrics Reference

| Metric | Task | Range | Implementation |
|---|---|---|---|
| BLEU-1/2/3/4 | captioning, chat | 0–1 | NLTK corpus_bleu |
| METEOR | captioning, chat | 0–1 | NLTK |
| ROUGE-1/2/L | captioning, chat | 0–1 | rouge-score |
| CIDEr-D | captioning | 0–10 | Self-contained TF-IDF (no external dep) |
| VQA Accuracy | vqa | 0–1 | VQA v2 soft scoring (min(1, matches/3)) |
| Exact Match | vqa | 0–1 | After VQA-style normalisation |
| Token-F1 | vqa | 0–1 | SQuAD-style bag-of-tokens overlap |
| Recall@k | retrieval | 0–1 | Cosine similarity in embed space |
| mAP | retrieval | 0–1 | Average precision (single-positive) |
| MRR | retrieval | 0–1 | Mean reciprocal rank |
| Accuracy | classification | 0–1 | Pure numpy, no sklearn required |
| Macro-F1 | classification | 0–1 | Unweighted average per-class F1 |
| Weighted-F1 | classification | 0–1 | Support-weighted average F1 |
| BERTScore F1 | chat | 0–1 | bert-score (optional) |

---

## Advanced Usage

### QLoRA Fine-tuning (memory-efficient)

```bash
python main.py train \
    --model llava-1.5-13b \
    --task vqa \
    --dataset_type hf --hf_dataset HuggingFaceM4/VQAv2 \
    --use_qlora \           # 4-bit NF4 quantisation + LoRA
    --lora_r 64 --lora_alpha 128 \
    --bf16 --grad_accum 8 \
    --epochs 1 --batch_size 2
```

### Multi-checkpoint comparison

```bash
python main.py evaluate \
    --model blip2-opt-2.7b \
    --checkpoint ./ckpt/epoch-3 \
    --task captioning \
    --data_dir ./data/coco --test_file test.jsonl \
    --compare "./ckpt/epoch-1,./ckpt/epoch-2,./ckpt/epoch-3"
```

### Streaming interactive chat

```bash
python main.py predict \
    --model llava-1.5-13b \
    --mode interactive \
    --task chat
# In the REPL:
# You: <your question here>
# Commands: 'new image <path>', 'clear', 'history', 'exit'
```

### Programmatic training loop

```python
from config import get_config, validate_config
from preprocess import get_processor, get_dataloaders
from train import get_model, Trainer

cfg = get_config("blip2-opt-2.7b", "captioning",
    **{"training.use_lora": True, "training.bf16": True,
       "data.train_file": "train.jsonl", "data.val_file": "val.jsonl"})

warnings = validate_config(cfg)
processor = get_processor(cfg.model.hub_id, cfg)
train_dl, val_dl = get_dataloaders(cfg, processor)
model = get_model(cfg)

trainer = Trainer(model=model, train_loader=train_dl,
                  val_loader=val_dl, config=cfg, processor=processor)
results = trainer.train()
print(f"Best val_loss: {results['best_val_loss']:.4f}")
```

### Custom model lookup

```python
from config import MODEL_REGISTRY, TASK_METRICS, PROMPT_TEMPLATES

# Check if a model supports a task
print(MODEL_REGISTRY["llava-1.5-7b"]["tasks"])   # ["captioning", "vqa", "chat", "grounding"]

# Find the prompt template for a model family
print(PROMPT_TEMPLATES["llava-1.5"])              # "USER: <image>\n{question}\nASSISTANT:"

# Get all metrics for a task
print(TASK_METRICS["vqa"])                        # ["vqa_accuracy", "exact_match", "f1"]
```

---

## Project Structure

```
VLM_Codebase/
├── config.py        # MODEL_REGISTRY, VLMConfig dataclasses, helpers
├── preprocess.py    # DatasetLoader, VLMDataset, get_dataloaders, get_processor
├── train.py         # get_model, apply_lora, freeze_backbone, Trainer
├── evaluate.py      # Metric functions, Evaluator, compare_checkpoints
├── predict.py       # VLMPredictor (single/batch/interactive/classify)
├── main.py          # Unified argparse CLI entry point
└── README.md        # This file
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ using [HuggingFace Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), and [PyTorch](https://pytorch.org/).*
