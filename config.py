"""
config.py
=========
Central configuration module for the VLM (Vision-Language Model) end-to-end pipeline.

Responsibilities
----------------
* MODEL_REGISTRY  – metadata for every supported model hub checkpoint.
* Dataclasses     – typed, validated configuration containers for models,
                    training, data, paths, and the top-level VLMConfig.
* Constants       – task lists, metric mappings, prompt templates, image sizes.
* Helper functions – factory, query, and validation utilities.

All public symbols are importable directly from this module:

    from config import get_config, MODEL_REGISTRY, VLMConfig

Author : VLM Pipeline
Created: 2026-05-20
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field, fields, asdict
from typing import Any, Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported tasks (single source of truth)
# ---------------------------------------------------------------------------

SUPPORTED_TASKS: List[str] = [
    "captioning",
    "vqa",
    "retrieval",
    "classification",
    "grounding",
    "chat",
]

# ---------------------------------------------------------------------------
# Model family groupings
# ---------------------------------------------------------------------------

#: Models that produce free-form text outputs (autoregressive / seq2seq).
GENERATIVE_MODELS: List[str] = [
    "blip2-opt-2.7b",
    "blip2-opt-6.7b",
    "blip2-flan-t5-xl",
    "blip2-flan-t5-xxl",
    "instructblip-vicuna-7b",
    "instructblip-vicuna-13b",
    "instructblip-flan-t5-xl",
    "instructblip-flan-t5-xxl",
    "llava-1.5-7b",
    "llava-1.5-13b",
    "llava-next-mistral-7b",
    "llava-next-vicuna-7b",
    "llava-next-34b",
    "paligemma-3b-pt-224",
    "paligemma-3b-mix-224",
    "paligemma-3b-mix-448",
    "idefics2-8b",
    "florence-2-base",
    "florence-2-large",
    "git-base",
    "git-large",
    "git-base-coco",
    "blip-base",
    "blip-large",
    "blip-vqa-base",
    "qwen2-vl-7b",
    "qwen2-vl-72b",
    "phi-3.5-vision",
    "internvl2-8b",
    "cogvlm2",
    "moondream2",
    "deepseek-vl-7b",
    "emu3",
]

#: Models trained with contrastive objectives (CLIP-style); no free-form text.
CONTRASTIVE_MODELS: List[str] = [
    "clip-vit-base-patch32",
    "clip-vit-large-patch14",
    "clip-vit-large-patch14-336",
    "flava",
    "vilt-vqa",
]

# ---------------------------------------------------------------------------
# Task → evaluation metrics
# ---------------------------------------------------------------------------

TASK_METRICS: Dict[str, List[str]] = {
    "captioning": ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "meteor", "rouge_l", "cider", "spice"],
    "vqa": ["vqa_accuracy", "exact_match", "f1"],
    "retrieval": ["recall_at_1", "recall_at_5", "recall_at_10", "map", "mrr"],
    "classification": ["accuracy", "f1_macro", "f1_micro", "precision", "recall", "auc_roc"],
    "grounding": ["iou", "pointing_accuracy", "recall_at_iou_50", "recall_at_iou_75"],
    "chat": ["bleu_4", "rouge_l", "meteor", "bertscore", "gpt4_score"],
}

# ---------------------------------------------------------------------------
# Per-model-family prompt templates
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES: Dict[str, str] = {
    # BLIP-2 (OPT backend)
    "blip2-opt": "Question: {question} Answer:",
    # BLIP-2 (Flan-T5 backend)
    "blip2-flan-t5": "Question: {question} Short answer:",
    # InstructBLIP
    "instructblip": "{question}",
    # LLaVA 1.5
    "llava-1.5": "USER: <image>\n{question}\nASSISTANT:",
    # LLaVA-NeXT (1.6)
    "llava-next": "[INST] <image>\n{question} [/INST]",
    # PaliGemma
    "paligemma": "{question}\n",
    # IDEFICS2
    "idefics2": "User:<image>{question}<end_of_utterance>\nAssistant:",
    # Florence-2
    "florence-2": "<{task_token}>{question}",
    # GIT
    "git": "{question}",
    # Original BLIP
    "blip": "question: {question} answer:",
    # ViLT
    "vilt": "{question}",
    # FLAVA
    "flava": "{question}",
    # Qwen2-VL
    "qwen2-vl": (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "{question}<|im_end|>\n<|im_start|>assistant\n"
    ),
    # Phi-3.5-Vision
    "phi-3.5-vision": "<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n",
    # InternVL2
    "internvl2": "<image>\n{question}",
    # CogVLM2
    "cogvlm2": "USER: {question} ASSISTANT:",
    # Moondream2
    "moondream2": "{question}",
    # DeepSeek-VL
    "deepseek-vl": "User: <image_placeholder>{question}\nAssistant:",
    # Emu3
    "emu3": (
        "[|User|]: <image>{question}\n"
        "[|Assistant|]:"
    ),
    # CLIP (retrieval / classification – no question template needed)
    "clip": "{query}",
}

# ---------------------------------------------------------------------------
# Default image sizes per model family (width == height in pixels)
# ---------------------------------------------------------------------------

DEFAULT_IMAGE_SIZES: Dict[str, int] = {
    "blip2-opt": 224,
    "blip2-flan-t5": 224,
    "instructblip": 224,
    "llava-1.5": 336,
    "llava-next": 336,
    "clip-base": 224,
    "clip-large": 224,
    "clip-large-336": 336,
    "paligemma-224": 224,
    "paligemma-448": 448,
    "idefics2": 980,
    "florence-2": 224,
    "git": 224,
    "blip": 384,
    "vilt": 384,
    "flava": 224,
    "qwen2-vl": 448,
    "phi-3.5-vision": 336,
    "internvl2": 448,
    "cogvlm2": 490,
    "moondream2": 378,
    "deepseek-vl": 1024,
    "emu3": 512,
}

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_RAW_REGISTRY: List[Dict[str, Any]] = [
    # ── BLIP-2 ──────────────────────────────────────────────────────────────
    {
        "name": "blip2-opt-2.7b",
        "hub_id": "Salesforce/blip2-opt-2.7b",
        "model_class": "Blip2ForConditionalGeneration",
        "processor_class": "Blip2Processor",
        "tasks": ["captioning", "vqa", "chat"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        "default_dtype": "float16",
        "min_gpu_gb": 8,
    },
    {
        "name": "blip2-opt-6.7b",
        "hub_id": "Salesforce/blip2-opt-6.7b",
        "model_class": "Blip2ForConditionalGeneration",
        "processor_class": "Blip2Processor",
        "tasks": ["captioning", "vqa", "chat"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        "default_dtype": "float16",
        "min_gpu_gb": 16,
    },
    {
        "name": "blip2-flan-t5-xl",
        "hub_id": "Salesforce/blip2-flan-t5-xl",
        "model_class": "Blip2ForConditionalGeneration",
        "processor_class": "Blip2Processor",
        "tasks": ["captioning", "vqa", "chat"],
        "lora_target_modules": ["q", "v", "k", "o", "wi_0", "wi_1", "wo"],
        "default_dtype": "float16",
        "min_gpu_gb": 16,
    },
    {
        "name": "blip2-flan-t5-xxl",
        "hub_id": "Salesforce/blip2-flan-t5-xxl",
        "model_class": "Blip2ForConditionalGeneration",
        "processor_class": "Blip2Processor",
        "tasks": ["captioning", "vqa", "chat"],
        "lora_target_modules": ["q", "v", "k", "o", "wi_0", "wi_1", "wo"],
        "default_dtype": "float16",
        "min_gpu_gb": 40,
    },
    # ── InstructBLIP ─────────────────────────────────────────────────────────
    {
        "name": "instructblip-vicuna-7b",
        "hub_id": "Salesforce/instructblip-vicuna-7b",
        "model_class": "InstructBlipForConditionalGeneration",
        "processor_class": "InstructBlipProcessor",
        "tasks": ["captioning", "vqa", "chat"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],
        "default_dtype": "float16",
        "min_gpu_gb": 16,
    },
    {
        "name": "instructblip-vicuna-13b",
        "hub_id": "Salesforce/instructblip-vicuna-13b",
        "model_class": "InstructBlipForConditionalGeneration",
        "processor_class": "InstructBlipProcessor",
        "tasks": ["captioning", "vqa", "chat"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],
        "default_dtype": "float16",
        "min_gpu_gb": 40,
    },
    {
        "name": "instructblip-flan-t5-xl",
        "hub_id": "Salesforce/instructblip-flan-t5-xl",
        "model_class": "InstructBlipForConditionalGeneration",
        "processor_class": "InstructBlipProcessor",
        "tasks": ["captioning", "vqa", "chat"],
        "lora_target_modules": ["q", "v", "k", "o", "wi_0", "wi_1", "wo"],
        "default_dtype": "float16",
        "min_gpu_gb": 16,
    },
    {
        "name": "instructblip-flan-t5-xxl",
        "hub_id": "Salesforce/instructblip-flan-t5-xxl",
        "model_class": "InstructBlipForConditionalGeneration",
        "processor_class": "InstructBlipProcessor",
        "tasks": ["captioning", "vqa", "chat"],
        "lora_target_modules": ["q", "v", "k", "o", "wi_0", "wi_1", "wo"],
        "default_dtype": "float16",
        "min_gpu_gb": 40,
    },
    # ── LLaVA 1.5 ────────────────────────────────────────────────────────────
    {
        "name": "llava-1.5-7b",
        "hub_id": "llava-hf/llava-1.5-7b-hf",
        "model_class": "LlavaForConditionalGeneration",
        "processor_class": "LlavaProcessor",
        "tasks": ["captioning", "vqa", "chat", "grounding"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "default_dtype": "float16",
        "min_gpu_gb": 16,
    },
    {
        "name": "llava-1.5-13b",
        "hub_id": "llava-hf/llava-1.5-13b-hf",
        "model_class": "LlavaForConditionalGeneration",
        "processor_class": "LlavaProcessor",
        "tasks": ["captioning", "vqa", "chat", "grounding"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "default_dtype": "float16",
        "min_gpu_gb": 40,
    },
    # ── LLaVA-NeXT (1.6) ─────────────────────────────────────────────────────
    {
        "name": "llava-next-mistral-7b",
        "hub_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "model_class": "LlavaNextForConditionalGeneration",
        "processor_class": "LlavaNextProcessor",
        "tasks": ["captioning", "vqa", "chat", "grounding"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "default_dtype": "float16",
        "min_gpu_gb": 16,
    },
    {
        "name": "llava-next-vicuna-7b",
        "hub_id": "llava-hf/llava-v1.6-vicuna-7b-hf",
        "model_class": "LlavaNextForConditionalGeneration",
        "processor_class": "LlavaNextProcessor",
        "tasks": ["captioning", "vqa", "chat", "grounding"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "default_dtype": "float16",
        "min_gpu_gb": 16,
    },
    {
        "name": "llava-next-34b",
        "hub_id": "llava-hf/llava-v1.6-34b-hf",
        "model_class": "LlavaNextForConditionalGeneration",
        "processor_class": "LlavaNextProcessor",
        "tasks": ["captioning", "vqa", "chat", "grounding"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "default_dtype": "bfloat16",
        "min_gpu_gb": 80,
    },
    # ── CLIP ─────────────────────────────────────────────────────────────────
    {
        "name": "clip-vit-base-patch32",
        "hub_id": "openai/clip-vit-base-patch32",
        "model_class": "CLIPModel",
        "processor_class": "CLIPProcessor",
        "tasks": ["retrieval", "classification"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],
        "default_dtype": "float32",
        "min_gpu_gb": 4,
    },
    {
        "name": "clip-vit-large-patch14",
        "hub_id": "openai/clip-vit-large-patch14",
        "model_class": "CLIPModel",
        "processor_class": "CLIPProcessor",
        "tasks": ["retrieval", "classification"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],
        "default_dtype": "float32",
        "min_gpu_gb": 8,
    },
    {
        "name": "clip-vit-large-patch14-336",
        "hub_id": "openai/clip-vit-large-patch14-336",
        "model_class": "CLIPModel",
        "processor_class": "CLIPProcessor",
        "tasks": ["retrieval", "classification"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],
        "default_dtype": "float32",
        "min_gpu_gb": 8,
    },
    # ── PaliGemma ─────────────────────────────────────────────────────────────
    {
        "name": "paligemma-3b-pt-224",
        "hub_id": "google/paligemma-3b-pt-224",
        "model_class": "PaliGemmaForConditionalGeneration",
        "processor_class": "PaliGemmaProcessor",
        "tasks": ["captioning", "vqa", "grounding"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "default_dtype": "bfloat16",
        "min_gpu_gb": 8,
    },
    {
        "name": "paligemma-3b-mix-224",
        "hub_id": "google/paligemma-3b-mix-224",
        "model_class": "PaliGemmaForConditionalGeneration",
        "processor_class": "PaliGemmaProcessor",
        "tasks": ["captioning", "vqa", "grounding", "classification"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "default_dtype": "bfloat16",
        "min_gpu_gb": 8,
    },
    {
        "name": "paligemma-3b-mix-448",
        "hub_id": "google/paligemma-3b-mix-448",
        "model_class": "PaliGemmaForConditionalGeneration",
        "processor_class": "PaliGemmaProcessor",
        "tasks": ["captioning", "vqa", "grounding", "classification"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "default_dtype": "bfloat16",
        "min_gpu_gb": 10,
    },
    # ── IDEFICS2 ──────────────────────────────────────────────────────────────
    {
        "name": "idefics2-8b",
        "hub_id": "HuggingFaceM4/idefics2-8b",
        "model_class": "Idefics2ForConditionalGeneration",
        "processor_class": "Idefics2Processor",
        "tasks": ["captioning", "vqa", "chat"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "default_dtype": "bfloat16",
        "min_gpu_gb": 20,
    },
    # ── Florence-2 ───────────────────────────────────────────────────────────
    {
        "name": "florence-2-base",
        "hub_id": "microsoft/Florence-2-base",
        "model_class": "Florence2ForConditionalGeneration",
        "processor_class": "Florence2Processor",
        "tasks": ["captioning", "vqa", "grounding", "classification"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],
        "default_dtype": "float32",
        "min_gpu_gb": 8,
    },
    {
        "name": "florence-2-large",
        "hub_id": "microsoft/Florence-2-large",
        "model_class": "Florence2ForConditionalGeneration",
        "processor_class": "Florence2Processor",
        "tasks": ["captioning", "vqa", "grounding", "classification"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],
        "default_dtype": "float32",
        "min_gpu_gb": 16,
    },
    # ── GIT ──────────────────────────────────────────────────────────────────
    {
        "name": "git-base",
        "hub_id": "microsoft/git-base",
        "model_class": "GitForCausalLM",
        "processor_class": "GitProcessor",
        "tasks": ["captioning", "vqa"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],
        "default_dtype": "float32",
        "min_gpu_gb": 4,
    },
    {
        "name": "git-large",
        "hub_id": "microsoft/git-large",
        "model_class": "GitForCausalLM",
        "processor_class": "GitProcessor",
        "tasks": ["captioning", "vqa"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],
        "default_dtype": "float32",
        "min_gpu_gb": 8,
    },
    {
        "name": "git-base-coco",
        "hub_id": "microsoft/git-base-coco",
        "model_class": "GitForCausalLM",
        "processor_class": "GitProcessor",
        "tasks": ["captioning"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],
        "default_dtype": "float32",
        "min_gpu_gb": 4,
    },
    # ── Original BLIP ─────────────────────────────────────────────────────────
    {
        "name": "blip-base",
        "hub_id": "Salesforce/blip-image-captioning-base",
        "model_class": "BlipForConditionalGeneration",
        "processor_class": "BlipProcessor",
        "tasks": ["captioning"],
        "lora_target_modules": ["query", "value", "key", "dense"],
        "default_dtype": "float32",
        "min_gpu_gb": 4,
    },
    {
        "name": "blip-large",
        "hub_id": "Salesforce/blip-image-captioning-large",
        "model_class": "BlipForConditionalGeneration",
        "processor_class": "BlipProcessor",
        "tasks": ["captioning"],
        "lora_target_modules": ["query", "value", "key", "dense"],
        "default_dtype": "float32",
        "min_gpu_gb": 8,
    },
    {
        "name": "blip-vqa-base",
        "hub_id": "Salesforce/blip-vqa-base",
        "model_class": "BlipForQuestionAnswering",
        "processor_class": "BlipProcessor",
        "tasks": ["vqa"],
        "lora_target_modules": ["query", "value", "key", "dense"],
        "default_dtype": "float32",
        "min_gpu_gb": 4,
    },
    # ── ViLT ─────────────────────────────────────────────────────────────────
    {
        "name": "vilt-vqa",
        "hub_id": "dandelin/vilt-b32-finetuned-vqa",
        "model_class": "ViltForQuestionAnswering",
        "processor_class": "ViltProcessor",
        "tasks": ["vqa", "classification"],
        "lora_target_modules": ["query", "value", "key"],
        "default_dtype": "float32",
        "min_gpu_gb": 4,
    },
    # ── FLAVA ─────────────────────────────────────────────────────────────────
    {
        "name": "flava",
        "hub_id": "facebook/flava-full",
        "model_class": "FlavaModel",
        "processor_class": "FlavaProcessor",
        "tasks": ["retrieval", "classification", "vqa"],
        "lora_target_modules": ["query", "value", "key"],
        "default_dtype": "float32",
        "min_gpu_gb": 8,
    },
    # ── Qwen2-VL ──────────────────────────────────────────────────────────────
    {
        "name": "qwen2-vl-7b",
        "hub_id": "Qwen/Qwen2-VL-7B-Instruct",
        "model_class": "Qwen2VLForConditionalGeneration",
        "processor_class": "Qwen2VLProcessor",
        "tasks": ["captioning", "vqa", "chat", "grounding"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "default_dtype": "bfloat16",
        "min_gpu_gb": 20,
    },
    {
        "name": "qwen2-vl-72b",
        "hub_id": "Qwen/Qwen2-VL-72B-Instruct",
        "model_class": "Qwen2VLForConditionalGeneration",
        "processor_class": "Qwen2VLProcessor",
        "tasks": ["captioning", "vqa", "chat", "grounding"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "default_dtype": "bfloat16",
        "min_gpu_gb": 160,
    },
    # ── Phi-3.5-Vision ────────────────────────────────────────────────────────
    {
        "name": "phi-3.5-vision",
        "hub_id": "microsoft/Phi-3.5-vision-instruct",
        "model_class": "AutoModelForCausalLM",
        "processor_class": "AutoProcessor",
        "tasks": ["captioning", "vqa", "chat", "grounding"],
        "lora_target_modules": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        "default_dtype": "bfloat16",
        "min_gpu_gb": 20,
    },
    # ── InternVL2 ─────────────────────────────────────────────────────────────
    {
        "name": "internvl2-8b",
        "hub_id": "OpenGVLab/InternVL2-8B",
        "model_class": "AutoModel",
        "processor_class": "AutoTokenizer",
        "tasks": ["captioning", "vqa", "chat", "grounding"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "default_dtype": "bfloat16",
        "min_gpu_gb": 20,
    },
    # ── CogVLM2 ───────────────────────────────────────────────────────────────
    {
        "name": "cogvlm2",
        "hub_id": "THUDM/cogvlm2-llama3-chat-19B",
        "model_class": "AutoModelForCausalLM",
        "processor_class": "AutoTokenizer",
        "tasks": ["captioning", "vqa", "chat"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "default_dtype": "bfloat16",
        "min_gpu_gb": 48,
    },
    # ── Moondream2 ────────────────────────────────────────────────────────────
    {
        "name": "moondream2",
        "hub_id": "vikhyatk/moondream2",
        "model_class": "AutoModelForCausalLM",
        "processor_class": "AutoTokenizer",
        "tasks": ["captioning", "vqa", "chat"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],
        "default_dtype": "float16",
        "min_gpu_gb": 6,
    },
    # ── DeepSeek-VL ───────────────────────────────────────────────────────────
    {
        "name": "deepseek-vl-7b",
        "hub_id": "deepseek-ai/deepseek-vl-7b-chat",
        "model_class": "AutoModelForCausalLM",
        "processor_class": "AutoProcessor",
        "tasks": ["captioning", "vqa", "chat"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "default_dtype": "bfloat16",
        "min_gpu_gb": 20,
    },
    # ── Emu3 ──────────────────────────────────────────────────────────────────
    {
        "name": "emu3",
        "hub_id": "BAAI/Emu3-Chat",
        "model_class": "AutoModelForCausalLM",
        "processor_class": "AutoProcessor",
        "tasks": ["captioning", "vqa", "chat"],
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "default_dtype": "bfloat16",
        "min_gpu_gb": 40,
    },
]

# Build the public registry dict  {name -> dict}
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    entry["name"]: {k: v for k, v in entry.items() if k != "name"}
    for entry in _RAW_REGISTRY
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """
    Immutable metadata for a single model checkpoint.

    Attributes
    ----------
    hub_id : str
        Hugging Face Hub repository ID (e.g. ``"Salesforce/blip2-opt-2.7b"``).
    model_class : str
        Transformers class name used to load the model
        (e.g. ``"Blip2ForConditionalGeneration"``).
    processor_class : str
        Transformers class name used to load the processor / tokenizer
        (e.g. ``"Blip2Processor"``).
    tasks : list of str
        Tasks this model natively supports (subset of ``SUPPORTED_TASKS``).
    lora_target_modules : list of str
        Linear layer names to target when applying LoRA / QLoRA adapters.
    default_dtype : str
        Recommended ``torch.dtype`` as a string: ``"float32"``, ``"float16"``,
        or ``"bfloat16"``.
    min_gpu_gb : int
        Approximate minimum VRAM in gigabytes required for inference at the
        model's ``default_dtype`` (without quantisation).
    """

    hub_id: str
    model_class: str
    processor_class: str
    tasks: List[str] = field(default_factory=list)
    lora_target_modules: List[str] = field(default_factory=list)
    default_dtype: str = "float32"
    min_gpu_gb: int = 8

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def torch_dtype(self) -> torch.dtype:
        """Return the resolved ``torch.dtype`` for this model."""
        _map: Dict[str, torch.dtype] = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if self.default_dtype not in _map:
            raise ValueError(
                f"Unknown dtype '{self.default_dtype}'. "
                f"Must be one of {list(_map.keys())}."
            )
        return _map[self.default_dtype]

    def supports_task(self, task: str) -> bool:
        """Return ``True`` if ``task`` is in this model's task list."""
        return task in self.tasks


@dataclass
class TrainingConfig:
    """
    Hyper-parameters and switches that control the training loop.

    Attributes
    ----------
    learning_rate : float
        Peak learning rate used by the optimiser.
    num_epochs : int
        Total number of training epochs.
    batch_size : int
        Per-device training batch size.
    gradient_accumulation_steps : int
        Number of forward passes before each optimiser step.  Effective
        batch size = ``batch_size × gradient_accumulation_steps × num_gpus``.
    warmup_ratio : float
        Fraction of total training steps used for LR warmup.
    weight_decay : float
        L2 regularisation coefficient applied by the optimiser.
    max_grad_norm : float
        Global gradient clipping threshold.
    use_lora : bool
        Enable LoRA (Low-Rank Adaptation) adapters.
    lora_r : int
        LoRA rank (number of decomposition dimensions).
    lora_alpha : int
        LoRA scaling factor (``alpha / r`` scales the adapter weights).
    lora_dropout : float
        Dropout probability applied inside LoRA adapter layers.
    use_qlora : bool
        Enable QLoRA (quantised LoRA).  Requires ``load_in_4bit=True``.
    load_in_4bit : bool
        Load model weights in 4-bit (NF4) precision via ``bitsandbytes``.
    load_in_8bit : bool
        Load model weights in 8-bit precision via ``bitsandbytes``.
    bf16 : bool
        Use ``bfloat16`` mixed-precision training.
    fp16 : bool
        Use ``float16`` mixed-precision training.
    save_steps : int
        Save a checkpoint every ``save_steps`` optimiser steps.
    eval_steps : int
        Run evaluation every ``eval_steps`` optimiser steps.
    logging_steps : int
        Log metrics to console / W&B every ``logging_steps`` steps.
    save_total_limit : int
        Maximum number of checkpoints to keep on disk.
    resume_from_checkpoint : str or None
        Path to a checkpoint directory to resume training from.
    use_wandb : bool
        Enable Weights & Biases experiment tracking.
    wandb_project : str
        W&B project name to log runs under.
    """

    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # LoRA / QLoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_qlora: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # Precision
    bf16: bool = True
    fp16: bool = False

    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None

    # Experiment tracking
    use_wandb: bool = False
    wandb_project: str = "vlm-pipeline"


@dataclass
class DataConfig:
    """
    Dataset source and pre-processing configuration.

    Attributes
    ----------
    dataset_type : str
        Data source format. One of ``"jsonl"``, ``"csv"``, ``"hf"``, ``"coco"``.
    data_dir : str or None
        Root directory containing image assets (used with ``jsonl`` / ``csv``).
    train_file : str or None
        Path to the training annotation file.
    val_file : str or None
        Path to the validation annotation file.
    test_file : str or None
        Path to the test annotation file.
    hf_dataset_name : str or None
        Hugging Face Hub dataset repository ID (used when
        ``dataset_type="hf"``).
    hf_dataset_config : str or None
        Optional configuration name for the HF dataset (e.g. ``"en"``).
    image_col : str
        Column / key name for the image path or URL in the annotation file.
    question_col : str
        Column / key name for the question text (VQA / chat tasks).
    answer_col : str
        Column / key name for the ground-truth answer.
    caption_col : str
        Column / key name for the caption text (captioning tasks).
    max_length : int
        Maximum tokenised sequence length (tokens).
    image_size : int
        Image resolution in pixels (height = width = ``image_size``).
    max_train_samples : int or None
        Cap on the number of training examples (``None`` = use all).
    max_val_samples : int or None
        Cap on the number of validation examples (``None`` = use all).
    """

    dataset_type: str = "jsonl"       # jsonl | csv | hf | coco
    data_dir: Optional[str] = None
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None

    # HuggingFace Hub dataset
    hf_dataset_name: Optional[str] = None
    hf_dataset_config: Optional[str] = None

    # Column names
    image_col: str = "image_path"
    question_col: str = "question"
    answer_col: str = "answer"
    caption_col: str = "caption"

    # Pre-processing
    max_length: int = 512
    image_size: int = 224

    # Subsampling
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None


@dataclass
class PathConfig:
    """
    File-system locations for pipeline artefacts.

    Attributes
    ----------
    output_dir : str
        Root directory where trained model weights and final results are saved.
    checkpoint_dir : str
        Directory for intermediate training checkpoints.
    log_dir : str
        Directory for TensorBoard event files and text logs.
    cache_dir : str or None
        Custom Hugging Face Hub cache directory.  ``None`` uses the
        environment default (``~/.cache/huggingface``).
    """

    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    cache_dir: Optional[str] = None


@dataclass
class VLMConfig:
    """
    Top-level configuration container for a full VLM pipeline run.

    Parameters
    ----------
    model_name : str
        Short registry key (must be a key in ``MODEL_REGISTRY``).
    task : str
        Target task (must be a value in ``SUPPORTED_TASKS``).
    model : ModelConfig
        Model-specific metadata (populated automatically by ``get_config``).
    training : TrainingConfig
        Training hyper-parameters.
    data : DataConfig
        Dataset and pre-processing configuration.
    paths : PathConfig
        File-system path configuration.

    Examples
    --------
    >>> cfg = get_config("llava-1.5-7b", "vqa")
    >>> cfg.model.hub_id
    'llava-hf/llava-1.5-7b-hf'
    >>> cfg.training.use_lora
    True
    """

    model_name: str
    task: str
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the entire configuration to a plain dictionary."""
        return asdict(self)

    def __post_init__(self) -> None:
        """Basic integrity assertions executed after ``__init__``."""
        if not self.model_name:
            raise ValueError("'model_name' must not be empty.")
        if not self.task:
            raise ValueError("'task' must not be empty.")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_config(
    model_name: str,
    task: str,
    **overrides: Any,
) -> VLMConfig:
    """
    Build a fully-populated :class:`VLMConfig` for the requested model and task.

    The function looks up *model_name* in ``MODEL_REGISTRY``, creates a
    :class:`ModelConfig` from the registry entry, then applies any keyword
    overrides.  Overrides are matched against the nested dataclass field names
    using a *dotted-path* syntax:

    * ``training.learning_rate=1e-5``   → sets ``config.training.learning_rate``
    * ``data.image_size=336``           → sets ``config.data.image_size``
    * ``paths.output_dir="my_runs"``    → sets ``config.paths.output_dir``

    Flat names (e.g. ``batch_size=8``) search across all nested configs in
    the order ``training`` → ``data`` → ``paths`` and apply to the first match.

    Parameters
    ----------
    model_name : str
        Registry key (see ``MODEL_REGISTRY``).
    task : str
        Downstream task key (see ``SUPPORTED_TASKS``).
    **overrides : Any
        Arbitrary keyword overrides applied to the nested config objects.

    Returns
    -------
    VLMConfig
        Fully populated configuration object.

    Raises
    ------
    KeyError
        If *model_name* is not found in ``MODEL_REGISTRY``.
    ValueError
        If *task* is not in ``SUPPORTED_TASKS``.

    Examples
    --------
    >>> cfg = get_config(
    ...     "blip2-opt-2.7b",
    ...     "vqa",
    ...     **{"training.learning_rate": 5e-5, "data.batch_size": 8},
    ... )
    >>> cfg.training.learning_rate
    5e-05
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(
            f"Model '{model_name}' not found in MODEL_REGISTRY. "
            f"Available models:\n  {available}"
        )
    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"Task '{task}' is not supported. "
            f"Choose from: {SUPPORTED_TASKS}"
        )

    # Build ModelConfig from registry entry
    registry_entry = copy.deepcopy(MODEL_REGISTRY[model_name])
    model_cfg = ModelConfig(
        hub_id=registry_entry["hub_id"],
        model_class=registry_entry["model_class"],
        processor_class=registry_entry["processor_class"],
        tasks=registry_entry["tasks"],
        lora_target_modules=registry_entry["lora_target_modules"],
        default_dtype=registry_entry["default_dtype"],
        min_gpu_gb=registry_entry["min_gpu_gb"],
    )

    training_cfg = TrainingConfig()
    data_cfg = DataConfig()
    paths_cfg = PathConfig()

    # Apply default dtype precision flags
    if model_cfg.default_dtype == "bfloat16":
        training_cfg.bf16 = True
        training_cfg.fp16 = False
    elif model_cfg.default_dtype == "float16":
        training_cfg.bf16 = False
        training_cfg.fp16 = True

    # Apply image size default from family map
    _family_size = _resolve_family_image_size(model_name)
    if _family_size:
        data_cfg.image_size = _family_size

    # Apply caller-supplied overrides
    _apply_overrides(
        overrides,
        sub_configs={
            "training": training_cfg,
            "data": data_cfg,
            "paths": paths_cfg,
        },
    )

    cfg = VLMConfig(
        model_name=model_name,
        task=task,
        model=model_cfg,
        training=training_cfg,
        data=data_cfg,
        paths=paths_cfg,
    )
    logger.debug("Built VLMConfig for model='%s', task='%s'.", model_name, task)
    return cfg


def _resolve_family_image_size(model_name: str) -> Optional[int]:
    """
    Return the recommended image size for a model by matching its name
    against the ``DEFAULT_IMAGE_SIZES`` family keys.

    Parameters
    ----------
    model_name : str
        Short registry key.

    Returns
    -------
    int or None
        Image size in pixels, or ``None`` if no family match is found.
    """
    # Exact match shortcuts
    _exact: Dict[str, str] = {
        "clip-vit-base-patch32": "clip-base",
        "clip-vit-large-patch14": "clip-large",
        "clip-vit-large-patch14-336": "clip-large-336",
        "paligemma-3b-pt-224": "paligemma-224",
        "paligemma-3b-mix-224": "paligemma-224",
        "paligemma-3b-mix-448": "paligemma-448",
    }
    if model_name in _exact:
        return DEFAULT_IMAGE_SIZES.get(_exact[model_name])

    # Family prefix matching (longest-match wins)
    best_key: Optional[str] = None
    best_len = 0
    for family_key in DEFAULT_IMAGE_SIZES:
        normalised = family_key.replace("-", "_").replace(".", "_")
        candidate = model_name.replace("-", "_").replace(".", "_")
        if candidate.startswith(normalised) and len(normalised) > best_len:
            best_key = family_key
            best_len = len(normalised)

    return DEFAULT_IMAGE_SIZES.get(best_key) if best_key else None


def _apply_overrides(
    overrides: Dict[str, Any],
    sub_configs: Dict[str, Any],
) -> None:
    """
    Mutate nested dataclass instances according to ``overrides``.

    Dotted keys (e.g. ``"training.learning_rate"``) target a specific
    sub-config.  Flat keys are resolved by searching all sub-configs in
    insertion order.

    Parameters
    ----------
    overrides : dict
        Key-value pairs supplied by the caller via ``**overrides``.
    sub_configs : dict
        Mapping of prefix → dataclass instance (e.g. ``{"training": cfg}``).

    Raises
    ------
    AttributeError
        If a key cannot be matched to any field in any sub-config.
    """
    for raw_key, value in overrides.items():
        if "." in raw_key:
            prefix, attr = raw_key.split(".", maxsplit=1)
            if prefix not in sub_configs:
                raise AttributeError(
                    f"Unknown config namespace '{prefix}'. "
                    f"Expected one of: {list(sub_configs.keys())}."
                )
            target = sub_configs[prefix]
            if not hasattr(target, attr):
                raise AttributeError(
                    f"Config '{prefix}' has no field '{attr}'."
                )
            setattr(target, attr, value)
        else:
            # Search all sub-configs for the first matching field name
            matched = False
            for cfg_obj in sub_configs.values():
                if hasattr(cfg_obj, raw_key):
                    setattr(cfg_obj, raw_key, value)
                    matched = True
                    break
            if not matched:
                raise AttributeError(
                    f"Override key '{raw_key}' did not match any field in "
                    f"configs: {list(sub_configs.keys())}."
                )


def list_models(task: Optional[str] = None) -> List[str]:
    """
    Return a sorted list of model names available in the registry.

    Parameters
    ----------
    task : str or None
        If provided, only models that support *task* are returned.
        Must be one of ``SUPPORTED_TASKS`` when not ``None``.

    Returns
    -------
    list of str
        Sorted list of registry model keys.

    Raises
    ------
    ValueError
        If *task* is not ``None`` and is not in ``SUPPORTED_TASKS``.

    Examples
    --------
    >>> vqa_models = list_models(task="vqa")
    >>> "blip2-opt-2.7b" in vqa_models
    True
    >>> retrieval_models = list_models(task="retrieval")
    >>> "clip-vit-base-patch32" in retrieval_models
    True
    """
    if task is not None and task not in SUPPORTED_TASKS:
        raise ValueError(
            f"Task '{task}' is not recognised. "
            f"Valid tasks: {SUPPORTED_TASKS}"
        )

    result: List[str] = []
    for name, meta in MODEL_REGISTRY.items():
        if task is None or task in meta.get("tasks", []):
            result.append(name)
    return sorted(result)


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Return a rich information dictionary for a single model.

    The returned dict contains all registry metadata plus derived fields:

    * ``"is_generative"`` – whether the model generates free-form text.
    * ``"is_contrastive"`` – whether the model uses a contrastive objective.
    * ``"torch_dtype"``    – the resolved ``torch.dtype`` object.
    * ``"prompt_template_key"`` – key into ``PROMPT_TEMPLATES`` (best-effort).

    Parameters
    ----------
    model_name : str
        Short registry key.

    Returns
    -------
    dict
        Dictionary of model metadata plus derived convenience fields.

    Raises
    ------
    KeyError
        If *model_name* is not found in ``MODEL_REGISTRY``.

    Examples
    --------
    >>> info = get_model_info("llava-1.5-7b")
    >>> info["hub_id"]
    'llava-hf/llava-1.5-7b-hf'
    >>> info["is_generative"]
    True
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(
            f"Model '{model_name}' not found. Available: {available}"
        )

    meta = copy.deepcopy(MODEL_REGISTRY[model_name])

    # Resolve torch.dtype
    _dtype_map: Dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    meta["torch_dtype"] = _dtype_map.get(meta.get("default_dtype", "float32"), torch.float32)

    # Derived membership flags
    meta["is_generative"] = model_name in GENERATIVE_MODELS
    meta["is_contrastive"] = model_name in CONTRASTIVE_MODELS

    # Best-effort prompt template key
    meta["prompt_template_key"] = _find_prompt_template_key(model_name)

    # Recommended image size
    meta["recommended_image_size"] = _resolve_family_image_size(model_name)

    # Metrics for supported tasks
    meta["supported_metrics"] = {
        t: TASK_METRICS.get(t, []) for t in meta.get("tasks", [])
    }

    return meta


def _find_prompt_template_key(model_name: str) -> Optional[str]:
    """
    Heuristically match a model name to a key in ``PROMPT_TEMPLATES``.

    Parameters
    ----------
    model_name : str
        Short registry key.

    Returns
    -------
    str or None
        A key from ``PROMPT_TEMPLATES``, or ``None`` if no match is found.
    """
    # Check longest matching prefix first (most specific wins)
    for key in sorted(PROMPT_TEMPLATES.keys(), key=len, reverse=True):
        if model_name.startswith(key):
            return key
    # Fallback: partial substring scan
    for key in sorted(PROMPT_TEMPLATES.keys(), key=len, reverse=True):
        if key in model_name:
            return key
    return None


def validate_config(config: VLMConfig) -> List[str]:
    """
    Validate a :class:`VLMConfig` and return a list of human-readable warnings.

    Checks performed
    ----------------
    * ``model_name`` exists in ``MODEL_REGISTRY``.
    * ``task`` is in ``SUPPORTED_TASKS``.
    * The model supports the requested task.
    * ``bf16`` and ``fp16`` are not both enabled simultaneously.
    * QLoRA requires ``load_in_4bit=True``.
    * ``load_in_4bit`` and ``load_in_8bit`` are not both enabled.
    * ``lora_alpha`` ≥ ``lora_r`` (common best practice).
    * ``data.dataset_type`` is a recognised value.
    * If ``dataset_type`` is ``"hf"``, ``hf_dataset_name`` is set.
    * If ``dataset_type`` is not ``"hf"``, at least one file/dir is set.
    * ``training.learning_rate`` is in a sane range.
    * ``training.batch_size`` is a positive integer.
    * GPU VRAM check: warn when estimated VRAM < ``model.min_gpu_gb``.
    * ``paths.output_dir`` parent is writable.

    Parameters
    ----------
    config : VLMConfig
        The configuration to validate.

    Returns
    -------
    list of str
        A list of warning strings.  An empty list means no issues found.

    Examples
    --------
    >>> cfg = get_config("blip2-opt-2.7b", "retrieval")
    >>> warnings = validate_config(cfg)
    >>> any("does not natively support" in w for w in warnings)
    True
    """
    warnings: List[str] = []
    tr = config.training
    da = config.data
    pa = config.paths
    mo = config.model

    # 1. Model in registry
    if config.model_name not in MODEL_REGISTRY:
        warnings.append(
            f"[ERROR] model_name='{config.model_name}' is not in MODEL_REGISTRY."
        )

    # 2. Task validity
    if config.task not in SUPPORTED_TASKS:
        warnings.append(
            f"[ERROR] task='{config.task}' is not in SUPPORTED_TASKS={SUPPORTED_TASKS}."
        )

    # 3. Model ↔ task compatibility
    if config.model_name in MODEL_REGISTRY:
        supported = MODEL_REGISTRY[config.model_name].get("tasks", [])
        if config.task not in supported:
            warnings.append(
                f"[WARNING] Model '{config.model_name}' does not natively support "
                f"task='{config.task}'. Supported tasks: {supported}."
            )

    # 4. Precision flags
    if tr.bf16 and tr.fp16:
        warnings.append(
            "[ERROR] bf16=True and fp16=True cannot both be enabled. "
            "Disable one of them."
        )

    # 5. QLoRA requires 4-bit
    if tr.use_qlora and not tr.load_in_4bit:
        warnings.append(
            "[WARNING] use_qlora=True but load_in_4bit=False. "
            "QLoRA requires 4-bit quantisation; set load_in_4bit=True."
        )

    # 6. Cannot load in 4-bit AND 8-bit simultaneously
    if tr.load_in_4bit and tr.load_in_8bit:
        warnings.append(
            "[ERROR] load_in_4bit=True and load_in_8bit=True are mutually "
            "exclusive. Choose one or neither."
        )

    # 7. LoRA alpha ≥ rank
    if tr.use_lora and tr.lora_alpha < tr.lora_r:
        warnings.append(
            f"[WARNING] lora_alpha ({tr.lora_alpha}) < lora_r ({tr.lora_r}). "
            "Typically lora_alpha should be >= lora_r (often 2× lora_r)."
        )

    # 8. Dataset type
    valid_dataset_types = {"jsonl", "csv", "hf", "coco"}
    if da.dataset_type not in valid_dataset_types:
        warnings.append(
            f"[ERROR] data.dataset_type='{da.dataset_type}' is not recognised. "
            f"Choose from: {sorted(valid_dataset_types)}."
        )

    # 9. HF dataset name required for hf type
    if da.dataset_type == "hf" and not da.hf_dataset_name:
        warnings.append(
            "[ERROR] data.dataset_type='hf' requires data.hf_dataset_name to be set."
        )

    # 10. At least one data source for non-hf types
    if da.dataset_type != "hf":
        if not any([da.train_file, da.val_file, da.test_file, da.data_dir]):
            warnings.append(
                "[WARNING] No data source specified. "
                "Set at least one of: data.train_file, data.val_file, "
                "data.test_file, or data.data_dir."
            )

    # 11. Learning rate sanity
    if not (1e-8 <= tr.learning_rate <= 1.0):
        warnings.append(
            f"[WARNING] training.learning_rate={tr.learning_rate} is outside "
            "the typical range [1e-8, 1.0]."
        )

    # 12. Batch size positive
    if tr.batch_size <= 0:
        warnings.append(
            f"[ERROR] training.batch_size={tr.batch_size} must be a positive integer."
        )

    # 13. GPU VRAM availability (best-effort)
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        available_gb = gpu_props.total_memory / (1024 ** 3)
        required_gb = mo.min_gpu_gb
        # Account for quantisation reductions
        if tr.load_in_4bit:
            required_gb = required_gb * 0.25
        elif tr.load_in_8bit:
            required_gb = required_gb * 0.5

        if available_gb < required_gb:
            warnings.append(
                f"[WARNING] Model '{config.model_name}' requires ~{required_gb:.1f} GB VRAM "
                f"but GPU 0 has {available_gb:.1f} GB. "
                "Consider quantisation (load_in_4bit/load_in_8bit) or a larger GPU."
            )
    else:
        warnings.append(
            "[INFO] No CUDA GPU detected. Pipeline will run on CPU (very slow for large models)."
        )

    # 14. Output directory writability
    out_parent = os.path.dirname(os.path.abspath(pa.output_dir)) or "."
    if not os.access(out_parent, os.W_OK):
        warnings.append(
            f"[WARNING] Output directory parent '{out_parent}' is not writable. "
            "Check filesystem permissions."
        )

    return warnings


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("=" * 70)
    print(f"MODEL_REGISTRY contains {len(MODEL_REGISTRY)} models.")
    print(f"GENERATIVE_MODELS : {len(GENERATIVE_MODELS)}")
    print(f"CONTRASTIVE_MODELS: {len(CONTRASTIVE_MODELS)}")
    print()

    # --- list_models demo ---
    print("VQA-capable models:")
    for m in list_models("vqa"):
        print(f"  {m}")
    print()

    # --- get_config demo ---
    cfg = get_config(
        "llava-1.5-7b",
        "vqa",
        **{"training.learning_rate": 1e-4, "data.image_size": 336},
    )
    print("VLMConfig (llava-1.5-7b / vqa):")
    print(json.dumps(cfg.to_dict(), indent=2, default=str))
    print()

    # --- validate_config demo ---
    print("Validation warnings:")
    issues = validate_config(cfg)
    if issues:
        for w in issues:
            print(f"  {w}")
    else:
        print("  No issues found.")
    print()

    # --- get_model_info demo ---
    info = get_model_info("clip-vit-base-patch32")
    print(f"clip-vit-base-patch32 info:")
    for k, v in info.items():
        print(f"  {k}: {v}")
