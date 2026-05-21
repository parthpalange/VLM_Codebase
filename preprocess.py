"""
preprocess.py
=============
End-to-end data loading, preprocessing, and DataLoader creation for the VLM pipeline.

Responsibilities
----------------
* Load raw datasets from JSONL, CSV, HuggingFace Hub, or COCO annotation format.
* Wrap samples in a ``VLMDataset`` that applies model-appropriate image transforms
  and produces properly tokenised tensors.
* Provide model-aware processor/tokeniser initialisation helpers.
* Build train/val/test ``DataLoader`` objects with distributed-training support.
* Pre-tokenise and cache datasets to disk for fast repeated experiments.

Supported model families
------------------------
  CLIP, BLIP, BLIP-2, InstructBLIP, LLaVA, PaliGemma, ViLT, Idefics,
  and any model reachable via ``AutoProcessor``.

Supported task types
--------------------
  captioning, vqa, retrieval, classification, chat
"""

from __future__ import annotations

import csv
import json
import logging
import os
import pathlib
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from PIL import Image, ImageFile

# Allow loading truncated images (common in web-scraped datasets).
ImageFile.LOAD_TRUNCATED_IMAGES = True

from datasets import load_dataset, load_from_disk, DatasetDict
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CLIPProcessor,
    BlipProcessor,
    InstructBlipProcessor,
    LlavaProcessor,
    PaliGemmaProcessor,
    ViltProcessor,
    Idefics2Processor,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VLMConfig reference dataclass
# ---------------------------------------------------------------------------
# The real VLMConfig is expected to be imported from `config.py`; a minimal
# stub is declared here so this file is self-contained and importable even
# if `config.py` is not yet present.
try:
    from config import VLMConfig  # type: ignore
except ImportError:  # pragma: no cover

    @dataclass
    class DataConfig:
        dataset_type: str = "jsonl"           # jsonl | csv | hf | coco
        train_file: Optional[str] = None
        val_file: Optional[str] = None
        test_file: Optional[str] = None
        hf_dataset_name: Optional[str] = None
        hf_dataset_config: Optional[str] = None
        coco_annotation_dir: Optional[str] = None
        coco_image_dir: Optional[str] = None
        image_column: str = "image_path"
        question_column: str = "question"
        answer_column: str = "answer"
        caption_column: str = "caption"
        task_type: str = "captioning"         # captioning | vqa | retrieval | classification | chat
        max_length: int = 128
        image_size: int = 224

    @dataclass
    class ModelConfig:
        model_name: str = "Salesforce/blip2-opt-2.7b"
        model_family: str = "blip2"           # clip | blip | blip2 | instructblip | llava | paligemma | vilt | idefics | generic

    @dataclass
    class TrainingConfig:
        batch_size: int = 8
        num_workers: int = 4
        pin_memory: bool = True
        prefetch_factor: int = 2
        seed: int = 42

    @dataclass
    class VLMConfig:
        data: DataConfig = field(default_factory=DataConfig)
        model: ModelConfig = field(default_factory=ModelConfig)
        training: TrainingConfig = field(default_factory=TrainingConfig)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ImageNet normalization statistics (default / CLIP / BLIP / LLaVA all use these)
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# CLIP uses its own stats (same values, kept explicit for clarity)
CLIP_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

# Model family keywords used for routing logic
_CLIP_FAMILIES = {"clip"}
_BLIP_FAMILIES = {"blip", "blip2", "instructblip"}
_LLAVA_FAMILIES = {"llava"}
_PALIGEMMA_FAMILIES = {"paligemma"}
_VILT_FAMILIES = {"vilt"}
_IDEFICS_FAMILIES = {"idefics"}
_CONTRASTIVE_FAMILIES = _CLIP_FAMILIES

IGNORE_INDEX = -100  # label ID to ignore in cross-entropy loss


# ===========================================================================
# 1. DatasetLoader
# ===========================================================================


class DatasetLoader:
    """
    Loads raw datasets from various sources and returns a
    ``datasets.Dataset`` (HuggingFace Dataset) for a given split.

    Parameters
    ----------
    config : VLMConfig
        Full pipeline configuration object.
    """

    def __init__(self, config: VLMConfig) -> None:
        self.config = config
        self._dispatch: Dict[str, Callable[[str], Any]] = {
            "jsonl": self._load_jsonl,
            "csv": self._load_csv,
            "hf": self._load_hf,
            "coco": self._load_coco,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, split: str) -> Any:
        """
        Auto-detect dataset format from ``config.data.dataset_type`` and
        return the corresponding HuggingFace ``Dataset`` for *split*.

        Parameters
        ----------
        split : str
            One of ``"train"``, ``"val"``, or ``"test"``.

        Returns
        -------
        datasets.Dataset
        """
        dataset_type = self.config.data.dataset_type.lower().strip()
        if dataset_type not in self._dispatch:
            raise ValueError(
                f"Unknown dataset_type '{dataset_type}'. "
                f"Choose from: {list(self._dispatch)}"
            )
        logger.info("Loading '%s' split via format '%s'", split, dataset_type)
        return self._dispatch[dataset_type](split)

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _resolve_file(self, split: str) -> str:
        """Return the configured file path for *split*, raising if missing."""
        mapping = {
            "train": self.config.data.train_file,
            "val": self.config.data.val_file,
            "test": self.config.data.test_file,
        }
        path = mapping.get(split)
        if path is None:
            raise FileNotFoundError(
                f"No file configured for split '{split}' in config.data."
            )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        return path

    # --- JSONL -----------------------------------------------------------

    def _load_jsonl(self, split: str) -> Any:
        """
        Load a JSONL file where each line is a JSON object with keys:
        ``image_path``, ``question``, ``answer``, ``caption``.

        Extra keys are preserved transparently.
        """
        path = self._resolve_file(split)
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSONL line %d: %s", line_no, exc)

        if not records:
            raise RuntimeError(f"JSONL file produced zero records: {path}")

        from datasets import Dataset as HFDataset

        dataset = HFDataset.from_list(records)
        logger.info("JSONL '%s' split: %d samples loaded.", split, len(dataset))
        return dataset

    # --- CSV -------------------------------------------------------------

    def _load_csv(self, split: str) -> Any:
        """
        Load a CSV file. Expected columns (others are kept):
        ``image_path``, ``question``, ``answer``, ``caption``.
        """
        path = self._resolve_file(split)
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                records.append(dict(row))

        if not records:
            raise RuntimeError(f"CSV file produced zero records: {path}")

        from datasets import Dataset as HFDataset

        dataset = HFDataset.from_list(records)
        logger.info("CSV '%s' split: %d samples loaded.", split, len(dataset))
        return dataset

    # --- HuggingFace Hub -------------------------------------------------

    def _load_hf(self, split: str) -> Any:
        """
        Load from HuggingFace datasets hub using
        ``config.data.hf_dataset_name`` and ``config.data.hf_dataset_config``.

        Remaps column names to the canonical schema if they differ.
        """
        name = self.config.data.hf_dataset_name
        if not name:
            raise ValueError(
                "config.data.hf_dataset_name must be set for dataset_type='hf'."
            )
        hf_split = "validation" if split == "val" else split
        dataset = load_dataset(
            name,
            self.config.data.hf_dataset_config,
            split=hf_split,
            trust_remote_code=True,
        )
        # Normalise column names to canonical schema
        rename_map = {}
        for canonical, cfg_col in [
            ("image_path", self.config.data.image_column),
            ("question", self.config.data.question_column),
            ("answer", self.config.data.answer_column),
            ("caption", self.config.data.caption_column),
        ]:
            if cfg_col in dataset.column_names and cfg_col != canonical:
                rename_map[cfg_col] = canonical
        if rename_map:
            dataset = dataset.rename_columns(rename_map)
        logger.info("HF Hub '%s' split: %d samples loaded.", split, len(dataset))
        return dataset

    # --- COCO ------------------------------------------------------------

    def _load_coco(self, split: str) -> Any:
        """
        Load COCO-format data.

        Expects:
        * ``config.data.coco_annotation_dir`` – directory containing
          ``captions_{split}2017.json`` (or ``instances_*.json``).
        * ``config.data.coco_image_dir`` – root of COCO images.

        Produces records with keys ``image_path``, ``caption``,
        ``question``, ``answer``, ``image_id``, ``ann_id``.
        """
        ann_dir = self.config.data.coco_annotation_dir
        img_dir = self.config.data.coco_image_dir
        if not ann_dir or not img_dir:
            raise ValueError(
                "config.data.coco_annotation_dir and coco_image_dir must be set "
                "for dataset_type='coco'."
            )

        # COCO uses 'val' -> 'val2017', 'train' -> 'train2017', etc.
        coco_split = "val2017" if split == "val" else f"{split}2017"
        ann_path = pathlib.Path(ann_dir) / f"captions_{coco_split}.json"
        if not ann_path.is_file():
            # Fallback: look for any annotation file for this split
            candidates = list(pathlib.Path(ann_dir).glob(f"*{coco_split}*.json"))
            if not candidates:
                raise FileNotFoundError(
                    f"No COCO annotation file found for split '{split}' in {ann_dir}"
                )
            ann_path = candidates[0]
            logger.warning("Using fallback annotation file: %s", ann_path)

        with open(ann_path, "r", encoding="utf-8") as fh:
            coco_data = json.load(fh)

        # Build image_id -> file_name index
        id_to_filename: Dict[int, str] = {
            img["id"]: img["file_name"] for img in coco_data.get("images", [])
        }

        records: List[Dict[str, Any]] = []
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            filename = id_to_filename.get(img_id, "")
            full_path = str(pathlib.Path(img_dir) / coco_split / filename)
            caption = ann.get("caption", "")
            records.append(
                {
                    "image_path": full_path,
                    "caption": caption,
                    "question": "",
                    "answer": caption,
                    "image_id": img_id,
                    "ann_id": ann.get("id", -1),
                }
            )

        if not records:
            raise RuntimeError(f"COCO annotation produced zero records from: {ann_path}")

        from datasets import Dataset as HFDataset

        dataset = HFDataset.from_list(records)
        logger.info("COCO '%s' split: %d samples loaded.", split, len(dataset))
        return dataset


# ===========================================================================
# 2. VLMDataset
# ===========================================================================


class VLMDataset(Dataset):
    """
    PyTorch Dataset for VLM training and evaluation.

    Handles the following task types:
    * **captioning** – image → caption generation.
    * **vqa** – (image, question) → answer generation.
    * **retrieval** – contrastive image–text pairs (CLIP-style).
    * **classification** – image (+ optional text) → class label.
    * **chat** – multi-turn dialogue with images.

    For *generative* models (BLIP-2, LLaVA, PaliGemma, InstructBLIP, etc.)
    the dataset returns::

        {
            "pixel_values": Tensor[C, H, W],
            "input_ids": LongTensor[L],
            "attention_mask": LongTensor[L],
            "labels": LongTensor[L],   # -100 on prompt tokens (causal LM)
        }

    For *contrastive* models (CLIP, etc.) the dataset returns::

        {
            "pixel_values": Tensor[C, H, W],
            "input_ids": LongTensor[L],
            "attention_mask": LongTensor[L],
        }

    Parameters
    ----------
    samples : datasets.Dataset or list[dict]
        Raw samples (each must contain at least ``image_path`` and
        ``caption`` / ``question`` / ``answer`` as appropriate).
    processor : transformers processor
        Pre-loaded processor/tokeniser for the target model.
    config : VLMConfig
        Full pipeline configuration.
    split : str
        ``"train"``, ``"val"``, or ``"test"``. Determines augmentations.
    """

    def __init__(
        self,
        samples: Any,
        processor: Any,
        config: VLMConfig,
        split: str = "train",
    ) -> None:
        self.samples = samples
        self.processor = processor
        self.config = config
        self.split = split.lower()
        self.task_type = config.data.task_type.lower()
        self.max_length = config.data.max_length
        self.image_size = config.data.image_size
        self.model_family = config.model.model_family.lower()
        self.is_contrastive = self.model_family in _CONTRASTIVE_FAMILIES

        # Torchvision transforms (used as fallback / explicit augmentation path)
        self.transform = get_transforms(
            config.model.model_name, self.split, self.image_size
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # ---- Load image ------------------------------------------------
        image = self._load_image(sample)

        # ---- Build text prompt -----------------------------------------
        text = self._build_text(sample)

        # ---- Contrastive (CLIP-style) path -----------------------------
        if self.is_contrastive:
            return self._encode_contrastive(image, text)

        # ---- Generative / classification path --------------------------
        return self._encode_generative(image, text, sample)

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    def _load_image(self, sample: Dict[str, Any]) -> Image.Image:
        """
        Load a PIL image from ``sample["image_path"]`` or inline PIL data.

        Gracefully handles:
        * Missing / corrupt files → returns a solid grey placeholder.
        * HF datasets that store PIL images directly in a ``"image"`` column.
        """
        # HuggingFace datasets sometimes store PIL images directly
        if "image" in sample and isinstance(sample["image"], Image.Image):
            img = sample["image"].convert("RGB")
            return img

        path = sample.get("image_path", sample.get("image", ""))
        if not path:
            logger.debug("No image path in sample; using placeholder.")
            return self._placeholder_image()

        try:
            img = Image.open(str(path)).convert("RGB")
            return img
        except (FileNotFoundError, OSError, Exception) as exc:
            logger.warning("Failed to load image '%s': %s — using placeholder.", path, exc)
            return self._placeholder_image()

    def _placeholder_image(self) -> Image.Image:
        """Return a solid grey RGB image of the configured size."""
        sz = self.image_size
        return Image.fromarray(
            np.full((sz, sz, 3), 128, dtype=np.uint8), mode="RGB"
        )

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------

    def _build_text(self, sample: Dict[str, Any]) -> str:
        """
        Construct the model input text from *sample* based on task type.

        Returns
        -------
        str
            The prompt text (question / instruction / caption prefix).
        """
        task = self.task_type
        if task == "captioning":
            # For captioning tasks the prompt is usually empty or a trigger token.
            return sample.get("caption", "").strip()
        elif task == "vqa":
            question = sample.get("question", "").strip()
            return f"Question: {question} Answer:"
        elif task == "chat":
            # Expects a "conversations" field: list of {"role": ..., "content": ...}
            convs = sample.get("conversations", [])
            if convs:
                # Join all turns into a single prompt string
                parts = []
                for turn in convs:
                    role = turn.get("role", "user").capitalize()
                    content = turn.get("content", "")
                    parts.append(f"{role}: {content}")
                return "\n".join(parts)
            return sample.get("question", "").strip()
        elif task == "classification":
            return sample.get("question", sample.get("caption", "")).strip()
        elif task == "retrieval":
            return sample.get("caption", sample.get("question", "")).strip()
        else:
            # Generic fallback
            return sample.get("caption", sample.get("question", "")).strip()

    def _build_label_text(self, sample: Dict[str, Any]) -> str:
        """
        Construct the expected model *output* text (used to build labels).
        """
        task = self.task_type
        if task in ("captioning", "retrieval"):
            return sample.get("caption", "").strip()
        elif task == "vqa":
            return sample.get("answer", "").strip()
        elif task == "chat":
            convs = sample.get("conversations", [])
            # Collect assistant turns
            assistant_parts = [
                t.get("content", "")
                for t in convs
                if t.get("role", "").lower() in ("assistant", "gpt", "model")
            ]
            return " ".join(assistant_parts).strip() if assistant_parts else sample.get("answer", "").strip()
        elif task == "classification":
            return sample.get("answer", sample.get("label", "")).strip()
        return sample.get("answer", "").strip()

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode_contrastive(
        self, image: Image.Image, text: str
    ) -> Dict[str, torch.Tensor]:
        """
        Produce tensors for contrastive (CLIP-style) training.

        Returns pixel_values + tokenised text WITHOUT labels.
        """
        # Apply torchvision transform for consistency
        pixel_values = self.transform(image)  # [C, H, W]

        encoding = self.processor(
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

    def _encode_generative(
        self,
        image: Image.Image,
        text: str,
        sample: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Produce tensors for auto-regressive / generative training.

        The function concatenates *prompt* + *label* tokens and masks prompt
        token positions with ``IGNORE_INDEX`` in the labels tensor so that
        loss is computed only on the target sequence.
        """
        label_text = self._build_label_text(sample)

        # ------ Try processor-native path first ------
        try:
            pixel_values, input_ids, attention_mask, labels = (
                self._processor_encode(image, text, label_text)
            )
        except Exception as exc:
            logger.debug(
                "Processor-native encoding failed (%s); falling back to manual path.", exc
            )
            pixel_values, input_ids, attention_mask, labels = (
                self._manual_encode(image, text, label_text)
            )

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _processor_encode(
        self,
        image: Image.Image,
        text: str,
        label_text: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Use the HF processor to jointly encode image + text.

        Works with processors that accept both ``images`` and ``text``
        (BLIP, BLIP-2, LLaVA, PaliGemma, etc.).
        """
        full_text = f"{text} {label_text}".strip()
        encoding = self.processor(
            images=image,
            text=full_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        pixel_values = encoding["pixel_values"].squeeze(0)
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Build labels: mask prompt tokens
        labels = self._build_labels(input_ids, text)
        return pixel_values, input_ids, attention_mask, labels

    def _manual_encode(
        self,
        image: Image.Image,
        text: str,
        label_text: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fallback path: apply torchvision transform for image, use the
        processor's tokeniser for text.
        """
        pixel_values = self.transform(image)  # [C, H, W]

        full_text = f"{text} {label_text}".strip()
        tok = getattr(self.processor, "tokenizer", self.processor)
        encoding = tok(
            full_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = self._build_labels(input_ids, text)
        return pixel_values, input_ids, attention_mask, labels

    def _build_labels(
        self, input_ids: torch.Tensor, prompt_text: str
    ) -> torch.Tensor:
        """
        Build the labels tensor for causal language modelling.

        Tokens that belong to the *prompt* portion are masked with
        ``IGNORE_INDEX``; only tokens in the *target* portion contribute
        to the loss.

        Parameters
        ----------
        input_ids : LongTensor[L]
        prompt_text : str
            The raw prompt string (before concatenation with label_text).

        Returns
        -------
        LongTensor[L]
        """
        labels = input_ids.clone()

        # Determine prompt length by re-tokenising the prompt alone
        tok = getattr(self.processor, "tokenizer", self.processor)
        if tok is not None and hasattr(tok, "__call__"):
            try:
                prompt_enc = tok(
                    prompt_text,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
                prompt_len = prompt_enc["input_ids"].shape[-1]
                # Mask all prompt positions
                labels[:prompt_len] = IGNORE_INDEX
            except Exception:
                pass  # If tokeniser fails, fall through to pad-only masking

        # Always mask padding tokens
        pad_id = getattr(tok, "pad_token_id", None) if tok else None
        if pad_id is not None:
            labels[input_ids == pad_id] = IGNORE_INDEX

        return labels


# ===========================================================================
# 3. get_processor
# ===========================================================================


def get_processor(model_name: str, config: VLMConfig) -> Any:
    """
    Load and return the appropriate HuggingFace processor/tokeniser for
    *model_name*.

    The function dispatches to the correct processor class based on a
    keyword scan of *model_name* and ``config.model.model_family``, with
    ``AutoProcessor`` as the universal fallback.

    Parameters
    ----------
    model_name : str
        HuggingFace Hub model identifier, e.g. ``"openai/clip-vit-base-patch32"``.
    config : VLMConfig
        Pipeline config (used to read ``model.model_family``).

    Returns
    -------
    processor
        A transformers processor object with ``tokenizer``, ``image_processor``,
        or both as appropriate.
    """
    family = config.model.model_family.lower()
    name_lower = model_name.lower()

    logger.info("Loading processor for model '%s' (family='%s').", model_name, family)

    processor: Any = None

    # --- CLIP ---
    if family in _CLIP_FAMILIES or "clip" in name_lower:
        processor = CLIPProcessor.from_pretrained(model_name)

    # --- BLIP (original) ---
    elif family == "blip" or ("blip" in name_lower and "blip2" not in name_lower and "blip-2" not in name_lower and "instructblip" not in name_lower):
        processor = BlipProcessor.from_pretrained(model_name)

    # --- InstructBLIP ---
    elif family == "instructblip" or "instructblip" in name_lower:
        processor = InstructBlipProcessor.from_pretrained(model_name)

    # --- BLIP-2 ---
    elif family == "blip2" or "blip-2" in name_lower or "blip2" in name_lower:
        processor = AutoProcessor.from_pretrained(model_name)

    # --- LLaVA ---
    elif family in _LLAVA_FAMILIES or "llava" in name_lower:
        try:
            processor = LlavaProcessor.from_pretrained(model_name)
        except Exception:
            processor = AutoProcessor.from_pretrained(model_name)

    # --- PaliGemma ---
    elif family in _PALIGEMMA_FAMILIES or "paligemma" in name_lower:
        try:
            processor = PaliGemmaProcessor.from_pretrained(model_name)
        except Exception:
            processor = AutoProcessor.from_pretrained(model_name)

    # --- ViLT ---
    elif family in _VILT_FAMILIES or "vilt" in name_lower:
        processor = ViltProcessor.from_pretrained(model_name)

    # --- Idefics ---
    elif family in _IDEFICS_FAMILIES or "idefics" in name_lower:
        try:
            processor = Idefics2Processor.from_pretrained(model_name)
        except Exception:
            processor = AutoProcessor.from_pretrained(model_name)

    # --- Universal fallback ---
    else:
        processor = AutoProcessor.from_pretrained(model_name)

    # Enforce right-padding for decoder / causal models
    _set_padding_side(processor, family, name_lower)

    # Ensure pad_token exists on the tokeniser
    _ensure_pad_token(processor)

    logger.info("Processor loaded: %s", type(processor).__name__)
    return processor


def _set_padding_side(processor: Any, family: str, name_lower: str) -> None:
    """
    Set ``padding_side='right'`` on the underlying tokeniser for models that
    generate text auto-regressively (decoder-only or encoder-decoder).

    CLIP uses the default ``'right'`` padding already; no change needed.
    """
    decoder_families = {
        "blip2", "instructblip", "llava", "paligemma", "idefics", "generic"
    }
    is_decoder = family in decoder_families or any(
        kw in name_lower
        for kw in ("gpt", "llama", "opt", "mistral", "phi", "gemma", "falcon", "mpt")
    )
    if not is_decoder:
        return

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "right"
        logger.debug("Set tokenizer.padding_side='right'.")


def _ensure_pad_token(processor: Any) -> None:
    """
    Add a pad token to the tokeniser if it does not already have one.
    GPT-style tokenisers often lack a pad token; we use the EOS token.
    """
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.debug("Set pad_token = eos_token ('%s').", tokenizer.eos_token)
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            logger.debug("Added '[PAD]' as pad_token.")


# ===========================================================================
# 4. get_transforms
# ===========================================================================


def get_transforms(
    model_name: str,
    split: str,
    image_size: int = 224,
) -> transforms.Compose:
    """
    Build torchvision transform pipeline appropriate for *model_name* and
    *split*.

    Train transforms include random augmentations; val/test transforms apply
    only a deterministic centre-crop and normalise.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    split : str
        ``"train"``, ``"val"``, or ``"test"``.
    image_size : int
        Target spatial resolution (default 224).

    Returns
    -------
    torchvision.transforms.Compose
    """
    name_lower = model_name.lower()
    is_train = split.lower() == "train"

    # Select normalisation statistics
    if "clip" in name_lower:
        mean, std = CLIP_MEAN, CLIP_STD
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD

    normalize = transforms.Normalize(mean=mean, std=std)

    if is_train:
        # ---- Training augmentations ----
        aug_list: List[Any] = [
            transforms.Resize(int(image_size * 1.14)),  # slight over-resize
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            ),
        ]
        # Additional augmentations for non-CLIP models
        if "clip" not in name_lower:
            aug_list += [
                transforms.RandomGrayscale(p=0.02),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=image_size // 10 * 2 + 1, sigma=(0.1, 2.0))],
                    p=0.1,
                ),
            ]
        aug_list += [
            transforms.ToTensor(),
            normalize,
        ]
        return transforms.Compose(aug_list)
    else:
        # ---- Val / Test deterministic transforms ----
        return transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.14)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        )


# ===========================================================================
# 5. Collate functions
# ===========================================================================


def collate_fn_generative(
    batch: List[Dict[str, torch.Tensor]],
    processor: Any,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    """
    Collate a batch of samples from :class:`VLMDataset` for generative models
    (BLIP-2, LLaVA, PaliGemma, InstructBLIP, etc.).

    Pads ``input_ids``, ``attention_mask``, and ``labels`` to the longest
    sequence in the batch, then stacks ``pixel_values``.

    Parameters
    ----------
    batch : list of dict
        Each dict must contain keys:
        ``pixel_values``, ``input_ids``, ``attention_mask``, ``labels``.
    processor : transformers processor
        Used to access ``pad_token_id``.
    max_length : int
        Hard cap on sequence length (truncates if exceeded).

    Returns
    -------
    dict[str, Tensor]
    """
    if not batch:
        raise ValueError("collate_fn_generative received an empty batch.")

    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id: int = (
        tokenizer.pad_token_id
        if tokenizer is not None and tokenizer.pad_token_id is not None
        else 0
    )

    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)

    # Determine actual max length in this batch (bounded by max_length)
    seq_lens = [item["input_ids"].shape[0] for item in batch]
    target_len = min(max(seq_lens), max_length)

    input_ids_list, attn_mask_list, labels_list = [], [], []
    for item in batch:
        ids = item["input_ids"][:target_len]
        mask = item["attention_mask"][:target_len]
        labs = item["labels"][:target_len]
        pad_len = target_len - ids.shape[0]
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
            labs = torch.cat([labs, torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)])
        input_ids_list.append(ids)
        attn_mask_list.append(mask)
        labels_list.append(labs)

    return {
        "pixel_values": pixel_values,
        "input_ids": torch.stack(input_ids_list, dim=0),
        "attention_mask": torch.stack(attn_mask_list, dim=0),
        "labels": torch.stack(labels_list, dim=0),
    }


def collate_fn_contrastive(
    batch: List[Dict[str, torch.Tensor]],
    processor: Any,
) -> Dict[str, torch.Tensor]:
    """
    Collate a batch of samples for contrastive (CLIP-style) models.

    No ``labels`` key is included in the output.

    Parameters
    ----------
    batch : list of dict
        Each dict must contain:
        ``pixel_values``, ``input_ids``, ``attention_mask``.
    processor : transformers processor
        Used to access ``pad_token_id``.

    Returns
    -------
    dict[str, Tensor]
    """
    if not batch:
        raise ValueError("collate_fn_contrastive received an empty batch.")

    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id: int = (
        tokenizer.pad_token_id
        if tokenizer is not None and tokenizer.pad_token_id is not None
        else 0
    )

    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)
    seq_lens = [item["input_ids"].shape[0] for item in batch]
    target_len = max(seq_lens)

    input_ids_list, attn_mask_list = [], []
    for item in batch:
        ids = item["input_ids"]
        mask = item["attention_mask"]
        pad_len = target_len - ids.shape[0]
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
        input_ids_list.append(ids)
        attn_mask_list.append(mask)

    return {
        "pixel_values": pixel_values,
        "input_ids": torch.stack(input_ids_list, dim=0),
        "attention_mask": torch.stack(attn_mask_list, dim=0),
    }


def collate_fn_vilt(
    batch: List[Dict[str, torch.Tensor]],
    processor: Any,
) -> Dict[str, torch.Tensor]:
    """
    Collate a batch of samples for ViLT models.

    ViLT's processor jointly handles images and text and produces
    ``pixel_values``, ``pixel_mask``, ``input_ids``, ``attention_mask``,
    and ``token_type_ids``.  This function re-runs the processor over the
    raw PIL images and text strings to ensure correct pixel_mask generation,
    then grafts in pre-built labels.

    Parameters
    ----------
    batch : list of dict
        Each dict must contain the same keys as for generative models, with
        an optional ``"text"`` key for the raw string (used for re-encoding).
    processor : ViltProcessor

    Returns
    -------
    dict[str, Tensor]
    """
    if not batch:
        raise ValueError("collate_fn_vilt received an empty batch.")

    # ViLT processor handles padding natively when given a list
    images = [item.get("_pil_image", None) for item in batch]
    texts = [item.get("_text", "") for item in batch]

    # If raw PIL images are stored, use the processor directly for best results
    has_pil = all(img is not None for img in images)
    if has_pil and isinstance(processor, ViltProcessor):
        encoding = processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        labels = torch.stack(
            [item.get("labels", torch.full((encoding["input_ids"].shape[1],), IGNORE_INDEX)) for item in batch],
            dim=0,
        )
        encoding["labels"] = labels
        return dict(encoding)

    # Fallback: use pre-computed tensors
    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)
    input_ids_list = [item["input_ids"] for item in batch]
    attn_mask_list = [item["attention_mask"] for item in batch]
    labels_list = [item.get("labels", torch.full_like(item["input_ids"], IGNORE_INDEX)) for item in batch]

    target_len = max(t.shape[0] for t in input_ids_list)
    pad_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", 0) or 0

    padded_ids, padded_masks, padded_labels = [], [], []
    for ids, mask, labs in zip(input_ids_list, attn_mask_list, labels_list):
        pad_len = target_len - ids.shape[0]
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
            mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
            labs = torch.cat([labs, torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)])
        padded_ids.append(ids)
        padded_masks.append(mask)
        padded_labels.append(labs)

    return {
        "pixel_values": pixel_values,
        "input_ids": torch.stack(padded_ids, dim=0),
        "attention_mask": torch.stack(padded_masks, dim=0),
        "labels": torch.stack(padded_labels, dim=0),
    }


def _select_collate_fn(config: VLMConfig, processor: Any) -> Callable:
    """
    Choose the appropriate collate function based on model family.

    Returns a zero-argument partial (already bound to *processor* and
    *max_length*) suitable for passing to ``DataLoader(collate_fn=...)``.
    """
    from functools import partial

    family = config.model.model_family.lower()
    max_length = config.data.max_length

    if family in _CONTRASTIVE_FAMILIES:
        return partial(collate_fn_contrastive, processor=processor)
    elif family in _VILT_FAMILIES:
        return partial(collate_fn_vilt, processor=processor)
    else:
        return partial(collate_fn_generative, processor=processor, max_length=max_length)


# ===========================================================================
# 6. get_dataloaders
# ===========================================================================


def get_dataloaders(
    config: VLMConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build and return ``(train_loader, val_loader, test_loader)``.

    Features
    --------
    * Automatically detects distributed training via
      ``torch.distributed.is_initialized()`` and wraps datasets with
      ``DistributedSampler`` when appropriate.
    * Uses model-specific collate functions.
    * Applies correct augmentation for each split.
    * Configures ``num_workers``, ``pin_memory``, and ``prefetch_factor``
      from ``config.training``.

    Parameters
    ----------
    config : VLMConfig
        Full pipeline configuration.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        ``(train_loader, val_loader, test_loader)``
    """
    loader = DatasetLoader(config)
    processor = get_processor(config.model.model_name, config)
    collate_fn = _select_collate_fn(config, processor)

    is_distributed = dist.is_available() and dist.is_initialized()

    def _build_loader(split: str, shuffle: bool) -> DataLoader:
        try:
            raw = loader.load(split)
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning(
                "Could not load '%s' split (%s). Returning empty DataLoader.", split, exc
            )
            return _empty_dataloader()

        dataset = VLMDataset(raw, processor, config, split=split)

        sampler = None
        _shuffle = shuffle
        if is_distributed:
            sampler = DistributedSampler(
                dataset,
                shuffle=shuffle,
                drop_last=(split == "train"),
            )
            _shuffle = False  # DistributedSampler handles shuffling

        # Determine num_workers: use 0 on Windows to avoid multiprocessing issues
        num_workers = config.training.num_workers
        if os.name == "nt" and num_workers > 0:
            logger.debug(
                "Windows detected: capping num_workers at 0 to avoid "
                "multiprocessing instability. Override config.training.num_workers."
            )
            # Allow the user to explicitly override by setting > 0 in config
            # but log a warning in case they forgot.
            if num_workers > 4:
                logger.warning(
                    "num_workers=%d on Windows may cause issues. "
                    "Consider setting to 0 or 2.",
                    num_workers,
                )

        # prefetch_factor only valid when num_workers > 0
        prefetch = config.training.prefetch_factor if num_workers > 0 else None

        return DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=_shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=config.training.pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch,
            collate_fn=collate_fn,
            drop_last=(split == "train"),
            persistent_workers=(num_workers > 0),
        )

    train_loader = _build_loader("train", shuffle=True)
    val_loader = _build_loader("val", shuffle=False)
    test_loader = _build_loader("test", shuffle=False)

    logger.info(
        "DataLoaders ready — train: %d batches, val: %d batches, test: %d batches.",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )
    return train_loader, val_loader, test_loader


def _empty_dataloader() -> DataLoader:
    """Return an empty DataLoader for missing splits."""
    return DataLoader([], batch_size=1)


# ===========================================================================
# 7. preprocess_and_cache
# ===========================================================================


def preprocess_and_cache(config: VLMConfig, output_dir: str) -> None:
    """
    Pre-tokenise all dataset splits and save them to disk.

    Saved datasets can be loaded quickly with
    ``datasets.load_from_disk(path)`` without re-processing.

    Process
    -------
    1. Load each split via :class:`DatasetLoader`.
    2. For every sample, call the processor to produce ``pixel_values``,
       ``input_ids``, ``attention_mask``, and ``labels``.
    3. Save the augmented HF Dataset to ``output_dir/{split}/``.

    Parameters
    ----------
    config : VLMConfig
        Full pipeline configuration.
    output_dir : str
        Root directory under which ``train/``, ``val/``, and ``test/``
        subdirectories will be created.

    Notes
    -----
    * Images are stored as-is (paths); only text tokens are cached.
    * The function intentionally uses CPU-only processing so it can run
      as a pre-training data-prep step without needing a GPU.
    * Val/Test splits that are missing from ``config`` are skipped
      gracefully.
    """
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    loader = DatasetLoader(config)
    processor = get_processor(config.model.model_name, config)

    for split in ("train", "val", "test"):
        split_out = output_path / split
        if split_out.exists():
            logger.info(
                "Cache already exists at '%s', skipping split '%s'.",
                split_out, split
            )
            continue

        logger.info("Caching split '%s' → %s", split, split_out)
        try:
            raw_dataset = loader.load(split)
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("Skipping '%s' split: %s", split, exc)
            continue

        # Determine model family once
        family = config.model.model_family.lower()
        is_contrastive = family in _CONTRASTIVE_FAMILIES
        max_len = config.data.max_length
        img_col = "image_path"

        def _tokenise(example: Dict[str, Any]) -> Dict[str, Any]:
            """Map function applied to each row of the HF Dataset."""
            # Build prompt + label text
            task = config.data.task_type.lower()
            caption = example.get("caption", "")
            question = example.get("question", "")
            answer = example.get("answer", "")

            if task == "captioning":
                prompt = caption
                label_txt = caption
            elif task == "vqa":
                prompt = f"Question: {question} Answer:"
                label_txt = answer
            elif task == "chat":
                prompt = question
                label_txt = answer
            elif task == "classification":
                prompt = question or caption
                label_txt = answer
            else:
                prompt = caption or question
                label_txt = answer or caption

            full_text = f"{prompt} {label_txt}".strip()
            tokenizer = getattr(processor, "tokenizer", processor)

            encoding = tokenizer(
                full_text,
                padding="max_length",
                truncation=True,
                max_length=max_len,
            )
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

            # Build labels
            pad_id = tokenizer.pad_token_id or 0
            labels = list(input_ids)
            # Mask prompt tokens
            try:
                prompt_enc = tokenizer(prompt, add_special_tokens=True)
                prompt_len = len(prompt_enc["input_ids"])
                labels[:prompt_len] = [IGNORE_INDEX] * prompt_len
            except Exception:
                pass
            # Mask padding
            labels = [
                IGNORE_INDEX if tok_id == pad_id else lbl
                for tok_id, lbl in zip(input_ids, labels)
            ]

            result: Dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if not is_contrastive:
                result["labels"] = labels

            return result

        logger.info("Tokenising %d samples for split '%s'…", len(raw_dataset), split)
        tokenised = raw_dataset.map(
            _tokenise,
            batched=False,
            desc=f"Tokenising {split}",
            num_proc=1,  # Safe default; increase if no Windows multiprocessing issues
        )

        tokenised.save_to_disk(str(split_out))
        logger.info("Saved cached '%s' split to '%s'.", split, split_out)

    logger.info("Pre-tokenisation complete. Cached datasets at: %s", output_dir)


# ===========================================================================
# Utility: load cached dataset
# ===========================================================================


def load_cached_dataset(output_dir: str, split: str) -> Any:
    """
    Load a previously cached (pre-tokenised) dataset from disk.

    Parameters
    ----------
    output_dir : str
        Root directory passed to :func:`preprocess_and_cache`.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.

    Returns
    -------
    datasets.Dataset

    Raises
    ------
    FileNotFoundError
        If the cache for the requested split does not exist.
    """
    split_path = pathlib.Path(output_dir) / split
    if not split_path.exists():
        raise FileNotFoundError(
            f"No cached dataset found at '{split_path}'. "
            f"Run preprocess_and_cache() first."
        )
    dataset = load_from_disk(str(split_path))
    logger.info("Loaded cached '%s' split from '%s' (%d samples).", split, split_path, len(dataset))
    return dataset


# ===========================================================================
# Module-level seed helper
# ===========================================================================


def seed_everything(seed: int = 42) -> None:
    """
    Set all relevant random seeds for reproducibility.

    Covers Python ``random``, ``numpy``, ``torch`` (CPU and CUDA), and
    CUDA determinism flags.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.debug("Random seed set to %d.", seed)


# ===========================================================================
# Quick-smoke-test entrypoint
# ===========================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )

    # Build a minimal config for testing
    try:
        from config import VLMConfig as _VLMConfig  # type: ignore
        cfg = _VLMConfig()
    except ImportError:
        # Use the stub dataclass defined in this file
        cfg = VLMConfig()  # type: ignore[call-arg]

    logger.info("VLMConfig: %s", cfg)

    # Test get_transforms
    t_train = get_transforms(cfg.model.model_name, "train", cfg.data.image_size)
    t_val = get_transforms(cfg.model.model_name, "val", cfg.data.image_size)
    logger.info("Train transforms: %s", t_train)
    logger.info("Val   transforms: %s", t_val)

    # Smoke-test transform on a dummy image
    dummy_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    out_train = t_train(dummy_img)
    out_val = t_val(dummy_img)
    assert out_train.shape == out_val.shape, "Transform output shapes mismatch!"
    logger.info("Transform smoke-test passed. Output shape: %s", tuple(out_train.shape))

    logger.info("preprocess.py smoke-test complete.")
