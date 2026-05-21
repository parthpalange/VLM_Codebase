"""
train.py — End-to-end VLM training pipeline.

Supports:
  • Full fine-tuning, LoRA, and QLoRA (4-bit / 8-bit)
  • Generative models (BLIP-2, InstructBLIP, LLaVA, LLaVA-NeXT, PaliGemma,
    Idefics2, Florence-2, GIT, BLIP, Qwen2-VL, …) and contrastive models (CLIP, FLAVA)
  • Mixed-precision training (bf16 / fp16 / float32)
  • Gradient accumulation and gradient clipping
  • Cosine LR schedule with linear warm-up
  • TensorBoard + optional Weights & Biases logging
  • Checkpoint saving / resuming

Usage (standalone):
    python train.py --model llava-1.5 --task vqa \\
        --data_dir /data --output_dir /runs/exp1 \\
        --epochs 3 --batch_size 4 --use_lora
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pathlib
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    Blip2ForConditionalGeneration,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
    CLIPModel,
    FlavaModel,
    GitForCausalLM,
    Idefics2ForConditionalGeneration,
    InstructBlipForConditionalGeneration,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
    ViltForQuestionAnswering,
    get_cosine_schedule_with_warmup,
)

# Florence-2 and Qwen2-VL may not be in older transformers versions; guard them.
try:
    from transformers import Florence2ForConditionalGeneration
except ImportError:  # pragma: no cover
    Florence2ForConditionalGeneration = None  # type: ignore

try:
    from transformers import Qwen2VLForConditionalGeneration
except ImportError:  # pragma: no cover
    Qwen2VLForConditionalGeneration = None  # type: ignore

from transformers import BitsAndBytesConfig

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# ---------------------------------------------------------------------------
# Project imports — config.py must live alongside this file.
# ---------------------------------------------------------------------------
from config import (
    CONTRASTIVE_MODELS,
    GENERATIVE_MODELS,
    MODEL_REGISTRY,
    VLMConfig,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional W&B — imported lazily so it remains optional
# ---------------------------------------------------------------------------
_wandb_available = False
try:
    import wandb  # type: ignore

    _wandb_available = True
except ImportError:
    pass


# ===========================================================================
# 1. apply_lora
# ===========================================================================

def apply_lora(model: nn.Module, config: VLMConfig) -> nn.Module:
    """
    Wrap *model* with LoRA / QLoRA adapters defined in *config*.

    Steps
    -----
    1. For QLoRA (4-bit), call ``prepare_model_for_kbit_training`` first so
       that frozen parameters are correctly cast and gradient check-pointing is
       enabled.
    2. Build a ``LoraConfig`` using target modules from MODEL_REGISTRY.
    3. Call ``get_peft_model`` and print trainable parameter statistics.

    Parameters
    ----------
    model:
        The base model (possibly already quantised via BitsAndBytesConfig).
    config:
        Project-wide ``VLMConfig`` instance.

    Returns
    -------
    nn.Module
        The PEFT-wrapped model.
    """
    lora_cfg = config.training.lora
    model_key = config.model_name

    # Determine target modules from registry; fall back to a sensible default.
    registry_entry: Dict[str, Any] = MODEL_REGISTRY.get(model_key, {})
    target_modules: List[str] = registry_entry.get(
        "lora_target_modules",
        ["q_proj", "v_proj"],  # safe universal default
    )

    # Task type: feature extraction for contrastive encoders, causal LM otherwise.
    arch_type: str = registry_entry.get("arch_type", "generative")
    if arch_type == "contrastive":
        task_type = TaskType.FEATURE_EXTRACTION
    else:
        task_type = TaskType.CAUSAL_LM

    # ------------------------------------------------------------------
    # QLoRA: prepare for k-bit training BEFORE creating the LoRA config.
    # ------------------------------------------------------------------
    use_qlora: bool = getattr(config.training, "use_qlora", False)
    if use_qlora:
        logger.info("QLoRA mode: calling prepare_model_for_kbit_training …")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=getattr(
                config.training, "gradient_checkpointing", True
            ),
        )

    lora_config = LoraConfig(
        r=getattr(lora_cfg, "r", 16),
        lora_alpha=getattr(lora_cfg, "lora_alpha", 32),
        target_modules=target_modules,
        lora_dropout=getattr(lora_cfg, "lora_dropout", 0.05),
        bias=getattr(lora_cfg, "bias", "none"),
        task_type=task_type,
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)

    # ---- Print trainable parameter count ---------------------------------
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable_params / all_params if all_params > 0 else 0.0
    logger.info(
        "LoRA applied — trainable params: %s / %s (%.4f %%)",
        f"{trainable_params:,}",
        f"{all_params:,}",
        pct,
    )

    return model


# ===========================================================================
# 2. freeze_backbone
# ===========================================================================

def freeze_backbone(model: nn.Module, config: VLMConfig) -> None:
    """
    Selectively freeze model components.

    Freezing policy (applied in order):
      1. Always freeze the vision encoder (image tower / visual encoder).
      2. Optionally freeze the LLM backbone when ``config.training.freeze_llm``
         is ``True``.
      3. Always leave projection / adapter / cross-attention layers trainable.

    Parameters
    ----------
    model:
        The model whose parameters will be frozen / unfrozen.
    config:
        Project-wide ``VLMConfig`` instance.
    """
    freeze_llm: bool = getattr(config.training, "freeze_llm", False)

    # ------------------------------------------------------------------ #
    # Name patterns that identify each component class                    #
    # ------------------------------------------------------------------ #
    _VISION_PATTERNS = (
        "vision_model",
        "visual_encoder",
        "image_encoder",
        "vision_tower",
        "visual_projection",  # only vision side — NOT language projection
        "patch_embedding",
        "vision_encoder",
    )
    _LLM_PATTERNS = (
        "language_model",
        "text_model",
        "decoder",
        "lm_head",
        "embed_tokens",
    )
    # Patterns that must NEVER be frozen regardless of policy
    _ADAPTER_PATTERNS = (
        "qformer",
        "query_tokens",
        "language_projection",
        "mm_projector",
        "connector",
        "cross_attention",
        "adapter",
        "lora_",
        "ia3_",
        "prompt_",
    )

    frozen_names: List[str] = []
    unfrozen_names: List[str] = []

    for name, param in model.named_parameters():
        name_lower = name.lower()

        # Adapters are always trainable — check first.
        if any(pat in name_lower for pat in _ADAPTER_PATTERNS):
            param.requires_grad_(True)
            unfrozen_names.append(name)
            continue

        # Vision encoder — always frozen.
        if any(pat in name_lower for pat in _VISION_PATTERNS):
            param.requires_grad_(False)
            frozen_names.append(name)
            continue

        # LLM backbone — conditionally frozen.
        if freeze_llm and any(pat in name_lower for pat in _LLM_PATTERNS):
            param.requires_grad_(False)
            frozen_names.append(name)
            continue

        # Everything else is trainable by default.
        param.requires_grad_(True)
        unfrozen_names.append(name)

    logger.info("freeze_backbone: %d params frozen, %d params trainable.",
                len(frozen_names), len(unfrozen_names))

    if frozen_names:
        logger.debug("Frozen parameters (first 10): %s", frozen_names[:10])
    if unfrozen_names:
        logger.debug("Trainable parameters (first 10): %s", unfrozen_names[:10])

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "After freeze_backbone — trainable: %s / %s (%.2f %%)",
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / total if total else 0.0,
    )


# ===========================================================================
# 3. get_model
# ===========================================================================

# Map model class names → constructor callables.
# Florence-2 / Qwen2-VL are guarded above.
_CLASS_MAP: Dict[str, Any] = {
    "Blip2ForConditionalGeneration": Blip2ForConditionalGeneration,
    "InstructBlipForConditionalGeneration": InstructBlipForConditionalGeneration,
    "LlavaForConditionalGeneration": LlavaForConditionalGeneration,
    "LlavaNextForConditionalGeneration": LlavaNextForConditionalGeneration,
    "CLIPModel": CLIPModel,
    "PaliGemmaForConditionalGeneration": PaliGemmaForConditionalGeneration,
    "Idefics2ForConditionalGeneration": Idefics2ForConditionalGeneration,
    "AutoModelForCausalLM": AutoModelForCausalLM,
    "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
    "GitForCausalLM": GitForCausalLM,
    "BlipForConditionalGeneration": BlipForConditionalGeneration,
    "BlipForQuestionAnswering": BlipForQuestionAnswering,
    "ViltForQuestionAnswering": ViltForQuestionAnswering,
    "FlavaModel": FlavaModel,
}
if Florence2ForConditionalGeneration is not None:
    _CLASS_MAP["Florence2ForConditionalGeneration"] = Florence2ForConditionalGeneration
if Qwen2VLForConditionalGeneration is not None:
    _CLASS_MAP["Qwen2VLForConditionalGeneration"] = Qwen2VLForConditionalGeneration


def _build_bnb_config(config: VLMConfig) -> Optional[BitsAndBytesConfig]:
    """
    Build a ``BitsAndBytesConfig`` for 4-bit or 8-bit quantisation.

    Returns ``None`` when quantisation is not requested.
    """
    quant_bits: int = getattr(config.training, "quant_bits", 0)
    use_qlora: bool = getattr(config.training, "use_qlora", False)

    if quant_bits == 4 or use_qlora:
        compute_dtype = (
            torch.bfloat16
            if getattr(config.training, "bf16", False)
            else torch.float16
        )
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    elif quant_bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)

    return None


def _resolve_dtype(config: VLMConfig) -> torch.dtype:
    """Return the ``torch.dtype`` requested in *config*."""
    if getattr(config.training, "bf16", False):
        return torch.bfloat16
    if getattr(config.training, "fp16", False):
        return torch.float16
    return torch.float32


def get_model(config: VLMConfig) -> nn.Module:
    """
    Factory function: load and configure a VLM from HuggingFace Hub.

    Behaviour
    ---------
    1. Resolve the HuggingFace model class from MODEL_REGISTRY (falls back to
       ``AutoModelForCausalLM``).
    2. Build a ``BitsAndBytesConfig`` for 4-bit / 8-bit quantisation if asked.
    3. Load the model with ``from_pretrained``, passing dtype, cache_dir and
       quantisation config.
    4. Optionally call ``freeze_backbone``.
    5. Optionally call ``apply_lora``.

    Parameters
    ----------
    config:
        Project-wide ``VLMConfig`` instance.

    Returns
    -------
    nn.Module
        The fully configured model ready for training.
    """
    model_key: str = config.model_name
    registry_entry: Dict[str, Any] = MODEL_REGISTRY.get(model_key, {})

    hf_model_id: str = registry_entry.get("hf_id", model_key)
    model_class_name: str = registry_entry.get("model_class", "AutoModelForCausalLM")
    model_cls = _CLASS_MAP.get(model_class_name, AutoModelForCausalLM)

    cache_dir: Optional[str] = None
    if hasattr(config, "paths") and hasattr(config.paths, "cache_dir"):
        cache_dir = str(config.paths.cache_dir) if config.paths.cache_dir else None

    dtype = _resolve_dtype(config)
    bnb_config = _build_bnb_config(config)

    # ---- Build from_pretrained kwargs ------------------------------------
    load_kwargs: Dict[str, Any] = {
        "pretrained_model_name_or_path": hf_model_id,
        "cache_dir": cache_dir,
        "trust_remote_code": registry_entry.get("trust_remote_code", False),
    }

    if bnb_config is not None:
        # When using bitsandbytes, dtype is embedded in the BnB config.
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = dtype

    # Some models require additional kwargs stored in the registry.
    extra_kwargs: Dict[str, Any] = registry_entry.get("extra_load_kwargs", {})
    load_kwargs.update(extra_kwargs)

    logger.info(
        "Loading model '%s' (class: %s, hf_id: %s, dtype: %s, quant: %s) …",
        model_key,
        model_class_name,
        hf_model_id,
        dtype,
        "4-bit" if (bnb_config and bnb_config.load_in_4bit) else
        "8-bit" if (bnb_config and bnb_config.load_in_8bit) else "none",
    )

    model: nn.Module = model_cls.from_pretrained(**load_kwargs)

    # ---- Freeze backbone if requested ------------------------------------
    use_freeze: bool = getattr(config.training, "freeze_backbone", False)
    if use_freeze:
        logger.info("Freezing backbone as requested …")
        freeze_backbone(model, config)

    # ---- Apply LoRA / QLoRA if requested ---------------------------------
    use_lora: bool = getattr(config.training, "use_lora", False)
    use_qlora: bool = getattr(config.training, "use_qlora", False)
    if use_lora or use_qlora:
        logger.info("Applying LoRA adapters …")
        model = apply_lora(model, config)

    return model


# ===========================================================================
# 4. evaluate_during_training
# ===========================================================================

@torch.no_grad()
def evaluate_during_training(
    model: nn.Module,
    val_loader: DataLoader,
    processor: Any,
    config: VLMConfig,
    *,
    num_generate_samples: int = 4,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Quick evaluation pass during training.

    Computes average validation loss and generates a handful of sample outputs
    for qualitative inspection. Returns a metrics dict.

    Parameters
    ----------
    model:
        The model being trained.
    val_loader:
        Validation ``DataLoader``.
    processor:
        HuggingFace processor / tokenizer used to decode generated token IDs.
    config:
        Project-wide ``VLMConfig`` instance.
    num_generate_samples:
        Number of batches from which to generate text samples.
    device:
        Target device; defaults to the device of the first model parameter.

    Returns
    -------
    dict
        ``{"val_loss": float, "samples": list[dict]}``
    """
    if device is None:
        device = next(model.parameters()).device

    model_key: str = config.model_name
    is_contrastive: bool = model_key in CONTRASTIVE_MODELS

    model.eval()

    total_loss = 0.0
    num_batches = 0
    samples: List[Dict[str, Any]] = []

    dtype_ctx = (
        torch.bfloat16
        if getattr(config.training, "bf16", False)
        else torch.float16
        if getattr(config.training, "fp16", False)
        else torch.float32
    )
    amp_enabled = dtype_ctx in (torch.bfloat16, torch.float16)

    for batch_idx, batch in enumerate(
        tqdm(val_loader, desc="Evaluating", leave=False, dynamic_ncols=True)
    ):
        batch = _move_batch_to_device(batch, device)

        with autocast(enabled=amp_enabled, dtype=dtype_ctx):
            loss = _compute_loss_impl(model, batch, config, is_contrastive, device)

        if loss is not None:
            total_loss += loss.item()
            num_batches += 1

        # Generate qualitative samples for the first few batches.
        if batch_idx < num_generate_samples and not is_contrastive:
            try:
                gen_out = _generate_samples(model, batch, processor, config, device)
                samples.extend(gen_out)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Sample generation failed (batch %d): %s", batch_idx, exc)

    avg_loss = total_loss / num_batches if num_batches > 0 else float("nan")
    return {"val_loss": avg_loss, "samples": samples}


# ===========================================================================
# 5. Trainer class
# ===========================================================================

class Trainer:
    """
    Full training loop for VLMs.

    Supports:
    - Mixed-precision (fp16 / bf16) via ``torch.cuda.amp.GradScaler``
    - Gradient accumulation
    - Gradient clipping
    - Cosine LR schedule with linear warmup
    - TensorBoard and optional W&B logging
    - Checkpoint save / resume
    - Best-model tracking by validation loss

    Parameters
    ----------
    model:
        Fully configured VLM (possibly LoRA-wrapped).
    train_loader:
        Training ``DataLoader``.
    val_loader:
        Validation ``DataLoader``.
    config:
        Project-wide ``VLMConfig``.
    processor:
        HuggingFace processor / tokenizer (used for sample generation and
        decoding in ``evaluate_during_training``).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: VLMConfig,
        processor: Any,
    ) -> None:
        self.config = config
        self.processor = processor
        self.train_loader = train_loader
        self.val_loader = val_loader

        # ---- Device -------------------------------------------------------
        self.device: torch.device = _resolve_device(config)
        self.model = model.to(self.device) if not _model_has_device_map(model) else model

        # ---- Training meta ------------------------------------------------
        self.model_key: str = config.model_name
        self.is_contrastive: bool = self.model_key in CONTRASTIVE_MODELS

        train_cfg = config.training
        self.num_epochs: int = getattr(train_cfg, "num_epochs", 3)
        self.grad_accum_steps: int = max(1, getattr(train_cfg, "grad_accum_steps", 1))
        self.max_grad_norm: float = getattr(train_cfg, "max_grad_norm", 1.0)
        self.use_fp16: bool = getattr(train_cfg, "fp16", False)
        self.use_bf16: bool = getattr(train_cfg, "bf16", False)

        # AMP — bf16 does not benefit from GradScaler on CUDA; fp16 does.
        self._amp_dtype: torch.dtype = (
            torch.bfloat16 if self.use_bf16
            else torch.float16 if self.use_fp16
            else torch.float32
        )
        self._amp_enabled: bool = self._amp_dtype in (torch.bfloat16, torch.float16)
        self.scaler: Optional[GradScaler] = (
            GradScaler() if (self.use_fp16 and torch.cuda.is_available()) else None
        )

        # ---- Output paths -------------------------------------------------
        self.output_dir = pathlib.Path(
            getattr(getattr(config, "paths", None), "output_dir", "outputs")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # ---- Logging ------------------------------------------------------
        self.tb_writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
        self._wandb_run = None
        if _wandb_available and getattr(train_cfg, "use_wandb", False):
            self._wandb_run = self._init_wandb()

        # ---- State --------------------------------------------------------
        self.global_step: int = 0
        self.best_val_loss: float = float("inf")
        self.start_epoch: int = 0

        # ---- Optimizer / scheduler (built lazily in train()) --------------
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None

        logger.info(
            "Trainer initialised — device: %s, epochs: %d, grad_accum: %d, "
            "amp: %s (%s)",
            self.device,
            self.num_epochs,
            self.grad_accum_steps,
            self._amp_enabled,
            self._amp_dtype,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """
        Execute the full training loop.

        Returns
        -------
        dict
            Aggregated metrics across all epochs:
            ``{"train_losses": [...], "val_losses": [...], "best_val_loss": float}``
        """
        self.setup_optimizer()
        self.setup_scheduler()

        resume_path: Optional[str] = getattr(self.config.training, "resume", None)
        if resume_path and pathlib.Path(resume_path).exists():
            self._load_checkpoint(resume_path)

        all_train_losses: List[float] = []
        all_val_losses: List[float] = []

        logger.info("Starting training for %d epoch(s) …", self.num_epochs)

        for epoch in range(self.start_epoch, self.num_epochs):
            epoch_start = time.time()

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            epoch_secs = time.time() - epoch_start
            train_loss = train_metrics.get("loss", float("nan"))
            val_loss = val_metrics.get("val_loss", float("nan"))

            all_train_losses.append(train_loss)
            all_val_losses.append(val_loss)

            logger.info(
                "Epoch %d/%d — train_loss: %.4f | val_loss: %.4f | %.1f s",
                epoch + 1, self.num_epochs, train_loss, val_loss, epoch_secs,
            )

            # ---- TensorBoard / W&B epoch-level scalars -------------------
            self.tb_writer.add_scalar("epoch/train_loss", train_loss, epoch)
            self.tb_writer.add_scalar("epoch/val_loss", val_loss, epoch)
            if self._wandb_run is not None:
                self._wandb_run.log(
                    {"epoch/train_loss": train_loss, "epoch/val_loss": val_loss},
                    step=self.global_step,
                )

            # ---- Checkpoint ----------------------------------------------
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            self._save_checkpoint(epoch, val_loss, is_best=is_best)

        self.tb_writer.close()
        if self._wandb_run is not None:
            self._wandb_run.finish()

        result = {
            "train_losses": all_train_losses,
            "val_losses": all_val_losses,
            "best_val_loss": self.best_val_loss,
        }
        self._save_metrics(result)
        logger.info("Training complete. Best val_loss: %.4f", self.best_val_loss)
        return result

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Run a single training epoch.

        Parameters
        ----------
        epoch:
            Zero-based epoch index.

        Returns
        -------
        dict
            ``{"loss": float, "lr": float}``
        """
        self.model.train()

        running_loss = 0.0
        num_steps = 0

        progress = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch + 1}/{self.num_epochs}",
            dynamic_ncols=True,
        )

        self.optimizer.zero_grad()

        for step, batch in progress:
            batch = _move_batch_to_device(batch, self.device)

            # ---- Forward pass ------------------------------------------
            amp_ctx = autocast(
                enabled=self._amp_enabled, dtype=self._amp_dtype
            ) if self._amp_enabled else nullcontext()

            with amp_ctx:
                loss = self._compute_loss(batch)
                # Scale for gradient accumulation
                loss = loss / self.grad_accum_steps

            # ---- Backward pass -----------------------------------------
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * self.grad_accum_steps
            num_steps += 1

            # ---- Optimizer step (every grad_accum_steps mini-batches) ---
            is_accum_step = (step + 1) % self.grad_accum_steps == 0
            is_last_step = (step + 1) == len(self.train_loader)

            if is_accum_step or is_last_step:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.optimizer.zero_grad()
                self.global_step += 1

                current_lr = self.optimizer.param_groups[0]["lr"]
                smooth_loss = running_loss / num_steps

                # ---- Step-level logging ---------------------------------
                self.tb_writer.add_scalar("train/loss", smooth_loss, self.global_step)
                self.tb_writer.add_scalar("train/lr", current_lr, self.global_step)
                if self._wandb_run is not None:
                    self._wandb_run.log(
                        {"train/loss": smooth_loss, "train/lr": current_lr},
                        step=self.global_step,
                    )

                progress.set_postfix(loss=f"{smooth_loss:.4f}", lr=f"{current_lr:.2e}")

        avg_loss = running_loss / num_steps if num_steps > 0 else float("nan")
        current_lr = self.optimizer.param_groups[0]["lr"]
        return {"loss": avg_loss, "lr": current_lr}

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, Any]:
        """
        Run the full validation loop.

        Parameters
        ----------
        epoch:
            Zero-based epoch index (used for logging).

        Returns
        -------
        dict
            ``{"val_loss": float, "samples": list}``
        """
        metrics = evaluate_during_training(
            self.model,
            self.val_loader,
            self.processor,
            self.config,
            device=self.device,
        )
        val_loss = metrics.get("val_loss", float("nan"))

        self.tb_writer.add_scalar("val/loss", val_loss, epoch)
        if self._wandb_run is not None:
            self._wandb_run.log({"val/loss": val_loss}, step=self.global_step)

        # Log sample generations to TensorBoard as text.
        for idx, sample in enumerate(metrics.get("samples", [])[:4]):
            if "generated" in sample:
                self.tb_writer.add_text(
                    f"val/sample_{idx}",
                    f"**GT:** {sample.get('reference', 'N/A')}\n\n"
                    f"**Pred:** {sample['generated']}",
                    epoch,
                )

        self.model.train()
        return metrics

    def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute the training loss for one mini-batch.

        Delegates to :func:`_compute_loss_impl` with the current model.

        Parameters
        ----------
        batch:
            A dict of tensors already on ``self.device``.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor (with gradient graph).
        """
        loss = _compute_loss_impl(
            self.model, batch, self.config, self.is_contrastive, self.device
        )
        if loss is None:
            raise RuntimeError(
                "Loss is None — check that the batch contains 'labels' or "
                "'input_ids' and that the model outputs a 'loss' field."
            )
        return loss

    def _save_checkpoint(
        self,
        epoch: int,
        metric: float,
        is_best: bool = False,
    ) -> None:
        """
        Persist training state to disk.

        Saves model weights (full state_dict or adapter weights for PEFT
        models), optimizer state, scheduler state, and metadata.

        Parameters
        ----------
        epoch:
            Current epoch (0-indexed).
        metric:
            Validation metric value to store alongside the checkpoint.
        is_best:
            If ``True``, also copies checkpoint to ``best_model/``.
        """
        ckpt_path = self.checkpoint_dir / f"epoch_{epoch + 1:03d}"
        ckpt_path.mkdir(parents=True, exist_ok=True)

        # Save model (PEFT models expose save_pretrained).
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(str(ckpt_path))
        else:
            torch.save(self.model.state_dict(), str(ckpt_path / "model.pt"))

        # Save optimiser + scheduler state.
        extra = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "metric": metric,
            "optimizer": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            extra["scheduler"] = self.scheduler.state_dict()
        if self.scaler is not None:
            extra["scaler"] = self.scaler.state_dict()

        torch.save(extra, str(ckpt_path / "training_state.pt"))

        # Write human-readable metadata.
        meta = {
            "epoch": epoch + 1,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "val_loss": metric,
            "model_key": self.model_key,
        }
        with open(ckpt_path / "meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)

        logger.info("Checkpoint saved → %s (val_loss=%.4f)", ckpt_path, metric)

        if is_best:
            best_path = self.output_dir / "best_model"
            best_path.mkdir(parents=True, exist_ok=True)
            if hasattr(self.model, "save_pretrained"):
                self.model.save_pretrained(str(best_path))
            else:
                torch.save(
                    self.model.state_dict(), str(best_path / "model.pt")
                )
            with open(best_path / "meta.json", "w") as fh:
                json.dump(meta, fh, indent=2)
            logger.info("Best model updated → %s", best_path)

    def _load_checkpoint(self, path: str) -> None:
        """
        Restore training state from a checkpoint directory or file.

        Loads model weights, optimizer, scheduler, and scaler states.
        Also restores ``global_step``, ``start_epoch``, and ``best_val_loss``.

        Parameters
        ----------
        path:
            Path to a checkpoint directory produced by ``_save_checkpoint``,
            or path to a ``training_state.pt`` file.
        """
        ckpt_path = pathlib.Path(path)
        if ckpt_path.is_dir():
            state_file = ckpt_path / "training_state.pt"
        else:
            state_file = ckpt_path

        if not state_file.exists():
            logger.warning("No training_state.pt found at %s — skipping resume.", path)
            return

        logger.info("Resuming from checkpoint: %s", state_file)
        state = torch.load(str(state_file), map_location=self.device)

        # ---- Model weights -----------------------------------------------
        model_dir = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent
        model_pt = model_dir / "model.pt"
        if model_pt.exists():
            self.model.load_state_dict(torch.load(str(model_pt), map_location=self.device))
        elif (model_dir / "adapter_config.json").exists():
            # PEFT model: load_pretrained handled externally; just warn.
            logger.info(
                "PEFT adapter weights detected in %s. "
                "Make sure to reload via from_pretrained if needed.",
                model_dir,
            )

        # ---- Optimizer / scheduler / scaler ------------------------------
        if self.optimizer is not None and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
        if self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])

        # ---- Training state ----------------------------------------------
        self.global_step = state.get("global_step", 0)
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        self.start_epoch = state.get("epoch", 0) + 1  # resume at next epoch

        logger.info(
            "Resumed: epoch=%d, global_step=%d, best_val_loss=%.4f",
            self.start_epoch,
            self.global_step,
            self.best_val_loss,
        )

    def setup_optimizer(self) -> None:
        """
        Build the AdamW optimiser with parameter-group-specific learning rates.

        Parameter groups
        ----------------
        - **Backbone** (vision encoder / LLM): lower LR
          (``config.training.backbone_lr``, default 1e-5).
        - **Adapter / projection** layers: standard LR
          (``config.training.lr``, default 2e-4).
        - Weight decay is applied only to non-bias, non-norm parameters.
        """
        train_cfg = self.config.training
        base_lr: float = getattr(train_cfg, "lr", 2e-4)
        backbone_lr: float = getattr(train_cfg, "backbone_lr", base_lr * 0.1)
        weight_decay: float = getattr(train_cfg, "weight_decay", 0.01)

        _ADAPTER_PATTERNS = (
            "qformer", "query_tokens", "language_projection",
            "mm_projector", "connector", "cross_attention",
            "adapter", "lora_", "ia3_", "prompt_",
        )
        _NO_DECAY_PATTERNS = ("bias", "layer_norm", "layernorm", "norm")

        adapter_decay, adapter_no_decay = [], []
        backbone_decay, backbone_no_decay = [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            name_lower = name.lower()
            is_adapter = any(p in name_lower for p in _ADAPTER_PATTERNS)
            no_decay = any(p in name_lower for p in _NO_DECAY_PATTERNS) or param.ndim == 1

            if is_adapter:
                (adapter_no_decay if no_decay else adapter_decay).append(param)
            else:
                (backbone_no_decay if no_decay else backbone_decay).append(param)

        param_groups = []
        if adapter_decay:
            param_groups.append({"params": adapter_decay, "lr": base_lr, "weight_decay": weight_decay})
        if adapter_no_decay:
            param_groups.append({"params": adapter_no_decay, "lr": base_lr, "weight_decay": 0.0})
        if backbone_decay:
            param_groups.append({"params": backbone_decay, "lr": backbone_lr, "weight_decay": weight_decay})
        if backbone_no_decay:
            param_groups.append({"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0})

        if not param_groups:
            raise ValueError(
                "No trainable parameters found. Check freeze_backbone settings."
            )

        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(getattr(train_cfg, "adam_beta1", 0.9), getattr(train_cfg, "adam_beta2", 0.999)),
            eps=getattr(train_cfg, "adam_eps", 1e-8),
        )
        logger.info(
            "Optimizer: AdamW | base_lr=%.2e | backbone_lr=%.2e | "
            "weight_decay=%.4f | param_groups=%d",
            base_lr, backbone_lr, weight_decay, len(param_groups),
        )

    def setup_scheduler(self) -> None:
        """
        Build a cosine LR scheduler with a linear warm-up phase.

        The total number of optimiser steps is:
        ``num_epochs × (len(train_loader) // grad_accum_steps)``

        Warm-up steps default to 5 % of total steps unless overridden by
        ``config.training.warmup_steps``.
        """
        train_cfg = self.config.training
        steps_per_epoch = math.ceil(len(self.train_loader) / self.grad_accum_steps)
        total_steps = self.num_epochs * steps_per_epoch
        warmup_steps = getattr(
            train_cfg, "warmup_steps", max(1, int(0.05 * total_steps))
        )

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        logger.info(
            "Scheduler: cosine with warmup | total_steps=%d | warmup_steps=%d",
            total_steps,
            warmup_steps,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_wandb(self) -> Any:
        """Initialise a W&B run and return the run object."""
        run_name = getattr(self.config.training, "run_name", None)
        project = getattr(self.config.training, "wandb_project", "vlm-training")
        cfg_dict: Dict[str, Any] = {}
        if hasattr(self.config, "__dict__"):
            cfg_dict = _nested_dict(self.config)

        run = wandb.init(project=project, name=run_name, config=cfg_dict)
        wandb.watch(self.model, log="gradients", log_freq=100)
        logger.info("W&B run initialised: %s", run.url)
        return run

    def _save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Persist aggregated training metrics as JSON."""
        out_file = self.output_dir / "training_metrics.json"
        with open(out_file, "w") as fh:
            json.dump(
                {k: (v if not isinstance(v, float) else round(v, 6)) for k, v in metrics.items()},
                fh, indent=2,
            )
        logger.info("Training metrics saved → %s", out_file)


# ===========================================================================
# 6. Module-level loss helper (used by both Trainer and evaluate_during_training)
# ===========================================================================

def _compute_loss_impl(
    model: nn.Module,
    batch: Dict[str, Any],
    config: VLMConfig,
    is_contrastive: bool,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Compute the loss for a single mini-batch.

    Generative models
    -----------------
    Call ``model(**batch)`` and return ``outputs.loss``.  Most HuggingFace
    generative models compute cross-entropy over the shifted labels internally
    when ``labels`` are supplied in the batch.

    Contrastive models (CLIP, FLAVA)
    --------------------------------
    Call ``model(**batch)`` and compute symmetric cross-entropy (NT-Xent)
    between ``image_embeds`` and ``text_embeds`` (normalised).

    Parameters
    ----------
    model:
        The model to forward through.
    batch:
        Dict of tensors on the correct device.
    config:
        Project-wide config (used for model_key lookup).
    is_contrastive:
        ``True`` for CLIP-style models.
    device:
        Target device.

    Returns
    -------
    Optional[torch.Tensor]
        Scalar loss, or ``None`` if the model returned no loss.
    """
    if is_contrastive:
        return _contrastive_loss(model, batch)
    else:
        return _generative_loss(model, batch)


def _generative_loss(
    model: nn.Module,
    batch: Dict[str, Any],
) -> Optional[torch.Tensor]:
    """
    Forward pass for generative VLMs.

    Relies on the model computing cross-entropy internally when ``labels``
    are present in the batch.
    """
    # Remove keys that the model doesn't accept (e.g. raw strings).
    safe_batch = {
        k: v for k, v in batch.items()
        if isinstance(v, torch.Tensor)
    }
    outputs = model(**safe_batch)

    if hasattr(outputs, "loss") and outputs.loss is not None:
        return outputs.loss

    # Fallback: manual cross-entropy over logits if model didn't compute loss.
    if hasattr(outputs, "logits") and "labels" in batch:
        logits: torch.Tensor = outputs.logits  # (B, T, vocab)
        labels: torch.Tensor = batch["labels"]  # (B, T)
        # Shift for causal LM.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss

    return None


def _contrastive_loss(
    model: nn.Module,
    batch: Dict[str, Any],
) -> Optional[torch.Tensor]:
    """
    Symmetric NT-Xent (CLIP-style) loss.

    Extracts ``image_embeds`` and ``text_embeds`` from model outputs,
    L2-normalises them and computes bidirectional cross-entropy.
    """
    safe_batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
    outputs = model(**safe_batch)

    # CLIP / FLAVA already return loss when return_loss=True.
    if hasattr(outputs, "loss") and outputs.loss is not None:
        return outputs.loss

    # Manual symmetric loss from embeddings.
    img_emb: Optional[torch.Tensor] = getattr(outputs, "image_embeds", None)
    txt_emb: Optional[torch.Tensor] = getattr(outputs, "text_embeds", None)

    if img_emb is None or txt_emb is None:
        return None

    img_emb = nn.functional.normalize(img_emb, dim=-1)
    txt_emb = nn.functional.normalize(txt_emb, dim=-1)

    logit_scale = getattr(model, "logit_scale", None)
    temperature = logit_scale.exp() if logit_scale is not None else torch.tensor(1.0)

    logits_per_image = temperature * img_emb @ txt_emb.t()  # (B, B)
    logits_per_text = logits_per_image.t()

    batch_size = img_emb.size(0)
    labels = torch.arange(batch_size, device=img_emb.device)

    loss_i = nn.functional.cross_entropy(logits_per_image, labels)
    loss_t = nn.functional.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2.0


# ===========================================================================
# 7. Utilities
# ===========================================================================

def _move_batch_to_device(
    batch: Dict[str, Any], device: torch.device
) -> Dict[str, Any]:
    """Recursively move all tensors in *batch* to *device*."""
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _resolve_device(config: VLMConfig) -> torch.device:
    """Return the preferred ``torch.device`` from config or auto-detect."""
    device_str: str = getattr(config.training, "device", "auto")
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _model_has_device_map(model: nn.Module) -> bool:
    """Return ``True`` if the model was loaded with ``device_map='auto'``."""
    return getattr(model, "hf_device_map", None) is not None


def _nested_dict(obj: Any) -> Any:
    """Recursively convert a config object (dataclass/namespace) to a dict."""
    if hasattr(obj, "__dict__"):
        return {k: _nested_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_nested_dict(i) for i in obj]
    return obj


def _generate_samples(
    model: nn.Module,
    batch: Dict[str, Any],
    processor: Any,
    config: VLMConfig,
    device: torch.device,
    *,
    max_new_tokens: int = 64,
) -> List[Dict[str, str]]:
    """
    Generate text for a single validation batch and return decoded strings.

    Parameters
    ----------
    model:
        The model (in eval mode, no grad).
    batch:
        Dict of tensors on *device*.
    processor:
        HuggingFace processor / tokenizer for decoding.
    config:
        Project config.
    device:
        Inference device.
    max_new_tokens:
        Token budget for each generated sequence.

    Returns
    -------
    list[dict]
        Each element has keys ``"generated"`` and optionally ``"reference"``.
    """
    gen_cfg = getattr(config.training, "generation", None)
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": getattr(gen_cfg, "max_new_tokens", max_new_tokens),
        "do_sample": getattr(gen_cfg, "do_sample", False),
        "num_beams": getattr(gen_cfg, "num_beams", 1),
    }

    # Build input dict — exclude labels and non-tensor keys.
    input_keys = {"input_ids", "attention_mask", "pixel_values", "pixel_values_videos",
                  "image_grid_thw", "pixel_values_images"}
    gen_input = {k: v for k, v in batch.items() if k in input_keys and isinstance(v, torch.Tensor)}

    generated_ids = model.generate(**gen_input, **gen_kwargs)
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)

    results: List[Dict[str, str]] = []
    ref_ids = batch.get("labels")
    if ref_ids is not None:
        # Replace -100 (ignore index) with pad token id.
        pad_id = getattr(processor, "pad_token_id", 0) or 0
        ref_ids = ref_ids.clone()
        ref_ids[ref_ids == -100] = pad_id
        references = processor.batch_decode(ref_ids, skip_special_tokens=True)
    else:
        references = [None] * len(decoded)

    for gen, ref in zip(decoded, references):
        entry: Dict[str, str] = {"generated": gen}
        if ref is not None:
            entry["reference"] = ref
        results.append(entry)

    return results


# ===========================================================================
# 8. __main__ — standalone entry point
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VLM Training Pipeline — standalone entry point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model / task
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model key as defined in MODEL_REGISTRY (e.g. 'llava-1.5', 'blip2').",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="vqa",
        choices=["vqa", "captioning", "classification", "retrieval", "grounding"],
        help="Downstream task type.",
    )

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory of the dataset (must contain train/ and val/ splits).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory for checkpoints, logs, and metrics.",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate.")
    parser.add_argument("--backbone_lr", type=float, default=None,
                        help="LR for backbone params (defaults to 10 %% of --lr).")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="Number of LR warm-up steps (default: 5 %% of total).")

    # Precision
    parser.add_argument("--fp16", action="store_true", help="Use float16 mixed precision.")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision.")

    # LoRA / QLoRA
    parser.add_argument("--use_lora", action="store_true", help="Apply LoRA adapters.")
    parser.add_argument("--use_qlora", action="store_true",
                        help="Apply QLoRA (4-bit quantisation + LoRA). Implies --use_lora.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling.")
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Freezing
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze vision encoder and optionally the LLM.")
    parser.add_argument("--freeze_llm", action="store_true",
                        help="Also freeze LLM backbone (only projection layers train).")

    # Misc
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HuggingFace cache directory.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint directory to resume from.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="vlm-training")

    return parser.parse_args()


def _build_config_from_args(args: argparse.Namespace) -> VLMConfig:
    """
    Construct a ``VLMConfig`` from parsed CLI arguments.

    ``VLMConfig`` is expected to be a dataclass / namespace-style class
    defined in ``config.py``.  We build it by setting attributes so this
    function remains forward-compatible with the actual implementation.
    """
    config = VLMConfig()
    config.model_name = args.model
    config.task = args.task

    # -- Paths ----------------------------------------------------------------
    if not hasattr(config, "paths") or config.paths is None:
        config.paths = type("Paths", (), {})()
    config.paths.data_dir = args.data_dir
    config.paths.output_dir = args.output_dir
    config.paths.cache_dir = args.cache_dir

    # -- Training -------------------------------------------------------------
    if not hasattr(config, "training") or config.training is None:
        config.training = type("Training", (), {})()

    t = config.training
    t.num_epochs = args.epochs
    t.batch_size = args.batch_size
    t.lr = args.lr
    t.backbone_lr = args.backbone_lr if args.backbone_lr else args.lr * 0.1
    t.weight_decay = args.weight_decay
    t.max_grad_norm = args.max_grad_norm
    t.grad_accum_steps = args.grad_accum
    t.warmup_steps = args.warmup_steps  # None → auto 5 %
    t.fp16 = args.fp16
    t.bf16 = args.bf16
    t.use_lora = args.use_lora or args.use_qlora
    t.use_qlora = args.use_qlora
    t.quant_bits = 4 if args.use_qlora else 0
    t.freeze_backbone = args.freeze_backbone
    t.freeze_llm = args.freeze_llm
    t.resume = args.resume
    t.use_wandb = args.use_wandb
    t.wandb_project = args.wandb_project
    t.device = "auto"

    # -- LoRA -----------------------------------------------------------------
    if not hasattr(t, "lora") or t.lora is None:
        t.lora = type("LoRA", (), {})()
    t.lora.r = args.lora_r
    t.lora.lora_alpha = args.lora_alpha
    t.lora.lora_dropout = args.lora_dropout
    t.lora.bias = "none"

    return config


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = _parse_args()

    # ---- Logging level -------------------------------------------------------
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # ---- Reproducibility -----------------------------------------------------
    _set_seed(args.seed)

    # ---- Build config --------------------------------------------------------
    config = _build_config_from_args(args)

    logger.info("=" * 70)
    logger.info("VLM Training Pipeline")
    logger.info("  Model   : %s", config.model_name)
    logger.info("  Task    : %s", config.task)
    logger.info("  Epochs  : %d", config.training.num_epochs)
    logger.info("  LoRA    : %s | QLoRA: %s", config.training.use_lora, config.training.use_qlora)
    logger.info("  Output  : %s", config.paths.output_dir)
    logger.info("=" * 70)

    # ---- Load processor & model ---------------------------------------------
    from transformers import AutoProcessor

    hf_model_id: str = MODEL_REGISTRY.get(config.model_name, {}).get("hf_id", config.model_name)
    cache_dir: Optional[str] = getattr(config.paths, "cache_dir", None)

    logger.info("Loading processor for '%s' …", hf_model_id)
    processor = AutoProcessor.from_pretrained(
        hf_model_id,
        cache_dir=cache_dir,
        trust_remote_code=MODEL_REGISTRY.get(config.model_name, {}).get("trust_remote_code", False),
    )

    logger.info("Loading model …")
    model = get_model(config)

    # ---- Build DataLoaders ---------------------------------------------------
    # NOTE: DataLoaders are built externally (e.g. from dataset.py / preprocess.py)
    # and passed to Trainer.  In standalone mode we construct a minimal example
    # dataset that wraps HuggingFace datasets so the pipeline can be tested
    # end-to-end without importing preprocess.py.

    from datasets import load_dataset  # type: ignore

    data_dir = pathlib.Path(args.data_dir)
    if (data_dir / "dataset_dict.json").exists() or (data_dir / "train").exists():
        # Locally saved HuggingFace dataset.
        raw_dataset = load_dataset(str(data_dir))
    else:
        raise FileNotFoundError(
            f"Dataset not found at '{data_dir}'. "
            "Pass a directory containing a saved HuggingFace DatasetDict "
            "(with 'train' and 'validation' splits)."
        )

    def _collate_fn(examples):
        """Minimal collator — delegates to the processor."""
        images = [ex["image"] for ex in examples if "image" in ex]
        texts = [ex.get("question", ex.get("text", "")) for ex in examples]
        labels = [ex.get("label", ex.get("answer", "")) for ex in examples]

        encoding = processor(
            images=images if images else None,
            text=texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if labels:
            label_enc = processor.tokenizer(
                labels,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            lids = label_enc["input_ids"].clone()
            lids[lids == processor.tokenizer.pad_token_id] = -100
            encoding["labels"] = lids
        return dict(encoding)

    train_split = raw_dataset.get("train", raw_dataset.get("train"))
    val_split = raw_dataset.get("validation", raw_dataset.get("test", None))

    if train_split is None:
        raise ValueError("Dataset must contain a 'train' split.")
    if val_split is None:
        logger.warning("No 'validation'/'test' split found — using 10 %% of train for val.")
        split = train_split.train_test_split(test_size=0.1, seed=args.seed)
        train_split, val_split = split["train"], split["test"]

    train_loader = DataLoader(
        train_split,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_split,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        "DataLoaders ready — train: %d batches | val: %d batches",
        len(train_loader), len(val_loader),
    )

    # ---- Train ---------------------------------------------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        processor=processor,
    )
    metrics = trainer.train()

    logger.info("Final metrics: %s", json.dumps(
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()
         if not isinstance(v, list)},
        indent=2,
    ))
