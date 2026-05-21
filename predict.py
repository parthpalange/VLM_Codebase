"""
predict.py
==========
End-to-end inference pipeline for Vision-Language Models (VLMs).

Supports:
  * Generative models  : BLIP-2, LLaVA, InstructBLIP, PaliGemma, Idefics, etc.
  * Contrastive models : CLIP, SigLIP, ALIGN, etc.

Usage examples
--------------
# Single image captioning
python predict.py --model Salesforce/blip2-opt-2.7b --image cat.jpg --task captioning

# VQA
python predict.py --model llava-hf/llava-1.5-7b-hf --image dog.jpg \
    --prompt "What breed is this dog?" --task vqa

# Batch inference from JSONL
python predict.py --model Salesforce/blip2-opt-2.7b --mode batch \
    --input_file inputs.jsonl --output_file predictions.jsonl

# Zero-shot classification with CLIP
python predict.py --model openai/clip-vit-base-patch32 --image cat.jpg \
    --mode classify --class_labels "cat,dog,bird,fish"

# Interactive chat
python predict.py --model llava-hf/llava-1.5-7b-hf --mode interactive
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import requests
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    CLIPModel,
    CLIPProcessor,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
    TextIteratorStreamer,
    logging as hf_logging,
)
from threading import Thread

# ---------------------------------------------------------------------------
# Attempt to import from config.py (sibling module).  If config.py does not
# yet exist, sensible fallbacks are defined below so this file is always
# runnable as a standalone script.
# ---------------------------------------------------------------------------
try:
    from config import (  # type: ignore
        MODEL_REGISTRY,
        VLMConfig,
        PROMPT_TEMPLATES,
        CONTRASTIVE_MODELS,
        GENERATIVE_MODELS,
    )
except ImportError:  # pragma: no cover — config.py not present yet
    logging.getLogger(__name__).warning(
        "config.py not found — using built-in fallback definitions."
    )

    # ---- Minimal fallback VLMConfig ----------------------------------------
    @dataclass
    class VLMConfig:  # type: ignore[no-redef]
        model_name: str = ""
        model_type: str = "generative"       # "generative" | "contrastive"
        arch: str = "auto"                   # blip2 | llava | llava_next | instructblip
                                             # paligemma | clip | auto
        use_flash_attention: bool = False
        trust_remote_code: bool = False
        max_new_tokens: int = 256

    # ---- Minimal fallback registries ----------------------------------------
    MODEL_REGISTRY: Dict[str, VLMConfig] = {  # type: ignore[misc]
        "Salesforce/blip2-opt-2.7b": VLMConfig(
            model_name="Salesforce/blip2-opt-2.7b",
            arch="blip2",
        ),
        "Salesforce/blip2-flan-t5-xl": VLMConfig(
            model_name="Salesforce/blip2-flan-t5-xl",
            arch="blip2",
        ),
        "llava-hf/llava-1.5-7b-hf": VLMConfig(
            model_name="llava-hf/llava-1.5-7b-hf",
            arch="llava",
        ),
        "llava-hf/llava-v1.6-mistral-7b-hf": VLMConfig(
            model_name="llava-hf/llava-v1.6-mistral-7b-hf",
            arch="llava_next",
        ),
        "Salesforce/instructblip-vicuna-7b": VLMConfig(
            model_name="Salesforce/instructblip-vicuna-7b",
            arch="instructblip",
        ),
        "google/paligemma-3b-pt-224": VLMConfig(
            model_name="google/paligemma-3b-pt-224",
            arch="paligemma",
        ),
        "openai/clip-vit-base-patch32": VLMConfig(
            model_name="openai/clip-vit-base-patch32",
            model_type="contrastive",
            arch="clip",
        ),
        "openai/clip-vit-large-patch14": VLMConfig(
            model_name="openai/clip-vit-large-patch14",
            model_type="contrastive",
            arch="clip",
        ),
    }

    PROMPT_TEMPLATES: Dict[str, Dict[str, str]] = {  # type: ignore[misc]
        "blip2": {
            "captioning": "a photo of",
            "vqa": "Question: {prompt} Answer:",
            "chat": "Question: {prompt} Answer:",
            "classify": "A photo of a {prompt}.",
        },
        "llava": {
            "captioning": "USER: <image>\nDescribe this image in detail.\nASSISTANT:",
            "vqa": "USER: <image>\n{prompt}\nASSISTANT:",
            "chat": "USER: <image>\n{prompt}\nASSISTANT:",
            "classify": "USER: <image>\nIs this a photo of {prompt}? Answer yes or no.\nASSISTANT:",
        },
        "llava_next": {
            "captioning": "[INST] <image>\nDescribe this image in detail. [/INST]",
            "vqa": "[INST] <image>\n{prompt} [/INST]",
            "chat": "[INST] <image>\n{prompt} [/INST]",
            "classify": "[INST] <image>\nIs this a photo of {prompt}? [/INST]",
        },
        "instructblip": {
            "captioning": "Describe this image in detail.",
            "vqa": "{prompt}",
            "chat": "{prompt}",
            "classify": "Is this a photo of {prompt}?",
        },
        "paligemma": {
            "captioning": "caption en",
            "vqa": "{prompt}",
            "chat": "{prompt}",
            "classify": "answer en {prompt}",
        },
        "auto": {
            "captioning": "{prompt}" if "{prompt}" else "Describe this image.",
            "vqa": "{prompt}",
            "chat": "{prompt}",
            "classify": "{prompt}",
        },
    }

    CONTRASTIVE_MODELS: List[str] = ["clip", "siglip", "align"]  # type: ignore[misc]
    GENERATIVE_MODELS: List[str] = [                              # type: ignore[misc]
        "blip2", "llava", "llava_next", "instructblip", "paligemma", "auto"
    ]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
hf_logging.set_verbosity_error()  # suppress noisy HF progress bars


# ---------------------------------------------------------------------------
# GenerationConfig dataclass
# ---------------------------------------------------------------------------
@dataclass
class GenerationConfig:
    """Hyperparameters controlling text generation for generative VLMs.

    All fields map 1-to-1 to :class:`transformers.GenerationConfig` kwargs
    and are passed directly to ``model.generate()``.
    """

    max_new_tokens: int = 256
    min_new_tokens: int = 1
    num_beams: int = 1
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    early_stopping: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict suitable for ``**kwargs`` unpacking."""
        return asdict(self)

    def merge(self, overrides: Dict[str, Any]) -> "GenerationConfig":
        """Return a *new* GenerationConfig with ``overrides`` applied."""
        base = self.to_dict()
        base.update({k: v for k, v in overrides.items() if k in base})
        return GenerationConfig(**base)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _auto_device(requested: str = "auto") -> torch.device:
    """Resolve the target device from a user-supplied string."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _auto_dtype(device: torch.device) -> torch.dtype:
    """Choose the best floating-point dtype for the given device."""
    if device.type == "cuda":
        # bf16 preferred on Ampere+ (sm_80); fall back to fp16
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    # MPS and CPU: use fp32 (MPS bf16 support is still partial)
    return torch.float32


def _arch_from_model_name(model_name: str, cfg: Optional[VLMConfig]) -> str:
    """Infer the model architecture string from config or model name heuristics."""
    if cfg is not None and cfg.arch not in ("auto", ""):
        return cfg.arch
    name_lower = model_name.lower()
    for arch_key in ("blip2", "blip-2", "blip_2"):
        if arch_key.replace("-", "").replace("_", "") in name_lower.replace("-", "").replace("_", ""):
            return "blip2"
    if "llava-next" in name_lower or "llavanext" in name_lower or "llava_next" in name_lower:
        return "llava_next"
    if "llava" in name_lower:
        return "llava"
    if "instructblip" in name_lower:
        return "instructblip"
    if "paligemma" in name_lower:
        return "paligemma"
    if "clip" in name_lower:
        return "clip"
    if "siglip" in name_lower:
        return "siglip"
    if "align" in name_lower:
        return "align"
    return "auto"


def _is_contrastive(arch: str) -> bool:
    return arch in CONTRASTIVE_MODELS


# ---------------------------------------------------------------------------
# VLMPredictor
# ---------------------------------------------------------------------------

class VLMPredictor:
    """Unified inference interface for generative and contrastive VLMs.

    Parameters
    ----------
    model_name_or_path:
        Hugging Face model ID (e.g. ``"Salesforce/blip2-opt-2.7b"``) **or**
        an absolute / relative path to a local checkpoint directory.
    config:
        Optional :class:`VLMConfig` instance.  When *None* the registry is
        consulted; if not found, sensible defaults are used.
    device:
        ``'auto'`` (default), ``'cuda'``, ``'cpu'``, or ``'mps'``.

    Examples
    --------
    >>> predictor = VLMPredictor("Salesforce/blip2-opt-2.7b")
    >>> predictor.predict_single("image.jpg", task="captioning")
    'a cat sitting on a windowsill'
    """

    def __init__(
        self,
        model_name_or_path: str,
        config: Optional[VLMConfig] = None,
        device: str = "auto",
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.config: VLMConfig = config or MODEL_REGISTRY.get(
            model_name_or_path, VLMConfig(model_name=model_name_or_path)
        )

        self.device: torch.device = _auto_device(device)
        self.dtype: torch.dtype = _auto_dtype(self.device)
        self.arch: str = _arch_from_model_name(model_name_or_path, self.config)
        self._contrastive: bool = _is_contrastive(self.arch)

        logger.info(
            "Loading %s | arch=%s | device=%s | dtype=%s",
            model_name_or_path,
            self.arch,
            self.device,
            self.dtype,
        )

        self.processor, self.model = self._load_model_and_processor()

        # Conversation history for multi-turn chat models
        self._conversation_history: List[Dict[str, str]] = []

        logger.info("Model loaded successfully.")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model_and_processor(self) -> Tuple[Any, Any]:
        """Dispatch to the correct loader based on detected architecture."""
        loaders = {
            "blip2":        self._load_blip2,
            "llava":        self._load_llava,
            "llava_next":   self._load_llava_next,
            "instructblip": self._load_instructblip,
            "paligemma":    self._load_paligemma,
            "clip":         self._load_clip,
            "siglip":       self._load_clip,   # same API
            "align":        self._load_clip,   # same API
        }
        loader = loaders.get(self.arch, self._load_auto)
        return loader()

    def _common_kwargs(self) -> Dict[str, Any]:
        """Shared model-loading kwargs."""
        kwargs: Dict[str, Any] = {
            "torch_dtype": self.dtype,
            "trust_remote_code": getattr(self.config, "trust_remote_code", False),
        }
        if self.device.type == "cuda":
            kwargs["device_map"] = "auto"
        if getattr(self.config, "use_flash_attention", False):
            kwargs["attn_implementation"] = "flash_attention_2"
        return kwargs

    def _load_blip2(self) -> Tuple[Blip2Processor, Blip2ForConditionalGeneration]:
        processor = Blip2Processor.from_pretrained(self.model_name_or_path)
        model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name_or_path, **self._common_kwargs()
        )
        if self.device.type != "cuda":
            model = model.to(self.device)
        return processor, model

    def _load_llava(self) -> Tuple[AutoProcessor, LlavaForConditionalGeneration]:
        processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name_or_path, **self._common_kwargs()
        )
        if self.device.type != "cuda":
            model = model.to(self.device)
        return processor, model

    def _load_llava_next(self) -> Tuple[AutoProcessor, LlavaNextForConditionalGeneration]:
        processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_name_or_path, **self._common_kwargs()
        )
        if self.device.type != "cuda":
            model = model.to(self.device)
        return processor, model

    def _load_instructblip(
        self,
    ) -> Tuple[InstructBlipProcessor, InstructBlipForConditionalGeneration]:
        processor = InstructBlipProcessor.from_pretrained(self.model_name_or_path)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            self.model_name_or_path, **self._common_kwargs()
        )
        if self.device.type != "cuda":
            model = model.to(self.device)
        return processor, model

    def _load_paligemma(
        self,
    ) -> Tuple[AutoProcessor, PaliGemmaForConditionalGeneration]:
        processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_name_or_path, **self._common_kwargs()
        )
        if self.device.type != "cuda":
            model = model.to(self.device)
        return processor, model

    def _load_clip(self) -> Tuple[CLIPProcessor, CLIPModel]:
        processor = CLIPProcessor.from_pretrained(self.model_name_or_path)
        model = CLIPModel.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.dtype,
            trust_remote_code=getattr(self.config, "trust_remote_code", False),
        )
        model = model.to(self.device)
        return processor, model

    def _load_auto(self) -> Tuple[AutoProcessor, Any]:
        """Generic fallback: tries AutoProcessor + AutoModelForVision2Seq,
        then falls back to AutoModelForCausalLM."""
        processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=getattr(self.config, "trust_remote_code", False),
        )
        kwargs = self._common_kwargs()
        try:
            model = AutoModelForVision2Seq.from_pretrained(
                self.model_name_or_path, **kwargs
            )
        except Exception:
            logger.warning(
                "AutoModelForVision2Seq failed; retrying with AutoModelForCausalLM."
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path, **kwargs
            )
        if self.device.type != "cuda":
            model = model.to(self.device)
        return processor, model

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _load_image(self, image_path: str) -> Image.Image:
        """Load a PIL image from a local file path or an HTTP(S) URL.

        Parameters
        ----------
        image_path:
            Filesystem path (absolute or relative) or a URL starting with
            ``http://`` or ``https://``.

        Returns
        -------
        PIL.Image.Image
            RGB image.

        Raises
        ------
        FileNotFoundError
            If ``image_path`` is a local path that does not exist.
        requests.HTTPError
            If the URL returns a non-200 status code.
        """
        if image_path.startswith("http://") or image_path.startswith("https://"):
            try:
                response = requests.get(image_path, stream=True, timeout=30)
                response.raise_for_status()
                img = Image.open(response.raw).convert("RGB")
            except requests.RequestException as exc:
                raise requests.HTTPError(
                    f"Failed to download image from {image_path}: {exc}"
                ) from exc
        else:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path.resolve()}")
            img = Image.open(path).convert("RGB")
        return img

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def _format_prompt(self, prompt: str, task: str) -> str:
        """Apply model-specific prompt template for the given task.

        Parameters
        ----------
        prompt:
            Raw user prompt string (may be empty for captioning).
        task:
            One of ``'captioning'``, ``'vqa'``, ``'chat'``, ``'classify'``.

        Returns
        -------
        str
            Formatted prompt ready to feed into the processor/tokenizer.
        """
        templates = PROMPT_TEMPLATES.get(self.arch) or PROMPT_TEMPLATES.get("auto", {})
        template = templates.get(task, "{prompt}")

        # Tasks that do not require a user prompt
        if task == "captioning" and not prompt:
            return template.format(prompt="") if "{prompt}" in template else template

        try:
            return template.format(prompt=prompt)
        except KeyError:
            # Template has no {prompt} slot — return as-is
            return template

    # ------------------------------------------------------------------
    # Processor dispatch helpers
    # ------------------------------------------------------------------

    def _encode_inputs(
        self, image: Image.Image, text: str
    ) -> Dict[str, torch.Tensor]:
        """Run the appropriate processor and return tensors on `self.device`."""
        if self.arch == "paligemma":
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt",
            )
        elif self.arch in ("instructblip",):
            inputs = self.processor(
                images=image,
                text=text,
                return_tensors="pt",
            )
        else:
            # Works for BLIP-2, LLaVA, LLaVA-Next, and most Auto models
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt",
            )
        return {k: v.to(self.device) for k, v in inputs.items()}

    # ------------------------------------------------------------------
    # Single prediction
    # ------------------------------------------------------------------

    def predict_single(
        self,
        image_path: str,
        prompt: str = "",
        task: str = "captioning",
        **gen_kwargs: Any,
    ) -> str:
        """Run inference on a single (image, prompt) pair.

        Parameters
        ----------
        image_path:
            Local path or URL to the image.
        prompt:
            User text prompt.  Can be empty for pure captioning.
        task:
            ``'captioning'`` | ``'vqa'`` | ``'chat'`` | ``'classify'``.
        **gen_kwargs:
            Override any :class:`GenerationConfig` field, e.g.
            ``max_new_tokens=128``.

        Returns
        -------
        str
            Decoded model output string (stripped of leading/trailing whitespace).
        """
        if self._contrastive:
            raise TypeError(
                "predict_single() is for generative models. "
                "Use compute_similarity() or zero_shot_classify() for CLIP-like models."
            )

        image = self._load_image(image_path)
        formatted_prompt = self._format_prompt(prompt, task)
        inputs = self._encode_inputs(image, formatted_prompt)

        gen_cfg = GenerationConfig().merge(gen_kwargs)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                **gen_cfg.to_dict(),
            )

        # Strip the input tokens from the generated ids when the model
        # echoes them back (e.g. causal LM style).
        input_token_len = inputs.get("input_ids", torch.tensor([])).shape[-1]
        generated_ids = output_ids[:, input_token_len:]

        decoded = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return decoded[0].strip() if decoded else ""

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        items: List[Dict[str, str]],
        task: str = "captioning",
        batch_size: int = 8,
        **gen_kwargs: Any,
    ) -> List[str]:
        """Run inference over a list of image-prompt pairs.

        Parameters
        ----------
        items:
            List of dicts, each with keys ``'image_path'`` and
            ``'prompt'`` (prompt may be absent for captioning).
        task:
            Inference task passed to :meth:`_format_prompt`.
        batch_size:
            Number of examples to process per forward pass.
        **gen_kwargs:
            Forwarded to ``model.generate()``.

        Returns
        -------
        list[str]
            Predictions in the same order as ``items``.
        """
        if self._contrastive:
            raise TypeError(
                "predict_batch() is for generative models. "
                "For CLIP, iterate compute_similarity() yourself."
            )

        gen_cfg = GenerationConfig().merge(gen_kwargs)
        results: List[str] = []

        for start in tqdm(
            range(0, len(items), batch_size),
            desc="Batch inference",
            unit="batch",
            file=sys.stdout,
        ):
            chunk = items[start : start + batch_size]
            batch_results = self._process_batch_chunk(chunk, task, gen_cfg)
            results.extend(batch_results)

        return results

    def _process_batch_chunk(
        self,
        chunk: List[Dict[str, str]],
        task: str,
        gen_cfg: GenerationConfig,
    ) -> List[str]:
        """Process a single mini-batch and return decoded strings."""
        images: List[Image.Image] = []
        texts: List[str] = []

        for item in chunk:
            try:
                img = self._load_image(item["image_path"])
            except (FileNotFoundError, requests.HTTPError) as exc:
                logger.warning("Skipping item due to image load error: %s", exc)
                # Insert a blank placeholder image so indices remain aligned
                img = Image.new("RGB", (224, 224), color=(0, 0, 0))
            images.append(img)
            texts.append(self._format_prompt(item.get("prompt", ""), task))

        try:
            inputs = self.processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    **gen_cfg.to_dict(),
                )

            input_token_len = inputs.get("input_ids", torch.tensor([])).shape[-1]
            generated_ids = output_ids[:, input_token_len:]

            decoded = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            return [d.strip() for d in decoded]

        except Exception as exc:  # noqa: BLE001
            logger.error("Batch chunk failed: %s — falling back to single inference.", exc)
            results = []
            for item in chunk:
                try:
                    pred = self.predict_single(
                        item["image_path"],
                        item.get("prompt", ""),
                        task,
                        **gen_cfg.to_dict(),
                    )
                    results.append(pred)
                except Exception as inner_exc:  # noqa: BLE001
                    logger.error("Single fallback also failed: %s", inner_exc)
                    results.append("")
            return results

    # ------------------------------------------------------------------
    # Interactive chat
    # ------------------------------------------------------------------

    def predict_interactive(self, image_path: Optional[str] = None) -> None:
        """Start an interactive CLI chat loop.

        Special commands
        ----------------
        ``quit`` / ``exit``
            Exit the loop.
        ``new image <path_or_url>``
            Switch to a different image.
        ``clear``
            Clear the conversation history.
        ``history``
            Print the current conversation history.

        Parameters
        ----------
        image_path:
            Optional initial image path or URL.  The user will be prompted
            to supply one if not provided.
        """
        print(
            "\n" + "=" * 60 + "\n"
            "  VLM Interactive Chat\n"
            "  Commands: 'quit', 'new image <path>', 'clear', 'history'\n"
            + "=" * 60
        )

        current_image_path: Optional[str] = image_path
        current_image: Optional[Image.Image] = None

        if current_image_path:
            try:
                current_image = self._load_image(current_image_path)
                print(f"[Image loaded: {current_image_path}]")
            except Exception as exc:  # noqa: BLE001
                logger.error("Could not load initial image: %s", exc)
                current_image_path = None

        self._conversation_history.clear()

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[Exiting interactive mode]")
                break

            if not user_input:
                continue

            # --- Special commands ---
            lower = user_input.lower()

            if lower in ("quit", "exit"):
                print("[Exiting interactive mode]")
                break

            if lower == "clear":
                self._conversation_history.clear()
                print("[Conversation history cleared]")
                continue

            if lower == "history":
                if not self._conversation_history:
                    print("[No history yet]")
                else:
                    for turn in self._conversation_history:
                        print(f"  User     : {turn['user']}")
                        print(f"  Assistant: {turn['assistant']}")
                continue

            if lower.startswith("new image "):
                new_path = user_input[len("new image "):].strip()
                try:
                    current_image = self._load_image(new_path)
                    current_image_path = new_path
                    self._conversation_history.clear()
                    print(f"[Image switched to: {new_path}]")
                except Exception as exc:  # noqa: BLE001
                    print(f"[Error loading image: {exc}]")
                continue

            # --- Need an image to proceed ---
            if current_image is None:
                provided = input("No image loaded. Enter image path or URL: ").strip()
                if not provided:
                    print("[Skipped — no image provided]")
                    continue
                try:
                    current_image = self._load_image(provided)
                    current_image_path = provided
                    print(f"[Image loaded: {provided}]")
                except Exception as exc:  # noqa: BLE001
                    print(f"[Error loading image: {exc}]")
                    continue

            # --- Build prompt with optional multi-turn context ---
            prompt_with_history = self._build_chat_prompt(user_input)

            try:
                t0 = time.perf_counter()
                response = self._generate_from_image(
                    current_image, prompt_with_history, task="chat"
                )
                elapsed = time.perf_counter() - t0
                print(f"\nAssistant: {response}")
                print(f"  [{elapsed:.2f}s]")
                self._conversation_history.append(
                    {"user": user_input, "assistant": response}
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Generation error: %s", exc)
                print(f"[Error during generation: {exc}]")

    def _build_chat_prompt(self, user_input: str) -> str:
        """Incorporate conversation history into the current turn's prompt.

        For simplicity, history is concatenated as plain text.  Models with
        dedicated chat templates (e.g. LLaVA) will handle this via their
        own tokenizer chat templates; others receive the concatenated string.
        """
        if not self._conversation_history:
            return user_input
        history_str = ""
        for turn in self._conversation_history[-4:]:  # last 4 turns
            history_str += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        return f"{history_str}User: {user_input}\nAssistant:"

    def _generate_from_image(
        self,
        image: Image.Image,
        prompt: str,
        task: str = "chat",
        **gen_kwargs: Any,
    ) -> str:
        """Low-level single-image generation from a pre-loaded PIL image."""
        formatted_prompt = self._format_prompt(prompt, task)
        inputs = self._encode_inputs(image, formatted_prompt)
        gen_cfg = GenerationConfig().merge(gen_kwargs)

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_cfg.to_dict())

        input_token_len = inputs.get("input_ids", torch.tensor([])).shape[-1]
        generated_ids = output_ids[:, input_token_len:]
        decoded = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return decoded[0].strip() if decoded else ""

    # ------------------------------------------------------------------
    # Streaming prediction
    # ------------------------------------------------------------------

    def stream_predict(
        self,
        image_path: str,
        prompt: str,
        task: str = "chat",
        **gen_kwargs: Any,
    ) -> Generator[str, None, None]:
        """Stream generated tokens one-by-one using a background thread.

        Yields
        ------
        str
            Individual decoded token strings as they are generated.

        Notes
        -----
        Uses :class:`transformers.TextIteratorStreamer` under the hood.
        Requires the model to support ``streamer=`` kwarg in ``generate()``.
        CLIP-like contrastive models are not supported.

        Example
        -------
        >>> for token in predictor.stream_predict("img.jpg", "Describe this"):
        ...     print(token, end="", flush=True)
        """
        if self._contrastive:
            raise TypeError("stream_predict() is not available for contrastive models.")

        image = self._load_image(image_path)
        formatted_prompt = self._format_prompt(prompt, task)
        inputs = self._encode_inputs(image, formatted_prompt)
        gen_cfg = GenerationConfig().merge(gen_kwargs)

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=60.0,
        )

        generate_kwargs = {
            **inputs,
            **gen_cfg.to_dict(),
            "streamer": streamer,
        }

        thread = Thread(target=self.model.generate, kwargs=generate_kwargs, daemon=True)
        thread.start()

        try:
            for new_text in streamer:
                yield new_text
        finally:
            thread.join(timeout=120)

    # ------------------------------------------------------------------
    # CLIP / Contrastive embedding & similarity
    # ------------------------------------------------------------------

    def embed_image(self, image_path: str) -> np.ndarray:
        """Compute a normalised image embedding using a CLIP-like model.

        Parameters
        ----------
        image_path:
            Local path or URL.

        Returns
        -------
        np.ndarray
            1-D float32 array of shape ``(embedding_dim,)``.
        """
        if not self._contrastive:
            raise TypeError(
                "embed_image() requires a contrastive model (CLIP/SigLIP). "
                f"Current arch: {self.arch}"
            )
        image = self._load_image(image_path)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            feats = self.model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats.squeeze(0).cpu().float().numpy()

    def embed_text(self, text: str) -> np.ndarray:
        """Compute a normalised text embedding using a CLIP-like model.

        Parameters
        ----------
        text:
            Input text string.

        Returns
        -------
        np.ndarray
            1-D float32 array of shape ``(embedding_dim,)``.
        """
        if not self._contrastive:
            raise TypeError(
                "embed_text() requires a contrastive model (CLIP/SigLIP). "
                f"Current arch: {self.arch}"
            )
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(
            self.device
        )

        with torch.inference_mode():
            feats = self.model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats.squeeze(0).cpu().float().numpy()

    def compute_similarity(
        self,
        image_path: str,
        texts: List[str],
    ) -> List[float]:
        """Compute cosine similarity between an image and a list of texts.

        Parameters
        ----------
        image_path:
            Local path or URL.
        texts:
            List of text strings to compare against.

        Returns
        -------
        list[float]
            Similarity scores in the range ``[0, 1]`` (softmax probabilities),
            one per text.
        """
        if not self._contrastive:
            raise TypeError(
                "compute_similarity() requires a contrastive model. "
                f"Current arch: {self.arch}"
            )
        if not texts:
            return []

        image = self._load_image(image_path)
        inputs = self.processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            # logits_per_image: (1, len(texts))
            logits = outputs.logits_per_image  # type: ignore[attr-defined]
            probs = logits.softmax(dim=-1).squeeze(0)

        return probs.cpu().float().tolist()

    def zero_shot_classify(
        self,
        image_path: str,
        class_labels: List[str],
    ) -> Dict[str, float]:
        """Zero-shot image classification via CLIP-style similarity.

        Parameters
        ----------
        image_path:
            Local path or URL.
        class_labels:
            Human-readable class names (e.g. ``["cat", "dog", "bird"]``).

        Returns
        -------
        dict[str, float]
            Mapping of label → softmax probability, sorted descending by score.
        """
        if not class_labels:
            raise ValueError("class_labels must not be empty.")

        # Wrap labels in a prompt for better zero-shot performance
        text_prompts = [f"a photo of a {label}" for label in class_labels]
        scores = self.compute_similarity(image_path, text_prompts)

        label_scores = dict(zip(class_labels, scores))
        # Sort descending by score
        return dict(sorted(label_scores.items(), key=lambda kv: kv[1], reverse=True))


# ---------------------------------------------------------------------------
# Batch predict from file
# ---------------------------------------------------------------------------

def batch_predict_from_file(
    input_file: str,
    output_file: str,
    predictor: VLMPredictor,
    task: str,
    batch_size: int = 8,
    **gen_kwargs: Any,
) -> None:
    """Read a JSONL input file and write predictions to a JSONL output file.

    Each line in ``input_file`` must be a JSON object with at least an
    ``"image_path"`` key and optionally a ``"prompt"`` key.  The function
    appends a ``"prediction"`` key to each object and writes the result to
    ``output_file``.

    Parameters
    ----------
    input_file:
        Path to the input ``.jsonl`` file.
    output_file:
        Path to the output ``.jsonl`` file (created or overwritten).
    predictor:
        Initialised :class:`VLMPredictor` instance.
    task:
        Task string forwarded to :meth:`VLMPredictor.predict_batch`.
    batch_size:
        Mini-batch size.
    **gen_kwargs:
        Extra generation keyword arguments.

    Raises
    ------
    FileNotFoundError
        If ``input_file`` does not exist.
    ValueError
        If a line cannot be parsed as valid JSON or is missing
        the ``"image_path"`` key.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path.resolve()}")

    records: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Line {line_no} in {input_file} is not valid JSON: {exc}"
                ) from exc
            if "image_path" not in record:
                raise ValueError(
                    f"Line {line_no} in {input_file} is missing 'image_path' key."
                )
            records.append(record)

    logger.info("Loaded %d records from %s", len(records), input_file)

    items = [{"image_path": r["image_path"], "prompt": r.get("prompt", "")} for r in records]
    predictions = predictor.predict_batch(
        items, task=task, batch_size=batch_size, **gen_kwargs
    )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_fh:
        for record, prediction in zip(records, predictions):
            record["prediction"] = prediction
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Predictions written to %s", output_path.resolve())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description="VLM end-to-end inference pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              # Captioning
              python predict.py --model Salesforce/blip2-opt-2.7b --image cat.jpg

              # VQA
              python predict.py --model llava-hf/llava-1.5-7b-hf \\
                  --image dog.jpg --prompt "What breed?" --task vqa

              # Batch
              python predict.py --model Salesforce/blip2-opt-2.7b \\
                  --mode batch --input_file in.jsonl --output_file out.jsonl

              # Zero-shot classification (CLIP)
              python predict.py --model openai/clip-vit-base-patch32 \\
                  --image photo.jpg --mode classify \\
                  --class_labels "cat,dog,bird,fish"

              # Interactive
              python predict.py --model llava-hf/llava-1.5-7b-hf --mode interactive
            """
        ),
    )

    # --- Core arguments ---
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        metavar="MODEL",
        help="HF model ID or local checkpoint path.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        metavar="PATH_OR_URL",
        help="Image path or URL (required for single/classify modes).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        metavar="TEXT",
        help="Text prompt for the model.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="captioning",
        choices=["captioning", "vqa", "chat", "classify"],
        help="Inference task type (default: captioning).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "batch", "interactive", "classify"],
        help="Run mode (default: single).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: 'auto', 'cuda', 'cpu', 'mps' (default: auto).",
    )

    # --- Batch mode ---
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        metavar="FILE",
        help="JSONL input file for batch mode.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="predictions.jsonl",
        metavar="FILE",
        help="JSONL output file for batch mode (default: predictions.jsonl).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        metavar="N",
        help="Batch size for batch mode (default: 8).",
    )

    # --- Generation config ---
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        metavar="N",
        help="Maximum number of new tokens to generate (default: 256).",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        metavar="N",
        help="Beam search width (default: 1 = greedy).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        metavar="T",
        help="Sampling temperature (default: 1.0).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        metavar="P",
        help="Nucleus sampling top-p (default: 1.0).",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=False,
        help="Enable sampling (default: greedy / beam search).",
    )

    # --- Classify mode ---
    parser.add_argument(
        "--class_labels",
        type=str,
        default=None,
        metavar="LABELS",
        help='Comma-separated class labels for classify mode, e.g. "cat,dog,bird".',
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:  # noqa: C901
    """CLI entry point.

    Returns
    -------
    int
        Exit code (0 = success, non-zero = error).
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # ---- Build generation config from CLI args ----
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.do_sample,
    }

    # ---- Initialise predictor ----
    try:
        predictor = VLMPredictor(
            model_name_or_path=args.model,
            device=args.device,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load model '%s': %s", args.model, exc)
        return 1

    # ====================================================================
    # Mode: batch
    # ====================================================================
    if args.mode == "batch":
        if not args.input_file:
            logger.error("--input_file is required for batch mode.")
            return 1
        try:
            batch_predict_from_file(
                input_file=args.input_file,
                output_file=args.output_file,
                predictor=predictor,
                task=args.task,
                batch_size=args.batch_size,
                **gen_kwargs,
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.error("%s", exc)
            return 1
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error during batch inference: %s", exc)
            return 1
        return 0

    # ====================================================================
    # Mode: interactive
    # ====================================================================
    if args.mode == "interactive":
        try:
            predictor.predict_interactive(image_path=args.image)
        except Exception as exc:  # noqa: BLE001
            logger.error("Interactive mode error: %s", exc)
            return 1
        return 0

    # ====================================================================
    # Mode: classify (zero-shot via CLIP)
    # ====================================================================
    if args.mode == "classify":
        if not args.image:
            logger.error("--image is required for classify mode.")
            return 1
        if not args.class_labels:
            logger.error("--class_labels is required for classify mode.")
            return 1

        labels = [lbl.strip() for lbl in args.class_labels.split(",") if lbl.strip()]
        if not labels:
            logger.error("No valid class labels found in --class_labels.")
            return 1

        try:
            scores = predictor.zero_shot_classify(args.image, labels)
        except TypeError as exc:
            # Model is generative — fall back to sequential VQA classification
            logger.warning(
                "Model is not contrastive. Falling back to VQA-style classification. (%s)",
                exc,
            )
            scores = {}
            for label in labels:
                q = f"Is this a photo of a {label}? Answer yes or no."
                try:
                    answer = predictor.predict_single(
                        args.image, prompt=q, task="vqa", **gen_kwargs
                    )
                    scores[label] = 1.0 if "yes" in answer.lower() else 0.0
                except Exception as inner_exc:  # noqa: BLE001
                    logger.error("Classification failed for label '%s': %s", label, inner_exc)
                    scores[label] = 0.0
        except Exception as exc:  # noqa: BLE001
            logger.error("Classification error: %s", exc)
            return 1

        print("\nZero-Shot Classification Results")
        print("-" * 40)
        for label, score in scores.items():
            bar = "█" * int(score * 30)
            print(f"  {label:<20s} {score:.4f}  {bar}")
        print()
        return 0

    # ====================================================================
    # Mode: single (default)
    # ====================================================================
    if not args.image:
        logger.error("--image is required for single mode.")
        return 1

    try:
        prediction = predictor.predict_single(
            image_path=args.image,
            prompt=args.prompt,
            task=args.task,
            **gen_kwargs,
        )
        print(f"\nPrediction: {prediction}\n")
    except Exception as exc:  # noqa: BLE001
        logger.error("Prediction failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
