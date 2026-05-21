"""
main.py — Unified CLI Entry Point for VLM Pipeline
====================================================
Ties together config, preprocess, train, evaluate, and predict modules
into a single command-line interface with subcommands:

    python main.py train   --model blip2-opt-2.7b --task captioning ...
    python main.py evaluate --model blip2-opt-2.7b --checkpoint ./ckpt ...
    python main.py predict  --model blip2-opt-2.7b --mode interactive ...
    python main.py preprocess --model blip2-opt-2.7b --dataset_type jsonl ...
    python main.py list-models --task captioning --format table

Author : VLM Pipeline
Created: 2026-05-20
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Pipeline module imports
# ---------------------------------------------------------------------------
from config import (
    MODEL_REGISTRY,
    DataConfig,
    PathConfig,
    TrainingConfig,
    VLMConfig,
    get_config,
    list_models,
    validate_config,
)
from evaluate import Evaluator, compare_checkpoints, evaluate_checkpoint
from predict import VLMPredictor
from preprocess import get_dataloaders, get_processor, preprocess_and_cache
from train import Trainer, freeze_backbone, get_model

# ---------------------------------------------------------------------------
# ASCII Banner
# ---------------------------------------------------------------------------
BANNER = r"""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║    ██╗   ██╗██╗      ███╗   ███╗    ██████╗ ██╗██████╗ ███████╗         ║
║    ██║   ██║██║      ████╗ ████║    ██╔══██╗██║██╔══██╗██╔════╝         ║
║    ██║   ██║██║      ██╔████╔██║    ██████╔╝██║██████╔╝█████╗           ║
║    ╚██╗ ██╔╝██║      ██║╚██╔╝██║    ██╔═══╝ ██║██╔═══╝ ██╔══╝          ║
║     ╚████╔╝ ███████╗ ██║ ╚═╝ ██║    ██║     ██║██║     ███████╗         ║
║      ╚═══╝  ╚══════╝ ╚═╝     ╚═╝    ╚═╝     ╚═╝╚═╝     ╚══════╝        ║
║                                                                          ║
║              Vision-Language Model Training & Inference Pipeline         ║
║                        github.com/your-org/vlm-pipeline                 ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
logger = logging.getLogger("vlm_pipeline")


def setup_logging(args: argparse.Namespace) -> None:
    """Configure root logger with console + optional rotating file handler.

    Log level is determined by the ``--log_level`` argument (default INFO).
    When ``--log_file`` is provided a FileHandler is also attached so every
    line is persisted to disk.

    Args:
        args: Parsed CLI namespace that may contain ``log_level`` and
              ``log_file`` attributes.
    """
    log_level_str: str = getattr(args, "log_level", "INFO").upper()
    numeric_level = getattr(logging, log_level_str, logging.INFO)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove pre-existing handlers to avoid duplicate output when the module
    # is imported multiple times during testing.
    root_logger.handlers.clear()

    # Console handler — always present
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(numeric_level)
    ch.setFormatter(fmt)
    root_logger.addHandler(ch)

    # Optional file handler
    log_file: Optional[str] = getattr(args, "log_file", None)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setLevel(numeric_level)
        fh.setFormatter(fmt)
        root_logger.addHandler(fh)
        logger.info("File logging enabled → %s", log_path)


# ---------------------------------------------------------------------------
# Config builder helpers
# ---------------------------------------------------------------------------

def _build_vlm_config(args: argparse.Namespace) -> VLMConfig:
    """Construct a :class:`VLMConfig` from a parsed CLI *args* namespace.

    Fields present on ``args`` are mapped to the corresponding config
    dataclass fields.  Missing optional fields fall back to dataclass defaults.

    Args:
        args: Namespace produced by ``argparse``.

    Returns:
        Fully populated :class:`VLMConfig` instance.
    """
    model_name: str = args.model

    # ---- Training sub-config ------------------------------------------------
    training_kwargs: Dict[str, Any] = {
        "task": getattr(args, "task", "captioning"),
        "epochs": getattr(args, "epochs", 3),
        "batch_size": getattr(args, "batch_size", 8),
        "learning_rate": getattr(args, "lr", 1e-4),
        "gradient_accumulation_steps": getattr(args, "grad_accum", 1),
        "use_lora": getattr(args, "use_lora", False),
        "use_qlora": getattr(args, "use_qlora", False),
        "lora_r": getattr(args, "lora_r", 16),
        "lora_alpha": getattr(args, "lora_alpha", 32),
        "lora_dropout": getattr(args, "lora_dropout", 0.05),
        "bf16": getattr(args, "bf16", False),
        "fp16": getattr(args, "fp16", False),
        "use_wandb": getattr(args, "use_wandb", False),
        "wandb_project": getattr(args, "wandb_project", "vlm_pipeline"),
        "freeze_backbone": getattr(args, "freeze_backbone", False),
        "max_length": getattr(args, "max_length", 128),
        "image_size": getattr(args, "image_size", 224),
        "num_workers": getattr(args, "num_workers", 4),
        "dry_run": getattr(args, "dry_run", False),
    }
    training_cfg = TrainingConfig(**training_kwargs)

    # ---- Data sub-config ----------------------------------------------------
    data_kwargs: Dict[str, Any] = {
        "dataset_type": getattr(args, "dataset_type", "jsonl"),
        "data_dir": getattr(args, "data_dir", None),
        "train_file": getattr(args, "train_file", None),
        "val_file": getattr(args, "val_file", None),
        "test_file": getattr(args, "test_file", None),
        "hf_dataset": getattr(args, "hf_dataset", None),
        "hf_config": getattr(args, "hf_config", None),
    }
    data_cfg = DataConfig(**data_kwargs)

    # ---- Path sub-config ----------------------------------------------------
    path_kwargs: Dict[str, Any] = {
        "output_dir": getattr(args, "output_dir", "./outputs"),
        "checkpoint_dir": getattr(args, "checkpoint_dir", "./checkpoints"),
    }
    path_cfg = PathConfig(**path_kwargs)

    return VLMConfig(
        model_name=model_name,
        training=training_cfg,
        data=data_cfg,
        paths=path_cfg,
    )


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------

def run_train(args: argparse.Namespace) -> None:
    """Execute the full training workflow.

    Steps:
        1. Build :class:`VLMConfig` from CLI arguments.
        2. Validate config via :func:`validate_config`.
        3. Obtain a processor via :func:`get_processor`.
        4. Build train/val dataloaders via :func:`get_dataloaders`.
        5. Instantiate the model via :func:`get_model`.
        6. Optionally freeze the vision backbone.
        7. Create a :class:`Trainer` and call :meth:`Trainer.train`.
        8. Print final training metrics.

    Args:
        args: Parsed CLI namespace for the ``train`` subcommand.
    """
    logger.info("═" * 72)
    logger.info("STARTING TRAINING")
    logger.info("═" * 72)

    # 1. Build config
    logger.info("Building VLMConfig …")
    config = _build_vlm_config(args)

    # 2. Validate
    logger.info("Validating config …")
    validate_config(config)

    # Dry-run notice
    if config.training.dry_run:
        logger.warning(
            "⚠  DRY-RUN mode active — training will stop after 1 step."
        )

    # 3. Processor
    logger.info("Loading processor for model '%s' …", config.model_name)
    processor = get_processor(config)

    # 4. Dataloaders
    logger.info("Building dataloaders …")
    train_loader, val_loader = get_dataloaders(config, processor)
    logger.info(
        "Train batches: %d | Val batches: %d",
        len(train_loader),
        len(val_loader) if val_loader is not None else 0,
    )

    # 5. Model
    logger.info("Instantiating model …")
    model = get_model(config)

    # 6. Freeze backbone (optional)
    if config.training.freeze_backbone:
        logger.info("Freezing vision backbone parameters …")
        freeze_backbone(model)

    # 7. Trainer
    resume_checkpoint: Optional[str] = getattr(args, "resume", None)
    trainer = Trainer(
        config=config,
        model=model,
        processor=processor,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        resume_from_checkpoint=resume_checkpoint,
    )

    start_time = time.time()
    metrics = trainer.train()
    elapsed = time.time() - start_time

    # 8. Report metrics
    logger.info("═" * 72)
    logger.info("TRAINING COMPLETE — elapsed: %.1f s", elapsed)
    logger.info("─" * 72)
    if metrics:
        for k, v in metrics.items():
            logger.info("  %-30s %s", k, v)
    logger.info("═" * 72)

    # Persist metrics as JSON next to the output directory
    metrics_path = (
        Path(config.paths.output_dir) / "train_metrics.json"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"elapsed_seconds": round(elapsed, 2), **(metrics or {})},
            fh,
            indent=2,
            default=str,
        )
    logger.info("Metrics saved → %s", metrics_path)


def run_evaluate(args: argparse.Namespace) -> None:
    """Execute the evaluation workflow for a saved checkpoint.

    Steps:
        1. Build :class:`VLMConfig` from CLI arguments.
        2. Validate config.
        3. Determine eval split (``test`` or ``val``).
        4. Load dataloader for the requested split.
        5. If ``--compare`` is given, run multi-checkpoint comparison.
        6. Otherwise, instantiate :class:`Evaluator` and call evaluate.
        7. Print and save the evaluation report.

    Args:
        args: Parsed CLI namespace for the ``evaluate`` subcommand.
    """
    logger.info("═" * 72)
    logger.info("STARTING EVALUATION")
    logger.info("═" * 72)

    # 1. Config
    config = _build_vlm_config(args)
    validate_config(config)

    checkpoint_path: str = args.checkpoint
    split: str = getattr(args, "split", "test")
    compare_str: Optional[str] = getattr(args, "compare", None)

    logger.info("Checkpoint : %s", checkpoint_path)
    logger.info("Split      : %s", split)

    # 2. Processor + dataloader
    processor = get_processor(config)
    _train_loader, val_loader = get_dataloaders(config, processor, splits=[split])
    eval_loader = val_loader  # get_dataloaders honours the ``splits`` arg

    # 3. Multi-checkpoint comparison
    if compare_str:
        checkpoints_to_compare: List[str] = [
            c.strip() for c in compare_str.split(",") if c.strip()
        ]
        # Include the primary checkpoint in the comparison
        if checkpoint_path not in checkpoints_to_compare:
            checkpoints_to_compare.insert(0, checkpoint_path)

        logger.info(
            "Comparing %d checkpoints …", len(checkpoints_to_compare)
        )
        comparison_report = compare_checkpoints(
            config=config,
            checkpoints=checkpoints_to_compare,
            dataloader=eval_loader,
            processor=processor,
        )

        _save_and_print_report(
            report=comparison_report,
            output_dir=config.paths.output_dir,
            filename="comparison_report.json",
        )
        return

    # 4. Single checkpoint evaluation
    evaluator = Evaluator(
        config=config,
        checkpoint_path=checkpoint_path,
        processor=processor,
        dataloader=eval_loader,
    )
    report = evaluator.evaluate()

    _save_and_print_report(
        report=report,
        output_dir=config.paths.output_dir,
        filename="eval_report.json",
    )


def _save_and_print_report(
    report: Dict[str, Any],
    output_dir: str,
    filename: str,
) -> None:
    """Pretty-print *report* and save it as JSON.

    Args:
        report: Dictionary of evaluation metrics / results.
        output_dir: Directory to write the JSON report to.
        filename: Name of the JSON file.
    """
    logger.info("─" * 72)
    logger.info("EVALUATION RESULTS")
    logger.info("─" * 72)
    for k, v in report.items():
        logger.info("  %-35s %s", k, v)
    logger.info("─" * 72)

    report_path = Path(output_dir) / filename
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info("Report saved → %s", report_path)


def run_predict(args: argparse.Namespace) -> None:
    """Execute the inference workflow.

    Routes execution to one of four inference modes depending on
    ``args.mode``:

    * ``single``      — predict on a single image + prompt.
    * ``batch``       — predict on every row in ``--input_file``, writing
                         results to ``--output_file``.
    * ``interactive`` — run an interactive REPL in the terminal.
    * ``classify``    — zero-shot classification using ``--class_labels``.

    Args:
        args: Parsed CLI namespace for the ``predict`` subcommand.
    """
    logger.info("═" * 72)
    logger.info("STARTING INFERENCE  [mode=%s]", args.mode)
    logger.info("═" * 72)

    checkpoint: Optional[str] = getattr(args, "checkpoint", None)
    device: str = getattr(args, "device", "auto")
    task: str = getattr(args, "task", "captioning")

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": getattr(args, "max_new_tokens", 256),
        "num_beams": getattr(args, "num_beams", 4),
        "temperature": getattr(args, "temperature", 1.0),
        "top_p": getattr(args, "top_p", 0.9),
    }

    predictor = VLMPredictor(
        model_name=args.model,
        checkpoint_path=checkpoint,
        task=task,
        device=device,
        generation_kwargs=generation_kwargs,
    )

    mode: str = args.mode.lower()

    if mode == "single":
        image_src: str = getattr(args, "image", "")
        prompt: str = getattr(args, "prompt", "")
        if not image_src:
            logger.error("--image is required for single mode.")
            sys.exit(1)

        result = predictor.predict_single(image=image_src, prompt=prompt)
        print("\n" + "─" * 60)
        print("PREDICTION:", result)
        print("─" * 60)

    elif mode == "batch":
        input_file: str = getattr(args, "input_file", "")
        output_file: str = getattr(args, "output_file", "predictions.jsonl")
        if not input_file:
            logger.error("--input_file is required for batch mode.")
            sys.exit(1)

        logger.info("Batch input  → %s", input_file)
        logger.info("Batch output → %s", output_file)
        results = predictor.predict_batch(
            input_file=input_file,
            output_file=output_file,
        )
        logger.info("Batch complete. %d predictions written.", len(results))

    elif mode == "interactive":
        print("\n" + "═" * 60)
        print("  Interactive VLM Session  (type 'exit' or Ctrl-C to quit)")
        print("═" * 60)
        predictor.predict_interactive()

    elif mode == "classify":
        image_src = getattr(args, "image", "")
        class_labels_raw: str = getattr(args, "class_labels", "")
        if not image_src:
            logger.error("--image is required for classify mode.")
            sys.exit(1)
        if not class_labels_raw:
            logger.error("--class_labels is required for classify mode.")
            sys.exit(1)

        class_labels: List[str] = [
            lbl.strip() for lbl in class_labels_raw.split(",") if lbl.strip()
        ]
        result = predictor.zero_shot_classify(
            image=image_src,
            class_labels=class_labels,
        )
        print("\n" + "─" * 60)
        print("CLASSIFICATION RESULT:")
        for lbl, score in result.items():
            print(f"  {lbl:<30} {score:.4f}")
        print("─" * 60)

    else:
        logger.error(
            "Unknown --mode '%s'. Choose from: single, batch, interactive, classify.",
            mode,
        )
        sys.exit(1)


def run_preprocess(args: argparse.Namespace) -> None:
    """Execute the preprocessing and caching workflow.

    Calls :func:`preprocess_and_cache` with a config built from the
    provided CLI arguments.  The cached tensors / processed files are
    stored under ``--output_dir``.

    Args:
        args: Parsed CLI namespace for the ``preprocess`` subcommand.
    """
    logger.info("═" * 72)
    logger.info("STARTING PREPROCESSING")
    logger.info("═" * 72)

    config = _build_vlm_config(args)
    validate_config(config)

    processor = get_processor(config)

    logger.info("Preprocessing dataset type: %s", config.data.dataset_type)
    logger.info("Output dir: %s", config.paths.output_dir)

    start_time = time.time()
    stats = preprocess_and_cache(config=config, processor=processor)
    elapsed = time.time() - start_time

    logger.info("Preprocessing complete — elapsed: %.1f s", elapsed)
    if stats:
        for k, v in stats.items():
            logger.info("  %-30s %s", k, v)


def list_models_cmd(args: argparse.Namespace) -> None:
    """List all models registered in MODEL_REGISTRY.

    Optionally filters by ``--task`` and formats the output as a rich
    ASCII table (requires *tabulate*) or compact JSON.

    Args:
        args: Parsed CLI namespace for the ``list-models`` subcommand.
    """
    task_filter: Optional[str] = getattr(args, "task", None)
    fmt: str = getattr(args, "format", "table")

    models: List[Dict[str, Any]] = list_models(task=task_filter)

    if not models:
        print("No models found" + (f" for task '{task_filter}'" if task_filter else "") + ".")
        return

    if fmt == "json":
        print(json.dumps(models, indent=2, default=str))
        return

    # ---- Table format -------------------------------------------------------
    headers = ["Model Name", "Hub ID", "Supported Tasks", "Min GPU (GB)"]
    rows: List[List[str]] = []
    for m in models:
        supported_tasks = ", ".join(m.get("tasks", []))
        rows.append(
            [
                m.get("name", "—"),
                m.get("hub_id", "—"),
                supported_tasks or "—",
                str(m.get("min_gpu_gb", "—")),
            ]
        )

    try:
        from tabulate import tabulate  # optional dependency

        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    except ImportError:
        # Fallback: manual ASCII table
        col_widths = [
            max(len(headers[i]), max((len(r[i]) for r in rows), default=0))
            for i in range(len(headers))
        ]
        sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
        fmt_row = "| " + " | ".join(f"{{:<{w}}}" for w in col_widths) + " |"

        print(sep)
        print(fmt_row.format(*headers))
        print(sep)
        for row in rows:
            print(fmt_row.format(*row))
        print(sep)

    print(f"\n{len(models)} model(s) listed.")


# ---------------------------------------------------------------------------
# CLI Definition
# ---------------------------------------------------------------------------

EPILOG = """
EXAMPLES
────────

# Train BLIP-2 on a local JSONL captioning dataset with LoRA, mixed precision
python main.py train \\
    --model blip2-opt-2.7b \\
    --task captioning \\
    --dataset_type jsonl \\
    --data_dir ./data/coco \\
    --train_file train.jsonl \\
    --val_file val.jsonl \\
    --output_dir ./outputs/blip2_coco \\
    --epochs 5 --batch_size 16 --lr 5e-5 \\
    --use_lora --lora_r 16 --lora_alpha 32 \\
    --bf16 --use_wandb --wandb_project my_vlm_runs

# Quick sanity-check (dry run — 1 step only)
python main.py train \\
    --model llava-1.5-7b --task vqa --dataset_type hf \\
    --hf_dataset HuggingFaceM4/VQAv2 --dry_run

# Evaluate a checkpoint on the test split
python main.py evaluate \\
    --model blip2-opt-2.7b \\
    --checkpoint ./checkpoints/epoch-5 \\
    --task captioning \\
    --data_dir ./data/coco --test_file test.jsonl \\
    --output_dir ./eval_results

# Compare multiple checkpoints
python main.py evaluate \\
    --model blip2-opt-2.7b \\
    --checkpoint ./ckpt/epoch-3 \\
    --task captioning \\
    --data_dir ./data/coco \\
    --compare "./ckpt/epoch-1,./ckpt/epoch-2,./ckpt/epoch-3"

# Single-image captioning from URL
python main.py predict \\
    --model blip2-opt-2.7b \\
    --checkpoint ./checkpoints/best \\
    --mode single \\
    --image https://example.com/dog.jpg \\
    --prompt "Describe the image."

# Batch inference from a JSONL file
python main.py predict \\
    --model llava-1.5-7b --mode batch \\
    --input_file ./queries.jsonl \\
    --output_file ./predictions.jsonl

# Zero-shot image classification
python main.py predict \\
    --model clip-vit-large --mode classify \\
    --image ./cat.jpg \\
    --class_labels "cat,dog,bird,car"

# Interactive chat session
python main.py predict \\
    --model llava-1.5-13b --mode interactive --task chat

# Preprocess and cache a COCO dataset
python main.py preprocess \\
    --model blip2-opt-2.7b \\
    --dataset_type coco \\
    --data_dir ./data/coco \\
    --output_dir ./data/coco_cached \\
    --image_size 384

# List all available models in table format
python main.py list-models

# List models supporting VQA only, in JSON format
python main.py list-models --task vqa --format json
"""


def _add_common_data_args(parser: argparse.ArgumentParser) -> None:
    """Add shared dataset arguments to *parser*.

    Reused by the ``train``, ``evaluate``, and ``preprocess`` subparsers.
    """
    parser.add_argument(
        "--dataset_type",
        choices=["jsonl", "csv", "hf", "coco"],
        default="jsonl",
        help="Dataset format / source (default: jsonl).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Root directory containing the dataset files.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        metavar="FILE",
        help="Train split filename relative to --data_dir.",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default=None,
        metavar="FILE",
        help="Validation split filename relative to --data_dir.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        metavar="FILE",
        help="Test split filename relative to --data_dir.",
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default=None,
        metavar="DATASET",
        help="HuggingFace datasets identifier (e.g. HuggingFaceM4/VQAv2).",
    )
    parser.add_argument(
        "--hf_config",
        type=str,
        default=None,
        metavar="CONFIG",
        help="HuggingFace dataset config / subset name.",
    )


def _add_common_logging_args(parser: argparse.ArgumentParser) -> None:
    """Add shared logging arguments to *parser*."""
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional path to write log output to a file.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct and return the top-level :class:`argparse.ArgumentParser`.

    Returns:
        Configured parser with ``train``, ``evaluate``, ``predict``,
        ``preprocess``, and ``list-models`` subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="vlm_pipeline",
        description="Vision-Language Model (VLM) end-to-end pipeline.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_logging_args(parser)

    subparsers = parser.add_subparsers(
        dest="command",
        title="subcommands",
        metavar="<command>",
    )
    subparsers.required = True

    # =========================================================================
    # TRAIN subcommand
    # =========================================================================
    train_parser = subparsers.add_parser(
        "train",
        help="Train a VLM model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_logging_args(train_parser)

    # Required
    train_parser.add_argument(
        "--model",
        required=True,
        metavar="MODEL",
        help="Model name from MODEL_REGISTRY (e.g. blip2-opt-2.7b).",
    )
    train_parser.add_argument(
        "--task",
        required=True,
        choices=["captioning", "vqa", "retrieval", "classification", "chat"],
        help="Task type the model is trained for.",
    )

    # Data
    _add_common_data_args(train_parser)

    # Paths
    train_parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        metavar="PATH",
        help="Directory for saving training outputs and logs.",
    )
    train_parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        metavar="PATH",
        help="Directory for saving model checkpoints.",
    )

    # Training hyper-parameters
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        metavar="N",
        help="Number of training epochs.",
    )
    train_parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        metavar="N",
        help="Per-device train batch size.",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="Peak learning rate.",
    )
    train_parser.add_argument(
        "--grad_accum",
        type=int,
        default=1,
        metavar="N",
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum).",
    )

    # LoRA / QLoRA
    train_parser.add_argument(
        "--use_lora",
        action="store_true",
        default=False,
        help="Enable LoRA parameter-efficient fine-tuning.",
    )
    train_parser.add_argument(
        "--use_qlora",
        action="store_true",
        default=False,
        help="Enable QLoRA (4-bit quantisation + LoRA).",
    )
    train_parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        metavar="R",
        help="LoRA rank.",
    )
    train_parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        metavar="ALPHA",
        help="LoRA alpha scaling factor.",
    )
    train_parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        metavar="P",
        help="LoRA dropout probability.",
    )

    # Precision
    train_parser.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="Train in bfloat16 mixed precision.",
    )
    train_parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Train in float16 mixed precision.",
    )

    # W&B
    train_parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases experiment tracking.",
    )
    train_parser.add_argument(
        "--wandb_project",
        type=str,
        default="vlm_pipeline",
        metavar="PROJECT",
        help="W&B project name.",
    )

    # Misc
    train_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Path to checkpoint to resume training from.",
    )
    train_parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        default=False,
        help="Freeze the vision encoder weights during training.",
    )
    train_parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        metavar="N",
        help="Maximum token sequence length.",
    )
    train_parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        metavar="N",
        help="Image resolution (square) fed to the vision encoder.",
    )
    train_parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        metavar="N",
        help="DataLoader worker processes.",
    )
    train_parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Run only 1 training step (sanity check — no real training).",
    )

    # =========================================================================
    # EVALUATE subcommand
    # =========================================================================
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_logging_args(eval_parser)

    eval_parser.add_argument(
        "--model",
        required=True,
        metavar="MODEL",
        help="Model name from MODEL_REGISTRY.",
    )
    eval_parser.add_argument(
        "--checkpoint",
        required=True,
        metavar="PATH",
        help="Path to the saved checkpoint directory or file.",
    )
    eval_parser.add_argument(
        "--task",
        required=True,
        choices=["captioning", "vqa", "retrieval", "classification", "chat"],
        help="Task type.",
    )

    _add_common_data_args(eval_parser)

    eval_parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_outputs",
        metavar="PATH",
        help="Directory to save evaluation reports.",
    )
    eval_parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        metavar="N",
        help="Evaluation batch size.",
    )
    eval_parser.add_argument(
        "--split",
        choices=["test", "val"],
        default="test",
        help="Dataset split to evaluate on.",
    )
    eval_parser.add_argument(
        "--compare",
        type=str,
        default=None,
        metavar="CKPT1,CKPT2,...",
        help=(
            "Comma-separated list of additional checkpoint paths to compare "
            "against the primary --checkpoint."
        ),
    )
    eval_parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        metavar="N",
        help="Maximum token sequence length.",
    )
    eval_parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        metavar="N",
        help="Image resolution fed to the vision encoder.",
    )
    eval_parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        metavar="N",
        help="DataLoader worker processes.",
    )

    # =========================================================================
    # PREDICT subcommand
    # =========================================================================
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run inference with a VLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_logging_args(predict_parser)

    predict_parser.add_argument(
        "--model",
        required=True,
        metavar="MODEL",
        help="Model name from MODEL_REGISTRY.",
    )
    predict_parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to fine-tuned checkpoint. If omitted the base HF hub "
            "weights are used."
        ),
    )
    predict_parser.add_argument(
        "--mode",
        choices=["single", "batch", "interactive", "classify"],
        default="single",
        help="Inference mode.",
    )
    predict_parser.add_argument(
        "--image",
        type=str,
        default=None,
        metavar="PATH_OR_URL",
        help="Image path or URL (required for single and classify modes).",
    )
    predict_parser.add_argument(
        "--prompt",
        type=str,
        default="",
        metavar="TEXT",
        help="Text prompt to accompany the image.",
    )
    predict_parser.add_argument(
        "--task",
        type=str,
        default="captioning",
        choices=["captioning", "vqa", "retrieval", "classification", "chat"],
        help="Task hint passed to the predictor.",
    )
    predict_parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        metavar="FILE",
        help="JSONL file with image paths and prompts (required for batch mode).",
    )
    predict_parser.add_argument(
        "--output_file",
        type=str,
        default="predictions.jsonl",
        metavar="FILE",
        help="JSONL file to write batch predictions to.",
    )
    predict_parser.add_argument(
        "--class_labels",
        type=str,
        default=None,
        metavar="LABEL1,LABEL2,...",
        help="Comma-separated class labels for zero-shot classification.",
    )
    predict_parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        metavar="N",
        help="Maximum number of new tokens to generate.",
    )
    predict_parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        metavar="N",
        help="Beam search width (1 = greedy).",
    )
    predict_parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        metavar="T",
        help="Sampling temperature (ignored when num_beams > 1).",
    )
    predict_parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        metavar="P",
        help="Nucleus sampling probability mass.",
    )
    predict_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        metavar="DEVICE",
        help="Torch device string: 'cpu', 'cuda', 'cuda:1', or 'auto'.",
    )

    # =========================================================================
    # PREPROCESS subcommand
    # =========================================================================
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess and cache a dataset for faster training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_logging_args(preprocess_parser)

    preprocess_parser.add_argument(
        "--model",
        required=True,
        metavar="MODEL",
        help="Model name from MODEL_REGISTRY (determines processor to use).",
    )

    _add_common_data_args(preprocess_parser)

    preprocess_parser.add_argument(
        "--output_dir",
        type=str,
        default="./preprocessed",
        metavar="PATH",
        help="Directory to save preprocessed/cached tensors.",
    )
    preprocess_parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        metavar="N",
        help="Maximum token sequence length during tokenisation.",
    )
    preprocess_parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        metavar="N",
        help="Image resolution used during preprocessing.",
    )
    preprocess_parser.add_argument(
        "--task",
        type=str,
        default="captioning",
        choices=["captioning", "vqa", "retrieval", "classification", "chat"],
        help="Task type (influences preprocessing logic).",
    )

    # =========================================================================
    # LIST-MODELS subcommand
    # =========================================================================
    list_parser = subparsers.add_parser(
        "list-models",
        help="Display all available models in MODEL_REGISTRY.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_logging_args(list_parser)

    list_parser.add_argument(
        "--task",
        type=str,
        default=None,
        metavar="TASK",
        help="Filter models by supported task (e.g. captioning, vqa).",
    )
    list_parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format.",
    )

    return parser


# ---------------------------------------------------------------------------
# Command dispatcher
# ---------------------------------------------------------------------------

_COMMAND_MAP = {
    "train": run_train,
    "evaluate": run_evaluate,
    "predict": run_predict,
    "preprocess": run_preprocess,
    "list-models": list_models_cmd,
}


def dispatch(args: argparse.Namespace) -> int:
    """Dispatch *args* to the appropriate subcommand handler.

    Args:
        args: Fully parsed CLI namespace.

    Returns:
        Exit code (0 on success, non-zero on handled error).
    """
    handler = _COMMAND_MAP.get(args.command)
    if handler is None:
        logger.error("Unknown subcommand: '%s'", args.command)
        return 1

    try:
        handler(args)
    except KeyboardInterrupt:
        print("\n[vlm_pipeline] Interrupted by user.")
        return 130  # Standard SIGINT exit code
    except SystemExit:
        raise  # propagate explicit sys.exit() calls
    except Exception:  # pylint: disable=broad-except
        logger.critical(
            "Unhandled exception in '%s':\n%s",
            args.command,
            traceback.format_exc(),
        )
        return 1

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and execute the requested pipeline subcommand."""
    print(BANNER)
    print(
        f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"| Python {sys.version.split()[0]}\n"
    )

    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args)

    logger.debug("Parsed arguments: %s", vars(args))

    exit_code = dispatch(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
