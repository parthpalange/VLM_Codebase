"""
evaluate.py
===========
End-to-end evaluation module for the VLM pipeline.

Supports the following tasks
    - captioning   : BLEU-1/2/3/4, ROUGE-L, METEOR, CIDEr
    - vqa          : Exact Match, Token-F1, VQA-Accuracy (soft)
    - retrieval    : Recall@1/5/10, mAP, MRR  (CLIP-style embed space)
    - classification: Accuracy, macro/weighted F1, per-class precision/recall/F1
    - chat         : ROUGE-L, BERTScore (optional)

Standalone metric functions are importable from this module.
CIDEr-D is implemented from scratch (no pycocoevalcap dependency).
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
import re
import string
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Optional heavy dependencies – imported lazily so the module loads even if
# a particular library is absent.
# ---------------------------------------------------------------------------
try:
    import nltk
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score as _nltk_meteor

    # Silently download required corpora if not present
    for _corpus in ("wordnet", "averaged_perceptron_tagger", "punkt", "omw-1.4"):
        try:
            nltk.data.find(f"tokenizers/{_corpus}")
        except LookupError:
            try:
                nltk.download(_corpus, quiet=True)
            except Exception:
                pass
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

try:
    from rouge_score import rouge_scorer as _rouge_scorer_lib

    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False

try:
    from bert_score import score as _bert_score_fn  # type: ignore

    _BERTSCORE_AVAILABLE = True
except ImportError:
    _BERTSCORE_AVAILABLE = False

try:
    import pandas as pd  # type: ignore

    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

# Project-local imports – tolerated to be absent during isolated unit tests
try:
    from config import CONTRASTIVE_MODELS, TASK_METRICS, VLMConfig  # type: ignore
except ImportError:
    VLMConfig = None  # type: ignore
    TASK_METRICS: Dict[str, List[str]] = {
        "captioning": ["bleu1", "bleu2", "bleu3", "bleu4", "rouge_l", "meteor", "cider"],
        "vqa": ["exact_match", "f1", "vqa_accuracy"],
        "retrieval": ["recall_at_1", "recall_at_5", "recall_at_10", "map", "mrr"],
        "classification": ["accuracy", "macro_f1", "weighted_f1"],
        "chat": ["rouge_l", "bertscore_f1"],
    }
    CONTRASTIVE_MODELS: List[str] = ["clip", "align", "blip", "flava"]

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# ── Text normalisation ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

_ARTICLES: frozenset = frozenset({"a", "an", "the"})
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, remove articles, collapse whitespace.

    This mirrors the normalisation used by the official VQA evaluation script
    so that surface-form differences do not penalise semantically correct
    answers.

    Args:
        s: Raw answer / prediction string.

    Returns:
        Normalised string.
    """
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    tokens = s.split()
    tokens = [t for t in tokens if t not in _ARTICLES]
    return " ".join(tokens).strip()


# ---------------------------------------------------------------------------
# ── BLEU ───────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def compute_bleu(
    references: List[str],
    hypotheses: List[str],
) -> Dict[str, float]:
    """Corpus-level BLEU-1 through BLEU-4 using NLTK.

    Args:
        references:  List of reference strings (one per sample).
        hypotheses:  List of hypothesis strings (one per sample).

    Returns:
        Dict with keys ``bleu1``, ``bleu2``, ``bleu3``, ``bleu4``.

    Raises:
        ImportError: If NLTK is not installed.
    """
    if not _NLTK_AVAILABLE:
        raise ImportError("nltk is required for BLEU computation. Run: pip install nltk")

    if len(references) != len(hypotheses):
        raise ValueError(
            f"Length mismatch: {len(references)} references vs {len(hypotheses)} hypotheses"
        )

    # NLTK corpus_bleu expects: list[list[list[str]]], list[list[str]]
    tokenised_refs = [[ref.lower().split()] for ref in references]
    tokenised_hyps = [hyp.lower().split() for hyp in hypotheses]

    smooth = SmoothingFunction().method1

    bleu_scores: Dict[str, float] = {}
    for n in range(1, 5):
        weights = tuple(1.0 / n if i < n else 0.0 for i in range(4))
        score = corpus_bleu(
            tokenised_refs,
            tokenised_hyps,
            weights=weights,
            smoothing_function=smooth,
        )
        bleu_scores[f"bleu{n}"] = round(float(score), 6)

    return bleu_scores


# ---------------------------------------------------------------------------
# ── ROUGE ──────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def compute_rouge(
    references: List[str],
    hypotheses: List[str],
) -> Dict[str, float]:
    """Corpus-averaged ROUGE-1, ROUGE-2 and ROUGE-L F1 scores.

    Args:
        references:  List of reference strings.
        hypotheses:  List of hypothesis strings.

    Returns:
        Dict with keys ``rouge1``, ``rouge2``, ``rouge_l``.

    Raises:
        ImportError: If rouge_score is not installed.
    """
    if not _ROUGE_AVAILABLE:
        raise ImportError(
            "rouge_score is required. Run: pip install rouge-score"
        )

    scorer = _rouge_scorer_lib.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    agg: Dict[str, List[float]] = {"rouge1": [], "rouge2": [], "rougeL": []}
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        agg["rouge1"].append(scores["rouge1"].fmeasure)
        agg["rouge2"].append(scores["rouge2"].fmeasure)
        agg["rougeL"].append(scores["rougeL"].fmeasure)

    return {
        "rouge1": round(float(np.mean(agg["rouge1"])), 6),
        "rouge2": round(float(np.mean(agg["rouge2"])), 6),
        "rouge_l": round(float(np.mean(agg["rougeL"])), 6),
    }


# ---------------------------------------------------------------------------
# ── METEOR ─────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def compute_meteor(
    references: List[str],
    hypotheses: List[str],
) -> float:
    """Corpus-averaged METEOR score.

    Args:
        references:  List of reference strings.
        hypotheses:  List of hypothesis strings.

    Returns:
        Average METEOR score in [0, 1].

    Raises:
        ImportError: If NLTK is not installed.
    """
    if not _NLTK_AVAILABLE:
        raise ImportError("nltk is required for METEOR. Run: pip install nltk")

    scores: List[float] = []
    for ref, hyp in zip(references, hypotheses):
        # NLTK METEOR expects tokenised lists
        score = _nltk_meteor([ref.lower().split()], hyp.lower().split())
        scores.append(float(score))

    return round(float(np.mean(scores)) if scores else 0.0, 6)


# ---------------------------------------------------------------------------
# ── CIDEr-D (self-contained TF-IDF implementation) ─────────────────────────
# ---------------------------------------------------------------------------


class _CiderScorer:
    """TF-IDF weighted CIDEr-D scorer (no external dependencies).

    Reference
    ---------
    Vedantam, R., Zitnick, C. L., & Parikh, D. (2015).
    CIDEr: Consensus-based Image Description Evaluation. CVPR.
    """

    def __init__(self, n: int = 4, sigma: float = 6.0) -> None:
        self.n = n          # maximum n-gram order
        self.sigma = sigma  # Gaussian length penalty σ

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenise(sentence: str) -> List[str]:
        return sentence.lower().split()

    def _get_ngrams(self, tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        ngrams: Dict[Tuple[str, ...], int] = collections.defaultdict(int)
        for i in range(len(tokens) - n + 1):
            ngrams[tuple(tokens[i : i + n])] += 1
        return dict(ngrams)

    def _compute_doc_freq(
        self,
        references: List[List[str]],  # list of reference sets
        n: int,
    ) -> Dict[Tuple[str, ...], int]:
        """Document frequency: how many *reference sentences* contain an n-gram."""
        df: Dict[Tuple[str, ...], int] = collections.defaultdict(int)
        for ref_set in references:
            # Count each n-gram at most once per reference sentence
            seen: set = set()
            for ref in ref_set:
                tokens = self._tokenise(ref)
                ngrams = self._get_ngrams(tokens, n)
                for ng in ngrams:
                    if ng not in seen:
                        df[ng] += 1
                        seen.add(ng)
        return dict(df)

    def _tf_idf_vector(
        self,
        ngrams: Dict[Tuple[str, ...], int],
        df: Dict[Tuple[str, ...], int],
        n_docs: int,
        n: int,
    ) -> Dict[Tuple[str, ...], float]:
        """Compute TF-IDF weighted vector for one sentence's n-grams."""
        total = sum(ngrams.values()) or 1
        vec: Dict[Tuple[str, ...], float] = {}
        for ng, cnt in ngrams.items():
            tf = cnt / total
            idf = math.log((n_docs + 1.0) / (df.get(ng, 0) + 1.0))
            vec[ng] = tf * idf
        return vec

    @staticmethod
    def _cosine_sim(
        v1: Dict[Tuple[str, ...], float],
        v2: Dict[Tuple[str, ...], float],
    ) -> float:
        dot = sum(v1.get(ng, 0.0) * v2.get(ng, 0.0) for ng in v2)
        norm1 = math.sqrt(sum(x * x for x in v1.values())) or 1e-9
        norm2 = math.sqrt(sum(x * x for x in v2.values())) or 1e-9
        return dot / (norm1 * norm2)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def score(
        self,
        references: List[List[str]],
        hypotheses: List[str],
    ) -> float:
        """Return the corpus CIDEr-D score.

        Args:
            references:  List of reference sets.  Each element is a list of
                         human-written reference captions for one image.
            hypotheses:  One generated caption per image.

        Returns:
            CIDEr-D score (scaled to [0, 10] range, same as original paper).
        """
        n_docs = len(references)
        cider_sum = 0.0

        for order in range(1, self.n + 1):
            df = self._compute_doc_freq(references, order)

            order_scores: List[float] = []
            for ref_set, hyp in zip(references, hypotheses):
                hyp_tokens = self._tokenise(hyp)
                hyp_ngrams = self._get_ngrams(hyp_tokens, order)
                if not hyp_ngrams:
                    order_scores.append(0.0)
                    continue

                hyp_vec = self._tf_idf_vector(hyp_ngrams, df, n_docs, order)

                sims: List[float] = []
                for ref in ref_set:
                    ref_tokens = self._tokenise(ref)
                    ref_ngrams = self._get_ngrams(ref_tokens, order)
                    ref_vec = self._tf_idf_vector(ref_ngrams, df, n_docs, order)

                    sim = self._cosine_sim(hyp_vec, ref_vec)

                    # Gaussian length penalty
                    len_diff = len(hyp_tokens) - len(ref_tokens)
                    penalty = math.exp(-(len_diff**2) / (2 * self.sigma**2))
                    sims.append(sim * penalty)

                order_scores.append(max(sims) if sims else 0.0)

            cider_sum += np.mean(order_scores) if order_scores else 0.0

        # Average over n-gram orders and scale to [0, 10]
        return round(float((cider_sum / self.n) * 10.0), 6)


def compute_cider(
    references: List[List[str]],
    hypotheses: List[str],
) -> float:
    """CIDEr-D corpus score.

    Args:
        references:  List of reference sets.  ``references[i]`` is a list of
                     human reference captions for sample ``i``.
        hypotheses:  Generated caption for each sample.

    Returns:
        Corpus CIDEr-D score (range 0–10).
    """
    scorer = _CiderScorer(n=4, sigma=6.0)
    return scorer.score(references, hypotheses)


# ---------------------------------------------------------------------------
# ── VQA Accuracy (soft scoring) ────────────────────────────────────────────
# ---------------------------------------------------------------------------


def compute_vqa_accuracy(
    predictions: List[str],
    answers: List[List[str]],
) -> float:
    """VQA v2-style soft accuracy.

    A prediction is credited *min(1, n_matching_annotators / 3)* where
    n_matching_annotators is the number of the 10 human annotators who gave
    the same answer as the model (after normalisation).

    Args:
        predictions: Generated answers.
        answers:     List of annotator answer lists (up to 10 per sample).

    Returns:
        Mean VQA accuracy in [0, 1].
    """
    if len(predictions) != len(answers):
        raise ValueError("predictions and answers must have the same length.")

    scores: List[float] = []
    for pred, ans_list in zip(predictions, answers):
        norm_pred = normalize_answer(pred)
        norm_ans = [normalize_answer(a) for a in ans_list]
        matches = sum(1 for a in norm_ans if a == norm_pred)
        scores.append(min(1.0, matches / 3.0))

    return round(float(np.mean(scores)) if scores else 0.0, 6)


# ---------------------------------------------------------------------------
# ── Exact Match ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def compute_exact_match(
    predictions: List[str],
    references: List[str],
) -> float:
    """Fraction of predictions that exactly match their reference (after normalisation).

    Args:
        predictions: Model outputs.
        references:  Ground-truth strings.

    Returns:
        Exact match ratio in [0, 1].
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length.")

    matches = sum(
        1 for p, r in zip(predictions, references)
        if normalize_answer(p) == normalize_answer(r)
    )
    return round(matches / max(len(predictions), 1), 6)


# ---------------------------------------------------------------------------
# ── Token-level F1 (SQuAD-style) ───────────────────────────────────────────
# ---------------------------------------------------------------------------


def compute_f1_token(prediction: str, reference: str) -> float:
    """Token-level F1 score between two strings.

    Mirrors the SQuAD / VQA evaluation methodology.

    Args:
        prediction: Predicted answer string.
        reference:  Ground-truth answer string.

    Returns:
        Token-level F1 in [0, 1].
    """
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    pred_counter: collections.Counter = collections.Counter(pred_tokens)
    ref_counter: collections.Counter = collections.Counter(ref_tokens)

    common = pred_counter & ref_counter
    n_common = sum(common.values())

    if n_common == 0:
        return 0.0

    precision = n_common / len(pred_tokens)
    recall = n_common / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)


def _corpus_f1_token(
    predictions: List[str],
    references: List[str],
) -> float:
    """Average token-level F1 over a corpus.  Uses *best* reference per sample."""
    scores = [
        compute_f1_token(p, r) for p, r in zip(predictions, references)
    ]
    return round(float(np.mean(scores)) if scores else 0.0, 6)


# ---------------------------------------------------------------------------
# ── CLIP-style retrieval metrics ───────────────────────────────────────────
# ---------------------------------------------------------------------------


def compute_clip_retrieval_metrics(
    image_embeds: Union[np.ndarray, "torch.Tensor"],
    text_embeds: Union[np.ndarray, "torch.Tensor"],
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """Compute image-to-text and text-to-image retrieval metrics.

    Assumes that ``image_embeds[i]`` and ``text_embeds[i]`` are paired
    (i.e. they correspond to the same sample).  Embeddings should already
    be L2-normalised.

    Args:
        image_embeds: (N, D) array of image embeddings.
        text_embeds:  (N, D) array of text embeddings.
        ks:           Values of k for Recall@k.

    Returns:
        Dict containing per-direction and average metrics:
        ``i2t_recall@k``, ``t2i_recall@k``, ``i2t_mrr``, ``t2i_mrr``,
        ``i2t_map``, ``t2i_map``.
    """
    if isinstance(image_embeds, torch.Tensor):
        image_embeds = image_embeds.cpu().float().numpy()
    if isinstance(text_embeds, torch.Tensor):
        text_embeds = text_embeds.cpu().float().numpy()

    image_embeds = np.array(image_embeds, dtype=np.float32)
    text_embeds = np.array(text_embeds, dtype=np.float32)

    n = image_embeds.shape[0]
    assert text_embeds.shape[0] == n, "Embed arrays must have the same first dimension."

    # Compute full similarity matrix  (N, N)
    sim_matrix = image_embeds @ text_embeds.T  # (N, N)

    metrics: Dict[str, float] = {}

    def _recall_at_k(sim: np.ndarray, k: int) -> float:
        """For each query (row), check if the diagonal is in top-k."""
        top_k_indices = np.argsort(-sim, axis=1)[:, :k]
        hits = sum(i in top_k_indices[i] for i in range(n))
        return hits / n

    def _mrr(sim: np.ndarray) -> float:
        """Mean Reciprocal Rank – rank of the correct item (diagonal)."""
        ranks = []
        for i in range(n):
            sorted_ids = np.argsort(-sim[i])
            rank = np.where(sorted_ids == i)[0][0] + 1  # 1-indexed
            ranks.append(1.0 / rank)
        return float(np.mean(ranks))

    def _map_at_r(sim: np.ndarray) -> float:
        """mAP@R with R=1 (single positive per query), equals Recall@1."""
        # For single-positive retrieval, AP = 1/rank_of_positive
        aps = []
        for i in range(n):
            sorted_ids = np.argsort(-sim[i])
            rank = np.where(sorted_ids == i)[0][0] + 1
            aps.append(1.0 / rank)
        return float(np.mean(aps))

    # Image-to-text  (rows = image queries, columns = text gallery)
    for k in ks:
        metrics[f"i2t_recall@{k}"] = round(_recall_at_k(sim_matrix, k), 6)

    metrics["i2t_mrr"] = round(_mrr(sim_matrix), 6)
    metrics["i2t_map"] = round(_map_at_r(sim_matrix), 6)

    # Text-to-image  (transpose the matrix)
    sim_t2i = sim_matrix.T
    for k in ks:
        metrics[f"t2i_recall@{k}"] = round(_recall_at_k(sim_t2i, k), 6)

    metrics["t2i_mrr"] = round(_mrr(sim_t2i), 6)
    metrics["t2i_map"] = round(_map_at_r(sim_t2i), 6)

    # Averages across directions
    for k in ks:
        metrics[f"mean_recall@{k}"] = round(
            (metrics[f"i2t_recall@{k}"] + metrics[f"t2i_recall@{k}"]) / 2.0, 6
        )

    metrics["mean_mrr"] = round((metrics["i2t_mrr"] + metrics["t2i_mrr"]) / 2.0, 6)
    metrics["mean_map"] = round((metrics["i2t_map"] + metrics["t2i_map"]) / 2.0, 6)

    return metrics


# ---------------------------------------------------------------------------
# ── BERTScore ──────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def compute_bertscore(
    references: List[str],
    hypotheses: List[str],
    model_type: str = "distilbert-base-uncased",
    device: Optional[str] = None,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Compute BERTScore P/R/F1.

    Falls back to an all-zeros dict if bert_score is not installed so that
    downstream code can always consume the result.

    Args:
        references:  Reference strings.
        hypotheses:  Hypothesis strings.
        model_type:  HuggingFace model to use for embeddings.
        device:      Torch device string (autodetected if None).
        batch_size:  Batch size for BERTScore computation.

    Returns:
        Dict with keys ``bertscore_precision``, ``bertscore_recall``,
        ``bertscore_f1``.  Values are corpus averages in [0, 1].
    """
    zero = {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
    if not _BERTSCORE_AVAILABLE:
        logger.warning("bert_score not installed; returning zeros. pip install bert-score")
        return zero

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        P, R, F = _bert_score_fn(
            hypotheses,
            references,
            model_type=model_type,
            device=device,
            batch_size=batch_size,
            verbose=False,
        )
        return {
            "bertscore_precision": round(float(P.mean()), 6),
            "bertscore_recall": round(float(R.mean()), 6),
            "bertscore_f1": round(float(F.mean()), 6),
        }
    except Exception as exc:
        logger.warning("BERTScore computation failed: %s", exc)
        return zero


# ---------------------------------------------------------------------------
# ── Classification metrics (pure numpy, no sklearn) ─────────────────────────
# ---------------------------------------------------------------------------


def _classification_metrics(
    predictions: List[Any],
    references: List[Any],
) -> Dict[str, Any]:
    """Accuracy, macro/weighted F1 and per-class P/R/F1.

    Implemented without sklearn so the only hard dependency is numpy.

    Args:
        predictions: Predicted class labels (int or str).
        references:  Ground-truth class labels.

    Returns:
        Dict with ``accuracy``, ``macro_f1``, ``weighted_f1``,
        and ``per_class`` (dict keyed by label).
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length.")

    labels = sorted(set(references) | set(predictions), key=str)
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    n_classes = len(labels)
    n_samples = len(predictions)

    # Confusion-matrix tallies
    tp = np.zeros(n_classes, dtype=np.int64)
    fp = np.zeros(n_classes, dtype=np.int64)
    fn = np.zeros(n_classes, dtype=np.int64)
    n_correct = 0

    for pred, ref in zip(predictions, references):
        pi = label_to_idx[pred]
        ri = label_to_idx[ref]
        if pi == ri:
            tp[pi] += 1
            n_correct += 1
        else:
            fp[pi] += 1
            fn[ri] += 1

    accuracy = n_correct / max(n_samples, 1)

    precision_per = np.where(
        (tp + fp) > 0, tp / (tp + fp).astype(float), 0.0
    )
    recall_per = np.where(
        (tp + fn) > 0, tp / (tp + fn).astype(float), 0.0
    )
    f1_per = np.where(
        (precision_per + recall_per) > 0,
        2 * precision_per * recall_per / (precision_per + recall_per),
        0.0,
    )

    # Support per class (number of ground-truth instances)
    support = tp + fn
    total_support = support.sum()

    macro_f1 = float(np.mean(f1_per))
    weighted_f1 = float(
        np.sum(f1_per * support) / max(total_support, 1)
    )

    per_class: Dict[str, Dict[str, float]] = {}
    for i, lbl in enumerate(labels):
        per_class[str(lbl)] = {
            "precision": round(float(precision_per[i]), 6),
            "recall": round(float(recall_per[i]), 6),
            "f1": round(float(f1_per[i]), 6),
            "support": int(support[i]),
        }

    return {
        "accuracy": round(accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# ── Evaluator class ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class Evaluator:
    """Unified evaluation harness for VLM tasks.

    Parameters
    ----------
    model:
        A HuggingFace-compatible VLM model (``generate`` interface).
        For retrieval tasks the model must expose an ``encode_image`` /
        ``encode_text`` API or produce logits suitable for cosine similarity.
    processor:
        Corresponding processor / tokenizer.
    config:
        A ``VLMConfig`` instance (or any object with the required attributes).
        Recognised attributes:

        - ``task``           (str)  – default task type
        - ``max_new_tokens`` (int)  – generation length cap  [default 128]
        - ``num_beams``      (int)  – beam search width      [default 4]
        - ``device``         (str)  – torch device string
        - ``bertscore_model``(str)  – model for BERTScore    [optional]
    """

    # Maps task name → evaluation method
    _TASK_DISPATCH: Dict[str, str] = {
        "captioning": "evaluate_captioning",
        "vqa": "evaluate_vqa",
        "retrieval": "evaluate_retrieval",
        "classification": "evaluate_classification",
        "chat": "evaluate_chat",
    }

    def __init__(
        self,
        model: Any,
        processor: Any,
        config: Any,
    ) -> None:
        self.model = model
        self.processor = processor
        self.config = config

        self.device: str = getattr(config, "device", None) or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.max_new_tokens: int = getattr(config, "max_new_tokens", 128)
        self.num_beams: int = getattr(config, "num_beams", 4)
        self.task: str = getattr(config, "task", "captioning").lower()
        self.bertscore_model: str = getattr(
            config, "bertscore_model", "distilbert-base-uncased"
        )

        # Move model to device if not already there
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        self.model.eval()

        logger.info(
            "Evaluator initialised | task=%s | device=%s | max_new_tokens=%d",
            self.task,
            self.device,
            self.max_new_tokens,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataloader: Any,
        split: str = "test",
    ) -> Dict[str, Any]:
        """Main evaluation entry point.  Routes to the appropriate method.

        Args:
            dataloader: PyTorch DataLoader yielding task-appropriate batches.
            split:      Dataset split label ('val', 'test', …) – used for
                        logging and report naming only.

        Returns:
            Metrics dict appropriate for the configured task.
        """
        task = self.task
        method_name = self._TASK_DISPATCH.get(task)
        if method_name is None:
            supported = list(self._TASK_DISPATCH.keys())
            raise ValueError(
                f"Unknown task '{task}'. Supported tasks: {supported}"
            )

        logger.info("Starting evaluation | split=%s | task=%s", split, task)
        method = getattr(self, method_name)
        metrics = method(dataloader)
        metrics["split"] = split
        metrics["task"] = task
        logger.info("Evaluation complete | %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Task-specific evaluators
    # ------------------------------------------------------------------

    def evaluate_captioning(self, dataloader: Any) -> Dict[str, float]:
        """Evaluate image captioning with BLEU-1/2/3/4, ROUGE-L, METEOR, CIDEr.

        Expected batch keys
        -------------------
        ``pixel_values`` or ``image``    – processed images
        ``labels`` or ``captions``       – list/tensor of reference captions
        ``input_ids`` (optional)         – prompt tokens

        Returns:
            Dict with keys: bleu1, bleu2, bleu3, bleu4, rouge1, rouge2,
            rouge_l, meteor, cider.
        """
        predictions_raw, references_raw, references_multi = [], [], []

        for batch in tqdm(dataloader, desc="[captioning]", leave=False):
            texts, refs, refs_multi = self._captioning_batch(batch)
            predictions_raw.extend(texts)
            references_raw.extend(refs)
            references_multi.extend(refs_multi)

        metrics: Dict[str, float] = {}

        # BLEU
        bleu = compute_bleu(references_raw, predictions_raw)
        metrics.update(bleu)

        # ROUGE
        rouge = compute_rouge(references_raw, predictions_raw)
        metrics.update(rouge)

        # METEOR
        metrics["meteor"] = compute_meteor(references_raw, predictions_raw)

        # CIDEr
        metrics["cider"] = compute_cider(references_multi, predictions_raw)

        return metrics

    def evaluate_vqa(self, dataloader: Any) -> Dict[str, float]:
        """Evaluate VQA: Exact Match, token-F1, VQA Accuracy (soft scoring).

        Expected batch keys
        -------------------
        ``pixel_values`` or ``image``
        ``input_ids`` / ``question`` – encoded question
        ``answers``                  – list of annotator answers per sample

        Returns:
            Dict with keys: exact_match, f1, vqa_accuracy.
        """
        all_preds: List[str] = []
        all_refs: List[str] = []
        all_answer_lists: List[List[str]] = []

        for batch in tqdm(dataloader, desc="[vqa]", leave=False):
            preds = self._generate_text(batch)
            answers = self._extract_field(batch, ("answers", "answer_list", "labels"))
            refs = self._primary_reference(answers)

            all_preds.extend(preds)
            all_refs.extend(refs)
            if isinstance(answers[0], (list, tuple)):
                all_answer_lists.extend(answers)
            else:
                all_answer_lists.extend([[a] for a in answers])

        metrics: Dict[str, float] = {
            "exact_match": compute_exact_match(all_preds, all_refs),
            "f1": _corpus_f1_token(all_preds, all_refs),
            "vqa_accuracy": compute_vqa_accuracy(all_preds, all_answer_lists),
        }
        return metrics

    def evaluate_retrieval(self, dataloader: Any) -> Dict[str, float]:
        """CLIP-style retrieval: Recall@1/5/10, mAP, MRR.

        The model must expose ``encode_image`` and ``encode_text`` methods
        (or its forward pass must return ``image_embeds`` / ``text_embeds``
        when called with the appropriate inputs).

        Expected batch keys
        -------------------
        ``pixel_values`` – processed images
        ``input_ids``    – tokenised captions
        ``attention_mask`` (optional)

        Returns:
            Dict with i2t/t2i and mean Recall@K, MRR, mAP.
        """
        all_image_embeds: List[np.ndarray] = []
        all_text_embeds: List[np.ndarray] = []

        for batch in tqdm(dataloader, desc="[retrieval]", leave=False):
            img_emb, txt_emb = self._embed_batch(batch)
            all_image_embeds.append(img_emb)
            all_text_embeds.append(txt_emb)

        image_embeds = np.concatenate(all_image_embeds, axis=0)
        text_embeds = np.concatenate(all_text_embeds, axis=0)

        # L2-normalise if not already done
        image_embeds /= np.linalg.norm(image_embeds, axis=1, keepdims=True) + 1e-9
        text_embeds /= np.linalg.norm(text_embeds, axis=1, keepdims=True) + 1e-9

        return compute_clip_retrieval_metrics(image_embeds, text_embeds)

    def evaluate_classification(self, dataloader: Any) -> Dict[str, Any]:
        """Evaluate zero-shot or fine-tuned classification.

        Expected batch keys
        -------------------
        ``pixel_values`` or ``image``
        ``labels``  – integer or string class labels

        Returns:
            Dict with accuracy, macro_f1, weighted_f1, per_class.
        """
        all_preds: List[Any] = []
        all_refs: List[Any] = []

        for batch in tqdm(dataloader, desc="[classification]", leave=False):
            preds = self._classify_batch(batch)
            refs = self._extract_field(batch, ("labels", "label", "class_id"))
            all_preds.extend(preds)
            all_refs.extend(refs if not isinstance(refs, torch.Tensor) else refs.tolist())

        return _classification_metrics(all_preds, all_refs)

    def evaluate_chat(self, dataloader: Any) -> Dict[str, float]:
        """Evaluate conversational / instruction-following output.

        Metrics: ROUGE-L, BERTScore (if bert_score installed).

        Expected batch keys
        -------------------
        ``pixel_values`` (optional)
        ``input_ids``    – the dialogue context
        ``labels`` or ``response`` – ground-truth response

        Returns:
            Dict with rouge_l, (optionally) bertscore_precision/recall/f1.
        """
        all_preds: List[str] = []
        all_refs: List[str] = []

        for batch in tqdm(dataloader, desc="[chat]", leave=False):
            preds = self._generate_text(batch)
            refs = self._primary_reference(
                self._extract_field(batch, ("labels", "response", "target"))
            )
            all_preds.extend(preds)
            all_refs.extend(refs)

        rouge = compute_rouge(all_refs, all_preds)
        metrics: Dict[str, float] = {"rouge_l": rouge["rouge_l"]}

        bs = compute_bertscore(all_refs, all_preds, model_type=self.bertscore_model)
        metrics.update(bs)

        return metrics

    # ------------------------------------------------------------------
    # Prediction generation
    # ------------------------------------------------------------------

    def generate_predictions(self, dataloader: Any) -> List[Dict[str, Any]]:
        """Generate predictions for every sample in ``dataloader``.

        Returns a list of dicts, each containing at minimum:
        ``{"sample_id": ..., "prediction": ..., "reference": ...}``.

        Args:
            dataloader: Any iterable of batches.

        Returns:
            List of prediction records.
        """
        records: List[Dict[str, Any]] = []
        global_idx = 0

        for batch in tqdm(dataloader, desc="[generate]", leave=False):
            preds = self._generate_text(batch)
            refs_raw = self._extract_field(
                batch,
                ("labels", "captions", "answers", "response", "target"),
                default=[""] * len(preds),
            )
            refs = self._primary_reference(refs_raw)
            ids = self._extract_field(
                batch,
                ("sample_id", "id", "image_id"),
                default=list(range(global_idx, global_idx + len(preds))),
            )

            for sid, pred, ref in zip(ids, preds, refs):
                records.append(
                    {
                        "sample_id": sid if not isinstance(sid, torch.Tensor) else sid.item(),
                        "prediction": pred,
                        "reference": ref,
                    }
                )
            global_idx += len(preds)

        return records

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def save_report(
        self,
        metrics: Dict[str, Any],
        output_path: str,
    ) -> None:
        """Persist metrics as JSON and a human-readable .txt summary.

        Args:
            metrics:     Metrics dict as returned by any ``evaluate_*`` method.
            output_path: Path prefix (without extension).  Two files will be
                         written: ``<output_path>.json`` and
                         ``<output_path>.txt``.
        """
        output_path = str(output_path)
        if output_path.endswith(".json"):
            output_path = output_path[:-5]

        json_path = output_path + ".json"
        txt_path = output_path + ".txt"

        Path(json_path).parent.mkdir(parents=True, exist_ok=True)

        # JSON dump
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, default=str)
        logger.info("Metrics JSON saved to %s", json_path)

        # Human-readable summary
        lines: List[str] = [
            "=" * 60,
            " VLM Evaluation Report",
            "=" * 60,
            f"  Task  : {metrics.get('task', 'unknown')}",
            f"  Split : {metrics.get('split', 'unknown')}",
            "-" * 60,
        ]
        self._format_metrics_txt(metrics, lines, indent=2)
        lines.append("=" * 60)
        report_text = "\n".join(lines) + "\n"

        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(report_text)

        logger.info("Human-readable report saved to %s", txt_path)
        print(report_text)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_text(self, batch: Dict[str, Any]) -> List[str]:
        """Run ``model.generate`` and decode output tokens."""
        inputs = self._prepare_inputs(batch)
        # Remove labels/decoder inputs before generation
        inputs.pop("labels", None)
        inputs.pop("decoder_input_ids", None)

        # Determine the prompt length to skip in decoding
        input_ids = inputs.get("input_ids")
        prompt_len = input_ids.shape[1] if input_ids is not None else 0

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            early_stopping=True,
        )

        # Decode only newly generated tokens
        generated = output_ids[:, prompt_len:]
        decoded = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )
        return [d.strip() for d in decoded]

    @torch.no_grad()
    def _embed_batch(
        self, batch: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract L2-normalised image and text embeddings from a batch."""
        inputs = self._prepare_inputs(batch)

        # Prefer explicit encode_image / encode_text if available
        if hasattr(self.model, "encode_image") and hasattr(self.model, "encode_text"):
            img_emb = self.model.encode_image(inputs.get("pixel_values"))
            txt_emb = self.model.encode_text(
                inputs.get("input_ids"), inputs.get("attention_mask")
            )
        else:
            # Fallback: forward pass returning image_embeds & text_embeds
            outputs = self.model(**inputs)
            img_emb = getattr(outputs, "image_embeds", None)
            txt_emb = getattr(outputs, "text_embeds", None)
            if img_emb is None or txt_emb is None:
                raise AttributeError(
                    "Model does not expose image_embeds/text_embeds. "
                    "Implement encode_image/encode_text or use a CLIP-compatible model."
                )

        img_np = img_emb.cpu().float().numpy()
        txt_np = txt_emb.cpu().float().numpy()
        return img_np, txt_np

    @torch.no_grad()
    def _classify_batch(self, batch: Dict[str, Any]) -> List[Any]:
        """Return predicted class labels for a classification batch."""
        inputs = self._prepare_inputs(batch)
        inputs.pop("labels", None)

        outputs = self.model(**inputs)

        # Handle logits
        logits = getattr(outputs, "logits", None)
        if logits is not None:
            preds = logits.argmax(dim=-1).cpu().tolist()
        else:
            # Fallback: use generation
            preds_text = self._generate_text(batch)
            preds = [normalize_answer(p) for p in preds_text]

        return preds if isinstance(preds, list) else list(preds)

    def _captioning_batch(
        self, batch: Dict[str, Any]
    ) -> Tuple[List[str], List[str], List[List[str]]]:
        """Generate captions and extract references from a captioning batch."""
        preds = self._generate_text(batch)
        captions = self._extract_field(
            batch,
            ("captions", "labels", "caption", "target"),
            default=[[""] for _ in preds],
        )

        # captions may be:  List[str]  or  List[List[str]]
        refs_primary: List[str] = []
        refs_multi: List[List[str]] = []
        for cap in captions:
            if isinstance(cap, (list, tuple)):
                refs_primary.append(cap[0] if cap else "")
                refs_multi.append([str(c) for c in cap])
            elif isinstance(cap, torch.Tensor):
                decoded = self.processor.decode(cap, skip_special_tokens=True).strip()
                refs_primary.append(decoded)
                refs_multi.append([decoded])
            else:
                refs_primary.append(str(cap))
                refs_multi.append([str(cap)])

        return preds, refs_primary, refs_multi

    def _prepare_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensor values in a batch to the model device."""
        processed: Dict[str, Any] = {}
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                processed[key] = val.to(self.device)
            else:
                processed[key] = val
        return processed

    @staticmethod
    def _extract_field(
        batch: Dict[str, Any],
        candidates: Union[Tuple[str, ...], List[str]],
        default: Any = None,
    ) -> Any:
        """Return the first present field from ``candidates`` in ``batch``."""
        for key in candidates:
            if key in batch:
                val = batch[key]
                if isinstance(val, torch.Tensor):
                    return val.tolist()
                return val
        return default if default is not None else []

    @staticmethod
    def _primary_reference(refs: Any) -> List[str]:
        """Extract a single reference string per sample from various formats."""
        result: List[str] = []
        if refs is None:
            return result
        for r in refs:
            if isinstance(r, (list, tuple)):
                result.append(str(r[0]) if r else "")
            elif isinstance(r, torch.Tensor):
                result.append(str(r.item()))
            else:
                result.append(str(r))
        return result

    @staticmethod
    def _format_metrics_txt(
        metrics: Dict[str, Any],
        lines: List[str],
        indent: int = 0,
    ) -> None:
        """Recursively format a metrics dict into human-readable lines."""
        pad = " " * indent
        for key, val in metrics.items():
            if key in ("task", "split"):
                continue
            if isinstance(val, dict):
                lines.append(f"{pad}{key}:")
                Evaluator._format_metrics_txt(val, lines, indent + 4)
            elif isinstance(val, float):
                lines.append(f"{pad}{key:<30s}: {val:.4f}")
            else:
                lines.append(f"{pad}{key:<30s}: {val}")


# ---------------------------------------------------------------------------
# ── Checkpoint utilities ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def evaluate_checkpoint(
    checkpoint_path: str,
    config: Any,
    dataloader: Any,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a model checkpoint, run full evaluation, and save report.

    Args:
        checkpoint_path: Path to a saved HuggingFace model directory or
                         a ``state_dict`` ``.pt`` / ``.bin`` file.
        config:          VLMConfig (or compatible) instance.
        dataloader:      Evaluation dataloader.
        output_dir:      Directory where the report is written.
                         Defaults to ``checkpoint_path/../eval_reports/``.

    Returns:
        Metrics dict.
    """
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers is required. pip install transformers")

    logger.info("Loading checkpoint from %s", checkpoint_path)
    cp = Path(checkpoint_path)

    device = getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu")

    # Determine load strategy
    if cp.is_dir():
        # HuggingFace model directory
        model = AutoModelForVision2Seq.from_pretrained(str(cp))
        try:
            from transformers import AutoProcessor as _AP

            processor = _AP.from_pretrained(str(cp))
        except Exception:
            model_name = getattr(config, "model_name", "Salesforce/blip2-opt-2.7b")
            from transformers import AutoProcessor as _AP

            processor = _AP.from_pretrained(model_name)
    elif cp.suffix in (".pt", ".bin", ".pth"):
        # Raw state-dict
        model_name = getattr(config, "model_name", None)
        if model_name is None:
            raise ValueError(
                "config.model_name must be set when loading a raw state-dict checkpoint."
            )
        model = AutoModelForVision2Seq.from_pretrained(model_name)
        state_dict = torch.load(str(cp), map_location="cpu")
        # Unwrap common wrappers
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        from transformers import AutoProcessor as _AP

        processor = _AP.from_pretrained(model_name)
    else:
        raise ValueError(
            f"Unrecognised checkpoint format: {cp}. "
            "Expected a directory or a .pt/.bin/.pth file."
        )

    evaluator = Evaluator(model, processor, config)
    split = getattr(config, "eval_split", "test")
    metrics = evaluator.evaluate(dataloader, split=split)
    metrics["checkpoint"] = str(checkpoint_path)

    # Save report
    if output_dir is None:
        output_dir = str(cp.parent / "eval_reports")
    report_name = cp.stem if cp.is_file() else cp.name
    report_path = Path(output_dir) / report_name
    evaluator.save_report(metrics, str(report_path))

    return metrics


def compare_checkpoints(
    checkpoint_paths: List[str],
    config: Any,
    dataloader: Any,
    output_dir: Optional[str] = None,
) -> Any:  # pd.DataFrame | List[Dict]
    """Evaluate multiple checkpoints and return a comparison table.

    Args:
        checkpoint_paths: Ordered list of checkpoint paths.
        config:           Shared VLMConfig for all checkpoints.
        dataloader:       Evaluation dataloader (iterated once per checkpoint).
        output_dir:       Directory for individual reports and the summary CSV.

    Returns:
        ``pandas.DataFrame`` if pandas is available, else a list of metric
        dicts.  Columns are metric names; rows are checkpoints.
    """
    rows: List[Dict[str, Any]] = []

    for ckpt_path in checkpoint_paths:
        logger.info("Comparing checkpoint: %s", ckpt_path)
        try:
            metrics = evaluate_checkpoint(ckpt_path, config, dataloader, output_dir)
        except Exception as exc:
            logger.error("Failed to evaluate %s: %s", ckpt_path, exc)
            metrics = {"checkpoint": ckpt_path, "error": str(exc)}
        rows.append(metrics)

    if output_dir is not None:
        summary_path = Path(output_dir) / "checkpoint_comparison.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2, default=str)
        logger.info("Checkpoint comparison JSON saved to %s", summary_path)

    if _PANDAS_AVAILABLE:
        import pandas as pd

        df = pd.DataFrame(rows)
        if "checkpoint" in df.columns:
            df = df.set_index("checkpoint")

        if output_dir is not None:
            csv_path = Path(output_dir) / "checkpoint_comparison.csv"
            df.to_csv(str(csv_path))
            logger.info("Checkpoint comparison CSV saved to %s", csv_path)

        return df

    return rows


# ---------------------------------------------------------------------------
# ── CLI entry point ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VLM Evaluation Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path to a HuggingFace model directory or a .pt/.bin state-dict "
            "file.  Separate multiple paths with commas to run comparison mode."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Base model name/path (HuggingFace Hub id or local dir). "
            "Required when --checkpoint points to a raw state-dict file."
        ),
    )
    parser.add_argument(
        "--task",
        type=str,
        default="captioning",
        choices=list(Evaluator._TASK_DISPATCH.keys()),
        help="Evaluation task.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory of the evaluation dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_reports",
        help="Directory where reports are written.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate ('val' or 'test').",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="DataLoader batch size.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate per sample.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Beam search width.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string (e.g. 'cuda:0'). Auto-detected if omitted.",
    )
    return parser


def _build_config_from_args(args: argparse.Namespace) -> Any:
    """Construct a minimal config-like object from parsed CLI arguments."""

    class _CLIConfig:
        pass

    cfg = _CLIConfig()
    cfg.task = args.task
    cfg.max_new_tokens = args.max_new_tokens
    cfg.num_beams = args.num_beams
    cfg.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.eval_split = args.split
    if args.model:
        cfg.model_name = args.model
    return cfg


def _build_dataloader(args: argparse.Namespace, processor: Any) -> Any:
    """Build a minimal DataLoader for CLI use.

    This is a best-effort implementation.  For production use, replace
    with a project-specific dataset class that handles your data format.
    """
    from torch.utils.data import DataLoader, Dataset

    data_dir = Path(args.data_dir)
    task = args.task

    class _MinimalDataset(Dataset):
        """Loads image-text pairs from a JSONL annotation file."""

        def __init__(self, annotation_file: Path, img_root: Path, task: str) -> None:
            self.samples: List[Dict[str, Any]] = []
            self.img_root = img_root
            self.task = task
            if annotation_file.exists():
                with open(annotation_file, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            self.samples.append(json.loads(line))
            else:
                logger.warning("Annotation file not found: %s", annotation_file)

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            from PIL import Image  # type: ignore

            sample = self.samples[idx]
            img_path = self.img_root / sample.get("image", "")
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                image = Image.new("RGB", (224, 224))

            text = sample.get("question", sample.get("caption", sample.get("text", "")))
            inputs = processor(images=image, text=text, return_tensors="pt", padding=True)
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}

            # Attach ground-truth answers
            inputs["answers"] = sample.get(
                "answers", [sample.get("answer", sample.get("caption", ""))]
            )
            inputs["captions"] = sample.get(
                "captions", [sample.get("caption", "")]
            )
            inputs["labels"] = sample.get("label", sample.get("class_id", -1))
            inputs["sample_id"] = sample.get("id", idx)
            return inputs

        @staticmethod
        def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
            """Custom collate that handles non-tensor fields."""
            out: Dict[str, Any] = {}
            tensor_keys = [k for k, v in batch[0].items() if isinstance(v, torch.Tensor)]
            list_keys = [k for k, v in batch[0].items() if not isinstance(v, torch.Tensor)]

            for key in tensor_keys:
                out[key] = torch.stack([b[key] for b in batch], dim=0)
            for key in list_keys:
                out[key] = [b[key] for b in batch]
            return out

    annotation_file = data_dir / f"{args.split}.jsonl"
    img_root = data_dir / "images"
    dataset = _MinimalDataset(annotation_file, img_root, task)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        collate_fn=_MinimalDataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for VLM evaluation."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not _TRANSFORMERS_AVAILABLE:
        parser.error("transformers is required. pip install transformers")

    config = _build_config_from_args(args)

    # Support comma-separated list of checkpoints → comparison mode
    checkpoint_paths = [p.strip() for p in args.checkpoint.split(",") if p.strip()]

    # Load processor from first checkpoint (or model name)
    first_ckpt = checkpoint_paths[0]
    processor_source = first_ckpt if Path(first_ckpt).is_dir() else args.model
    if processor_source is None:
        parser.error(
            "--model must be specified when checkpoints are raw state-dict files."
        )

    from transformers import AutoProcessor as _AP

    processor = _AP.from_pretrained(processor_source)

    dataloader = _build_dataloader(args, processor)

    output_dir = args.output_dir

    if len(checkpoint_paths) == 1:
        evaluate_checkpoint(checkpoint_paths[0], config, dataloader, output_dir)
    else:
        result = compare_checkpoints(checkpoint_paths, config, dataloader, output_dir)
        if _PANDAS_AVAILABLE:
            import pandas as pd

            if isinstance(result, pd.DataFrame):
                print("\n" + result.to_string())
        else:
            for row in result:
                print(json.dumps(row, indent=2, default=str))


if __name__ == "__main__":
    main()
