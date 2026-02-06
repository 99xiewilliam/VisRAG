import math
import re
from typing import Dict, Iterable, List

from ..utils import get_logger

logger = get_logger(__name__)


def _ensure_nltk() -> bool:
    try:
        import nltk  # noqa: F401
    except Exception:
        return False

    try:
        import nltk

        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet")
        try:
            nltk.data.find("corpora/omw-1.4")
        except LookupError:
            nltk.download("omw-1.4")
        return True
    except Exception:
        return False


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_f1(pred: str, ref: str) -> float:
    pred_toks = normalize_text(pred).split()
    ref_toks = normalize_text(ref).split()
    if not pred_toks and not ref_toks:
        return 1.0
    if not pred_toks or not ref_toks:
        return 0.0
    common: Dict[str, int] = {}
    for t in pred_toks:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in ref_toks:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    precision = overlap / len(pred_toks)
    recall = overlap / len(ref_toks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_len(a: List[str], b: List[str]) -> int:
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[-1]


def rouge_l(pred: str, ref: str) -> float:
    pred_toks = normalize_text(pred).split()
    ref_toks = normalize_text(ref).split()
    if not pred_toks or not ref_toks:
        return 0.0
    lcs = _lcs_len(pred_toks, ref_toks)
    precision = lcs / len(pred_toks)
    recall = lcs / len(ref_toks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def meteor_score(pred: str, ref: str, *, download: bool = False) -> float:
    try:
        import nltk  # noqa: F401
    except Exception:
        return 0.0
    if download:
        _ensure_nltk()
    try:
        from nltk.translate.meteor_score import single_meteor_score

        ref_tokens = normalize_text(ref).split()
        pred_tokens = normalize_text(pred).split()
        return float(single_meteor_score(ref_tokens, pred_tokens))
    except Exception:
        return 0.0


def _ngram_counts(tokens: List[str], n: int) -> Dict[tuple, int]:
    counts: Dict[tuple, int] = {}
    if n <= 0:
        return counts
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i : i + n])
        counts[ng] = counts.get(ng, 0) + 1
    return counts


def _bleu_precisions(pred_tokens: List[str], ref_tokens: List[str], max_n: int) -> List[float]:
    precisions: List[float] = []
    ref_counts_by_n = {n: _ngram_counts(ref_tokens, n) for n in range(1, max_n + 1)}
    for n in range(1, max_n + 1):
        pred_counts = _ngram_counts(pred_tokens, n)
        if not pred_counts:
            precisions.append(0.0)
            continue
        overlap = 0
        ref_counts = ref_counts_by_n[n]
        for ng, c in pred_counts.items():
            overlap += min(c, ref_counts.get(ng, 0))
        total = sum(pred_counts.values())
        precisions.append(overlap / total if total > 0 else 0.0)
    return precisions


def _brevity_penalty(pred_len: int, ref_len: int) -> float:
    if pred_len == 0:
        return 0.0
    if pred_len > ref_len:
        return 1.0
    return math.exp(1.0 - (ref_len / pred_len))


def bleu_n(pred: str, ref: str, n: int) -> float:
    pred_tokens = normalize_text(pred).split()
    ref_tokens = normalize_text(ref).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    max_n = max(1, min(4, n))
    precisions = _bleu_precisions(pred_tokens, ref_tokens, max_n)
    # Smoothing: replace zeros with a tiny epsilon
    eps = 1e-9
    precisions = [p if p > 0 else eps for p in precisions]
    weights = [1.0 / max_n] * max_n
    log_sum = 0.0
    for w, p in zip(weights, precisions):
        log_sum += w * math.log(p)
    bp = _brevity_penalty(len(pred_tokens), len(ref_tokens))
    return float(bp * math.exp(log_sum))


def compute_metrics(pred: str, ref: str) -> Dict[str, float]:
    return {
        "f1": token_f1(pred, ref),
        "rouge_l": rouge_l(pred, ref),
        "bleu_1": bleu_n(pred, ref, 1),
        "bleu_2": bleu_n(pred, ref, 2),
        "bleu_3": bleu_n(pred, ref, 3),
        "bleu_4": bleu_n(pred, ref, 4),
        "meteor": meteor_score(pred, ref, download=True),
    }


def average_metrics(items: Iterable[Dict[str, float]]) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    count = 0
    for item in items:
        count += 1
        for k, v in item.items():
            if isinstance(v, (int, float)):
                sums[k] = sums.get(k, 0.0) + float(v)
    if count == 0:
        return {k: 0.0 for k in sums}
    return {k: v / count for k, v in sums.items()}
