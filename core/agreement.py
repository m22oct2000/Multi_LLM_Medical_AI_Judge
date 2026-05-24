"""Agreement classification for the Multi-SLMs-as-Judge clinical panel.

Implements the paper's Agr metric (Eq. 1) and four routing levels:

  Agr = (2 / N(N-1)) * sum_{i<j} sum_k (1 - |s_ik - s_jk| / kappa_k)

where kappa_k is the maximum achievable score for criterion k,
normalising each per-criterion difference to [0, 1].
Consequently Agr in [0, 1].

Four agreement levels (paper Sec 4 / routing gate):
  Full Agreement  (FA) : Agr >= FA_THRESHOLD  (default 0.95)
  Majority Agreement   : MA_THRESHOLD <= Agr < FA_THRESHOLD (default 0.75)
  Split          (SP)  : SP_THRESHOLD <= Agr < MA_THRESHOLD  (default 0.50)
  Disagree        (D)  : Agr < SP_THRESHOLD

Only when Agr >= MA_THRESHOLD does the framework compute and deliver the
quality score S (see rubric_engine.compute_quality_score).
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

# Paper Sec 4: threshold calibration for a 4-judge panel
FA_THRESHOLD: float = 0.95   # Full Agreement
MA_THRESHOLD: float = 0.75   # Majority Agreement gate (Agr >= MA => report S)
SP_THRESHOLD: float = 0.50   # Split boundary


def classify_agreement_level(
    agr: float,
    fa: float = FA_THRESHOLD,
    ma: float = MA_THRESHOLD,
    sp: float = SP_THRESHOLD,
) -> str:
    """Map a scalar Agr in [0,1] to one of the four paper routing levels.

    Returns one of: 'full_agree' | 'majority_agree' | 'split' | 'disagree'
    """
    if agr >= fa:
        return "full_agree"
    if agr >= ma:
        return "majority_agree"
    if agr >= sp:
        return "split"
    return "disagree"


def compute_agr(
    judge_scores: Dict[str, Dict[str, float]],
    kappa: Dict[str, float],
) -> float:
    """Compute paper Eq. 1 pairwise agreement Agr in [0, 1].

    Parameters
    ----------
    judge_scores : {judge_id: {criterion_id: score}}
        Per-judge, per-criterion raw scores. Missing or NA items are skipped.
    kappa : {criterion_id: max_achievable_score}
        Maximum achievable score for each criterion (normalisation denominator).

    Returns
    -------
    float
        Agr in [0, 1]; 1.0 = perfect consensus, 0.0 = maximal disagreement.
        Returns 1.0 if fewer than 2 judges are present.
    """
    judge_ids = list(judge_scores.keys())
    n = len(judge_ids)
    if n < 2:
        return 1.0

    pairs = list(combinations(judge_ids, 2))
    total_weight = 0.0
    weighted_agreement = 0.0

    for ji, jj in pairs:
        scores_i = judge_scores[ji]
        scores_j = judge_scores[jj]
        common_criteria = [
            k for k in kappa
            if k in scores_i and k in scores_j
            and scores_i[k] is not None
            and scores_j[k] is not None
        ]
        for k in common_criteria:
            kap = kappa[k]
            if kap <= 0:
                continue
            diff = abs(scores_i[k] - scores_j[k]) / kap
            weighted_agreement += 1.0 - diff
            total_weight += 1.0

    if total_weight == 0.0:
        return 0.0

    return weighted_agreement / total_weight


def identify_outlier(
    judge_scores: Dict[str, Dict[str, float]],
    kappa: Dict[str, float],
) -> Optional[str]:
    """Return the judge ID whose mean pairwise agreement with others is lowest.

    Returns None if all judges agree or fewer than 3 judges are present.
    """
    judge_ids = list(judge_scores.keys())
    if len(judge_ids) < 3:
        return None

    mean_agr_with_others: Dict[str, float] = {}
    for ji in judge_ids:
        others = [j for j in judge_ids if j != ji]
        agr_vals = [
            compute_agr({ji: judge_scores[ji], jj: judge_scores[jj]}, kappa)
            for jj in others
        ]
        mean_agr_with_others[ji] = sum(agr_vals) / len(agr_vals) if agr_vals else 1.0

    min_judge = min(mean_agr_with_others, key=mean_agr_with_others.get)
    sorted_vals = sorted(mean_agr_with_others.values(), reverse=True)
    if len(sorted_vals) >= 2 and sorted_vals[0] - mean_agr_with_others[min_judge] > 0.10:
        return min_judge
    return None


def summarize_agreement(
    judge_scores: Dict[str, Dict[str, float]],
    kappa: Dict[str, float],
    fa: float = FA_THRESHOLD,
    ma: float = MA_THRESHOLD,
    sp: float = SP_THRESHOLD,
) -> Dict:
    """Return a full summary dict suitable for JSON results output.

    Parameters
    ----------
    judge_scores : {judge_id: {criterion_id: score}}
    kappa        : {criterion_id: max_achievable_score}
    fa, ma, sp   : routing thresholds (paper defaults: 0.95, 0.75, 0.50)

    Returns
    -------
    dict with keys:
      agr              - scalar Agr in [0, 1]
      agreement_level  - 'full_agree' | 'majority_agree' | 'split' | 'disagree'
      outlier_judge    - judge ID or None
      report_score     - bool: True only when Agr >= MA (score delivery gate)
      thresholds       - {FA, MA, SP} values used
      pairwise_agr     - per-pair Agr values
    """
    judge_ids = list(judge_scores.keys())
    agr = compute_agr(judge_scores, kappa)
    level = classify_agreement_level(agr, fa, ma, sp)
    outlier = identify_outlier(judge_scores, kappa)

    pairwise = {}
    for ji, jj in combinations(judge_ids, 2):
        pair_agr = compute_agr(
            {ji: judge_scores[ji], jj: judge_scores[jj]}, kappa
        )
        pairwise[f"{ji}|{jj}"] = round(pair_agr, 4)

    return {
        "agr": round(agr, 4),
        "agreement_level": level,
        "outlier_judge": outlier,
        "report_score": agr >= ma,
        "thresholds": {"FA": fa, "MA": ma, "SP": sp},
        "pairwise_agr": pairwise,
    }
