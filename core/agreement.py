# inter-judge agreement helpers for the clinical judge panel
# Agr formula is from Eq.1 in the paper - basically normalised pairwise
# distance averaged over all judge pairs and all rubric criteria.
#
# four routing levels:
#   full_agree    >= 0.95
#   majority_agree  0.75 - 0.95
#   split           0.50 - 0.75
#   disagree      < 0.50
#
# S is only reported when Agr >= 0.75 (MA threshold). below that we don't
# trust the panel enough to give a score.

from __future__ import annotations
from itertools import combinations
from typing import Dict, Optional

FA_THRESHOLD = 0.95
MA_THRESHOLD = 0.75
SP_THRESHOLD = 0.50


def classify_agreement_level(agr, fa=FA_THRESHOLD, ma=MA_THRESHOLD, sp=SP_THRESHOLD):
    if agr >= fa:
        return "full_agree"
    if agr >= ma:
        return "majority_agree"
    if agr >= sp:
        return "split"
    return "disagree"


def compute_agr(judge_scores, kappa):
    """Eq.1 - pairwise agreement in [0,1].

    judge_scores: {judge_id: {criterion_id: raw_score}}
    kappa:        {criterion_id: max_possible_score}

    returns 1.0 if fewer than 2 judges (trivially agreed).
    """
    judge_ids = list(judge_scores.keys())
    if len(judge_ids) < 2:
        return 1.0

    total = 0.0
    agree = 0.0
    for ji, jj in combinations(judge_ids, 2):
        si = judge_scores[ji]
        sj = judge_scores[jj]
        for k, kap in kappa.items():
            if k not in si or k not in sj:
                continue
            if si[k] is None or sj[k] is None:
                continue
            if kap <= 0:
                continue
            agree += 1.0 - abs(si[k] - sj[k]) / kap
            total += 1.0

    return 0.0 if total == 0.0 else agree / total


def identify_outlier(judge_scores, kappa):
    """find whichever judge is most out of step with the rest.
    returns None if < 3 judges or no clear outlier (gap < 0.10).
    """
    ids = list(judge_scores.keys())
    if len(ids) < 3:
        return None

    mean_agr = {}
    for ji in ids:
        others = [j for j in ids if j != ji]
        vals = [compute_agr({ji: judge_scores[ji], jj: judge_scores[jj]}, kappa)
                for jj in others]
        mean_agr[ji] = sum(vals) / len(vals)

    worst = min(mean_agr, key=mean_agr.get)
    best_val = sorted(mean_agr.values(), reverse=True)[0]
    if best_val - mean_agr[worst] > 0.10:
        return worst
    return None


def summarize_agreement(judge_scores, kappa,
                        fa=FA_THRESHOLD, ma=MA_THRESHOLD, sp=SP_THRESHOLD):
    """run compute_agr + classify + outlier detection and bundle into a dict
    that gets written to the results JSON.
    """
    ids = list(judge_scores.keys())
    agr = compute_agr(judge_scores, kappa)
    level = classify_agreement_level(agr, fa, ma, sp)
    outlier = identify_outlier(judge_scores, kappa)

    pairwise = {}
    for ji, jj in combinations(ids, 2):
        v = compute_agr({ji: judge_scores[ji], jj: judge_scores[jj]}, kappa)
        pairwise[f"{ji}|{jj}"] = round(v, 4)

    return {
        "agr": round(agr, 4),
        "agreement_level": level,
        "outlier_judge": outlier,
        "report_score": agr >= ma,
        "thresholds": {"FA": fa, "MA": ma, "SP": sp},
        "pairwise_agr": pairwise,
    }
