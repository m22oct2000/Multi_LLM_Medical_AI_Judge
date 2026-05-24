# rubric parser + score aggregation
# handles both BINARY (0/1/NA) and LIKERT rubrics.
# scale range is auto-detected from item metadata.
#
# two main things this does:
#   1. calculate_pairwise_agreement  - wraps compute_agr for two judge score lists
#   2. compute_quality_score         - Eq.2, only runs when Agr >= MA

import re
from typing import Dict, List, Optional, Tuple

from core.consensus_core.models import Rubric, RubricItem, JudgeScore
from core.agreement import compute_agr, MA_THRESHOLD


def _scale_range(item):
    """returns (lo, hi) for this rubric item.
    checks BINARY first, then tries to parse 'Score X-Y' from description,
    falls back to standard 1-5 Likert.
    """
    if (item.scale or '').upper() == 'BINARY':
        return 0.0, 1.0
    m = re.search(r'Score\s+(\d+)-(\d+)', item.description or '', re.IGNORECASE)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if hi > lo:
            return lo, hi
    return 1.0, 5.0


def _kappa(rubric):
    return {it.id: _scale_range(it)[1] for it in rubric.items}


def _is_na(score):
    return str(score).strip().upper() in ('', 'NA', 'N/A', 'NONE', 'NULL')


class DynamicRubricParser:

    def __init__(self, rubric):
        self.rubric = rubric
        self.item_by_id = {it.id: it for it in rubric.items}
        self.kappa = _kappa(rubric)
        # figure out if this is an all-binary rubric
        self.paradigm = 'BINARY' if all(
            (it.scale or '').upper() == 'BINARY' for it in rubric.items
        ) else 'LIKERT'

    def generate_judge_instructions(self, rubric=None):
        rb = rubric or self.rubric
        is_binary = all((it.scale or '').upper() == 'BINARY' for it in rb.items)
        lines = [
            'You are a strict medical domain judge evaluating a clinical QA answer.',
            'Score ONLY based on the rubric items below.',
            f'Rubric: {rb.name}',
        ]
        if is_binary:
            lines.append('For each item: 1 = meets criterion, 0 = does not meet, NA = not applicable.')
        else:
            lo, hi = _scale_range(rb.items[0]) if rb.items else (1, 5)
            lines.append(f'Score each item as an integer from {int(lo)} (poor) to {int(hi)} (excellent).')
        lines.append('One-line rationale then score for each item.')
        lines.append('\nItems:')
        for i, it in enumerate(rb.items, 1):
            lo, hi = _scale_range(it)
            sc = '1/0/NA' if (it.scale or '').upper() == 'BINARY' else f'{int(lo)}-{int(hi)}'
            lines.append(f'{i}. [{it.id}] {it.name}  (scale: {sc}, weight: {it.weight})')
            lines.append(f'   {it.description}')
        return '\n'.join(lines)

    def build_judge_scores_dict(self, judge_id, scores):
        """convert List[JudgeScore] -> {criterion_id: raw_score}
        needed by compute_agr which works on plain dicts.
        """
        out = {}
        for sc in scores:
            if sc.rubric_item_id not in self.item_by_id:
                continue
            if _is_na(sc.score):
                continue
            try:
                out[sc.rubric_item_id] = float(sc.score)
            except (ValueError, TypeError):
                continue
        return out

    def calculate_pairwise_agreement(self, scores_a, scores_b, judge_a='a', judge_b='b'):
        """pairwise Agr in [0,1] between two judges for a single question."""
        da = self.build_judge_scores_dict(judge_a, scores_a)
        db = self.build_judge_scores_dict(judge_b, scores_b)
        return compute_agr({judge_a: da, judge_b: db}, self.kappa)

    def compute_quality_score(self, all_judge_scores, agr,
                              ma_threshold=MA_THRESHOLD,
                              outlier_strategy='include', outlier_judge=None):
        """Eq.2 - weighted mean normalised score across all judges.
        returns None if Agr < MA (don't report S when panel disagrees).

        outlier_strategy: 'include' | 'remove' | 'downweight'
        """
        if agr < ma_threshold:
            return None

        ids = list(all_judge_scores.keys())
        weights = {jid: 1.0 for jid in ids}
        if outlier_judge and outlier_judge in weights:
            if outlier_strategy == 'remove':
                weights.pop(outlier_judge)
                ids = [j for j in ids if j != outlier_judge]
            elif outlier_strategy == 'downweight':
                weights[outlier_judge] = 0.5

        if not ids:
            return None

        parts = []
        for jid in ids:
            num = den = 0.0
            for sc in all_judge_scores[jid]:
                it = self.item_by_id.get(sc.rubric_item_id)
                if not it or _is_na(sc.score):
                    continue
                try:
                    v = float(sc.score)
                except (ValueError, TypeError):
                    continue
                kap = self.kappa.get(sc.rubric_item_id, 1.0)
                if kap <= 0:
                    continue
                w = float(it.weight or 1.0)
                num += w * (v / kap)
                den += w
            parts.append((weights[jid], num / den if den > 0 else 0.0))

        tw = sum(w for w, _ in parts)
        return None if tw == 0 else sum(w * s for w, s in parts) / tw

    def aggregate_score(self, scores):
        """legacy per-judge score kept for logging.
        returns 0-100 for BINARY, raw weighted mean for LIKERT.
        use compute_quality_score() for the paper metric.
        """
        if self.paradigm == 'BINARY':
            present = total = 0
            for sc in scores:
                if sc.rubric_item_id not in self.item_by_id or _is_na(sc.score):
                    continue
                try:
                    v = int(float(sc.score))
                except (ValueError, TypeError):
                    continue
                total += 1
                present += v == 1
            return 0.0 if total == 0 else (present / total) * 100.0

        num = den = 0.0
        for sc in scores:
            it = self.item_by_id.get(sc.rubric_item_id)
            if not it or _is_na(sc.score):
                continue
            try:
                val = float(sc.score)
            except (ValueError, TypeError):
                continue
            w = float(it.weight or 1.0)
            num += w * val
            den += w
        return 0.0 if den == 0 else num / den
