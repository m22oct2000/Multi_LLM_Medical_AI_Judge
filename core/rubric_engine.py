"""DynamicRubricParser: aggregates judge scores, computes Agr (Eq.1), and
computes the quality score S (Eq.2) -- gated on Agr >= MA.

Paper equations
---------------
Eq. 1  Agr = (2 / N(N-1)) * sum_{i<j} sum_k (1 - |s_ik - s_jk| / kappa_k)
       Implemented in core.agreement.compute_agr; called from here.

Eq. 2  S = (1/N) * sum_i  [sum_k w_k * (s_ik / kappa_k)] / [sum_k w_k]
       Computed only when Agr >= MA_THRESHOLD (score delivery gate).

Scale detection order:
  1. item.scale == 'BINARY'         => kappa = 1
  2. description suffix 'Score X-Y' => kappa = Y
  3. Default LIKERT range (1, 5)    => kappa = 5

Outlier strategies (paper Sec 3):
  'include'    - all judges weighted equally (default)
  'remove'     - drop the most deviant judge before computing S
  'downweight' - down-weight outlier judge by factor 0.5
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from core.consensus_core.models import Rubric, RubricItem, JudgeScore
from core.agreement import (
    compute_agr, classify_agreement_level, identify_outlier,
    FA_THRESHOLD, MA_THRESHOLD, SP_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Scale helpers
# ---------------------------------------------------------------------------

def _detect_scale_range(item: RubricItem) -> Tuple[float, float]:
    """Return (lo, hi) normalisation range for this item."""
    if (item.scale or '').upper() == 'BINARY':
        return 0.0, 1.0
    desc = (item.description or '').strip()
    m = re.search(r'Score\s+(\d+)-(\d+)\.?\s*$', desc, re.IGNORECASE)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if hi > lo:
            return lo, hi
    return 1.0, 5.0  # default LIKERT


def _build_kappa(rubric: Rubric) -> Dict[str, float]:
    """Return {item_id: kappa_k} where kappa_k = max achievable score (hi)."""
    return {it.id: _detect_scale_range(it)[1] for it in rubric.items}


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

class DynamicRubricParser:
    """Parse rubric scores, compute Agr (Eq.1) and S (Eq.2).

    Supports BINARY (0/1/NA) and LIKERT (any integer range).
    """

    def __init__(self, rubric: Rubric) -> None:
        self.rubric     = rubric
        self.paradigm   = self._detect_paradigm(rubric)
        self.item_by_id = {it.id: it for it in rubric.items}
        self.kappa      = _build_kappa(rubric)  # {item_id: max_score}

    @staticmethod
    def _detect_paradigm(rubric: Rubric) -> str:
        if rubric.items and all((it.scale or '').upper() == 'BINARY'
                                for it in rubric.items):
            return 'BINARY'
        return 'LIKERT'

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def generate_judge_instructions(self, rubric: Optional[Rubric] = None) -> str:
        rb       = rubric or self.rubric
        paradigm = self._detect_paradigm(rb)
        header   = [
            'You are a strict medical domain judge evaluating a clinical QA answer.',
            'Score ONLY based on the rubric items below.',
            f'This rubric is taken whole from: {rb.name}',
        ]
        if paradigm == 'BINARY':
            header.append('Score each item: 1=Present/Meets, 0=Absent/Does not meet, NA=Not Applicable.')
        else:
            lo, hi = _detect_scale_range(rb.items[0]) if rb.items else (1.0, 5.0)
            header.append(f'Score each item: integer from {int(lo)} (poor) to {int(hi)} (excellent).')
        header.append('For EACH item provide a one-line rationale then your score.')

        lines = ['\n'.join(header), '\nRubric Items:']
        for idx, it in enumerate(rb.items, start=1):
            lo, hi    = _detect_scale_range(it)
            scale_str = '1/0/NA' if (it.scale or '').upper() == 'BINARY' else f'{int(lo)}-{int(hi)}'
            lines.append(f'{idx}. [{it.id}] {it.name} (scale: {scale_str}, weight: {it.weight})')
            lines.append(f'   {it.description}')
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Score parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_na(score) -> bool:
        return str(score).strip().upper() in ('', 'NA', 'N/A', 'NONE', 'NULL')

    @staticmethod
    def _normalize_score(item: RubricItem, score) -> Tuple[float, bool]:
        """Normalise raw score to [0, 1] using item-specific scale range."""
        raw = str(score).strip().upper()
        if raw in ('', 'NA', 'N/A', 'NONE', 'NULL'):
            return 0.0, False
        try:
            v = float(raw)
        except (ValueError, TypeError):
            return 0.0, False
        if (item.scale or '').upper() == 'BINARY':
            return (1.0 if v >= 0.5 else 0.0), True
        lo, hi = _detect_scale_range(item)
        if hi == lo:
            return 0.0, False
        v = max(lo, min(hi, v))
        return (v - lo) / (hi - lo), True

    # ------------------------------------------------------------------
    # Build judge_scores dict for agreement computation
    # ------------------------------------------------------------------

    def build_judge_scores_dict(
        self,
        judge_id: str,
        scores: List[JudgeScore],
    ) -> Dict[str, float]:
        """Convert List[JudgeScore] to {criterion_id: raw_score} for compute_agr."""
        result = {}
        for sc in scores:
            if sc.rubric_item_id not in self.item_by_id:
                continue
            if self._is_na(sc.score):
                continue
            try:
                result[sc.rubric_item_id] = float(sc.score)
            except (ValueError, TypeError):
                continue
        return result

    # ------------------------------------------------------------------
    # Eq. 1: Pairwise agreement [0, 1]
    # ------------------------------------------------------------------

    def calculate_pairwise_agreement(
        self,
        scores_a: List[JudgeScore],
        scores_b: List[JudgeScore],
        judge_a: str = 'a',
        judge_b: str = 'b',
    ) -> float:
        """Pairwise agreement in [0, 1] using paper Eq. 1 normalisation.

        Each criterion difference is divided by kappa_k (max achievable score),
        making scores comparable across rubrics with different rating scales.
        """
        da = self.build_judge_scores_dict(judge_a, scores_a)
        db = self.build_judge_scores_dict(judge_b, scores_b)
        return compute_agr(
            {judge_a: da, judge_b: db},
            self.kappa,
        )

    # ------------------------------------------------------------------
    # Eq. 2: Quality score S -- only reported when Agr >= MA
    # ------------------------------------------------------------------

    def compute_quality_score(
        self,
        all_judge_scores: Dict[str, List[JudgeScore]],
        agr: float,
        ma_threshold: float = MA_THRESHOLD,
        outlier_strategy: str = 'include',
        outlier_judge: Optional[str] = None,
    ) -> Optional[float]:
        """Compute quality score S per paper Eq. 2.

        S = (1/N) * sum_i  [sum_k w_k * (s_ik / kappa_k)] / [sum_k w_k]

        Parameters
        ----------
        all_judge_scores  : {judge_id: List[JudgeScore]}
        agr               : Agr value from compute_agr (paper Eq. 1)
        ma_threshold      : gate -- only compute S when agr >= ma_threshold
        outlier_strategy  : 'include' | 'remove' | 'downweight'
        outlier_judge     : judge ID identified as outlier (for remove/downweight)

        Returns
        -------
        float in [0, 1] if agr >= ma_threshold, else None (route to human review).
        """
        if agr < ma_threshold:
            return None  # Agr < MA: do not report score, route to human review

        judge_ids = list(all_judge_scores.keys())

        # Apply outlier strategy
        weights: Dict[str, float] = {jid: 1.0 for jid in judge_ids}
        if outlier_judge and outlier_judge in weights:
            if outlier_strategy == 'remove':
                del weights[outlier_judge]
                judge_ids = [j for j in judge_ids if j != outlier_judge]
            elif outlier_strategy == 'downweight':
                weights[outlier_judge] = 0.5

        if not judge_ids:
            return None

        per_judge_s = []
        for jid in judge_ids:
            scores = all_judge_scores[jid]
            num = den = 0.0
            for sc in scores:
                it = self.item_by_id.get(sc.rubric_item_id)
                if not it or self._is_na(sc.score):
                    continue
                try:
                    v = float(sc.score)
                except (ValueError, TypeError):
                    continue
                kap = self.kappa.get(sc.rubric_item_id, 1.0)
                if kap <= 0:
                    continue
                w    = float(it.weight or 1.0)
                num += w * (v / kap)
                den += w
            per_judge_s.append((weights[jid], num / den if den > 0 else 0.0))

        total_w = sum(w for w, _ in per_judge_s)
        if total_w == 0:
            return None
        return sum(w * s for w, s in per_judge_s) / total_w

    # ------------------------------------------------------------------
    # Legacy aggregate_score (unnormalized; kept for backward compat)
    # ------------------------------------------------------------------

    def aggregate_score(self, scores: List[JudgeScore]) -> float:
        """Per-judge aggregate score (0-100 for BINARY, raw mean for LIKERT).

        Kept for internal logging; use compute_quality_score for paper S.
        """
        if self.paradigm == 'BINARY':
            present = total = 0
            for sc in scores:
                if sc.rubric_item_id not in self.item_by_id:
                    continue
                raw = str(sc.score).strip().upper()
                if raw in ('', 'NA', 'N/A', 'NONE', 'NULL'):
                    continue
                try:
                    v = int(float(raw))
                except (ValueError, TypeError):
                    continue
                total += 1
                if v == 1:
                    present += 1
            return 0.0 if total == 0 else (present / total) * 100.0

        num = den = 0.0
        for sc in scores:
            it = self.item_by_id.get(sc.rubric_item_id)
            if not it:
                continue
            raw = str(sc.score).strip().upper()
            if raw in ('', 'NA', 'N/A', 'NONE', 'NULL'):
                continue
            try:
                val = float(raw)
            except (ValueError, TypeError):
                continue
            w    = float(it.weight or 1.0)
            num += w * val
            den += w
        return 0.0 if den == 0.0 else num / den
