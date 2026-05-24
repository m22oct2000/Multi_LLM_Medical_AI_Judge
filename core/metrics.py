# metrics / logging for judge runs
# nothing fancy - just dataclasses + an in-memory collector
# results get written out as JSON at the end of each experiment

import logging
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s'
)


@dataclass
class EvalRecord:
    timestamp: str
    question_id: str
    rubric_id: str
    rubric_name: str
    judge_id: str
    aggregate_score: float
    rationales: Dict[str, str]
    latency_ms: float
    status: str
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class AgreementRecord:
    timestamp: str
    question_id: str
    rubric_id: str
    judge_a: str
    judge_b: str
    agreement_score: float
    agreement_class: str

    def to_dict(self):
        return asdict(self)


class MetricsCollector:

    def __init__(self):
        self.eval_records: List[EvalRecord] = []
        self.agreement_records: List[AgreementRecord] = []
        self.logger = logging.getLogger('clinical_judge_metrics')

    def record_eval(self, question_id, rubric_id, rubric_name, judge_id,
                    aggregate_score, rationales, latency_ms, status, error=None):
        rec = EvalRecord(
            timestamp=datetime.utcnow().isoformat(),
            question_id=question_id,
            rubric_id=rubric_id,
            rubric_name=rubric_name,
            judge_id=judge_id,
            aggregate_score=aggregate_score,
            rationales=rationales,
            latency_ms=latency_ms,
            status=status,
            error=error,
        )
        self.eval_records.append(rec)
        self.logger.info(f'eval recorded: {judge_id} Q={question_id} score={aggregate_score:.2f}')

    def record_agreement(self, question_id, rubric_id, judge_a, judge_b,
                         agreement_score, agreement_class):
        rec = AgreementRecord(
            timestamp=datetime.utcnow().isoformat(),
            question_id=question_id,
            rubric_id=rubric_id,
            judge_a=judge_a,
            judge_b=judge_b,
            agreement_score=agreement_score,
            agreement_class=agreement_class,
        )
        self.agreement_records.append(rec)
        self.logger.info(
            f'agreement: {judge_a}|{judge_b} Q={question_id} '
            f'agr={agreement_score:.4f} -> {agreement_class}'
        )

    def get_eval_records(self):
        return [r.to_dict() for r in self.eval_records]

    def get_agreement_records(self):
        return [r.to_dict() for r in self.agreement_records]


_collector = None

def get_metrics_collector():
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector
