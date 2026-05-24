# Consensus-Driven Medical Answer Evaluation: A Multi-SLM Judge Framework for Privacy-Preserving, Local Deployment

> **EMNLP 2026 Long Paper Submission** -- studying whether panels of locally
> deployed small language models (SLMs) can reliably evaluate medical answer
> quality, and how rubric design and scoring scale jointly affect inter-judge
> consensus.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![EMNLP 2026](https://img.shields.io/badge/Venue-EMNLP%202026-green)]()

Code for the paper:

> **"Consensus-Driven Medical Answer Evaluation: A Multi-SLM Judge Framework
> for Privacy-Preserving, Local Deployment"**

```
https://github.com/mmm-byte/Multi_LLM_Medical_AI_Judge
```

---

## Research Overview

Cloud-based frontier models carry HIPAA compliance risks and are unavailable in
offline or bandwidth-constrained clinical environments. This repository
implements **Multi-SLMs-as-Judge**: a privacy-preserving framework that assembles
a panel of locally deployed SLMs, measures inter-judge consensus, and routes
disagreements to human review rather than silently delivering an unreliable score.

| # | Research Question |
|---|---|
| RQ1 | Which rubric gives the most consistent judge agreement at its natural scoring scale? |
| RQ2 | How does scoring scale change agreement when rubric content is held constant? |
| RQ3 | Do agreement patterns differ across clinical specialties? |

---

## Framework: Two-Stage Pipeline

The pipeline has a strict order (paper Sec 3 / Fig. 1):

**Stage 1 -- Compute Agr (Eq. 1):** pairwise agreement across all N judges:

```
Agr = (2 / N(N-1)) * sum_{i<j} sum_k (1 - |s_ik - s_jk| / kappa_k)
```

where kappa_k is the maximum achievable score for criterion k, normalising
differences to [0, 1]. Agr in [0, 1].

**Stage 2 -- Routing gate:** four levels based on configurable thresholds:

| Level | Condition | Action |
|---|---|---|
| Full Agreement (FA) | Agr >= 0.95 | Compute and deliver S |
| Majority Agreement (MA) | 0.75 <= Agr < 0.95 | Compute and deliver S with note |
| Split (SP) | 0.50 <= Agr < 0.75 | Route to human reviewer |
| Disagree (D) | Agr < 0.50 | Escalate to domain expert |

**S is only reported when Agr >= MA (0.75).** Below that threshold the system
raises an alert with per-judge outputs for human review.

**Stage 3 -- Compute S (Eq. 2):** weighted quality score in [0, 1]:

```
S = (1/N) * sum_i  [sum_k w_k * (s_ik / kappa_k)] / [sum_k w_k]
```

Three outlier strategies available: *include* (equal weights), *remove* (drop
most deviant judge), *downweight* (outlier weight x 0.5).

---

## Benchmark

100 questions drawn from MedQuAD, MedDialog, and Medical Meadow, balanced
across five clinical domains (20 questions each):

| Domain | Example Topics |
|---|---|
| Cardiology | atrial fibrillation, heart failure, STEMI |
| Pharmacology | warfarin, drug interactions, dosing |
| Neurology | stroke, Alzheimer disease, seizures |
| Pediatrics | otitis media, dehydration, febrile seizures |
| Emergency | anaphylaxis, sepsis, tension pneumothorax |

---

## Rubrics

Five rubrics -- four published instruments plus one controlled scoring variant:

| ID | File | Scale | Purpose |
|---|---|---|---|
| R1 | `rubric1_pemat.json` | Binary | Patient-facing understandability / actionability (PEMAT) |
| R2 | `rubric2_healthbench.json` | Binary | Clinical correctness, safety, escalation, uncertainty (HealthBench) |
| R3 | `rubric3_clinical_eval.json` | Likert 1-5 | Accuracy, safety, relevance, completeness, clarity (ClinicalEval) |
| R4 | `rubric4_prometheus.json` | Likert 1-5 | Instruction-following, factuality, coherence, completeness (Prometheus) |
| R5 | `rubric5_pemat_likert.json` | Likert 1-5 | Same PEMAT criteria as R1 -- controlled scale comparison |

R5 isolates the scale effect: identical criteria, different scale => direct
test of binary vs. Likert agreement.

---

## Judge Panel

Four locally deployable SLMs spanning three distinct base architectures
(paper Sec 4):

| Judge ID | Model | Reference |
|---|---|---|
| `medgemma` | `google/medgemma-4b-it` | sellergren2025medgemma |
| `biomistral` | `BioMistral/BioMistral-7B-DARE` | labrak2024biomistral |
| `meditron` | `epfl-llm/meditron-7b` | chen2023meditron70b |
| `medalpaca` | `medalpaca/medalpaca-7b` | han2023medalpaca |

All models are served via local vLLM / FastAPI endpoints -- no data leaves
your infrastructure.

---

## Agreement Thresholds (paper Sec 4)

Calibrated for a four-judge panel:

| Threshold | Value | Meaning |
|---|---|---|
| FA | 0.95 | Near-unanimous; virtually no systematic disagreement |
| MA | 0.75 | Supermajority; panel broadly converges even if one judge diverges |
| SP | 0.50 | Coin-flip boundary; no stable majority below this point |

Thresholds are fully configurable via `agreement_thresholds` in the experiment
config JSON. Two or more thresholds set to the same value merge the corresponding
levels.

---

## Repository Structure

```text
core/
  wrapper.py                 # Judge-panel orchestration (3-stage scoring)
  model_adapters.py          # Per-model prompt/parse adapters
  rubric_engine.py           # Agr (Eq.1), S (Eq.2), outlier strategies
  agreement.py               # Panel agreement taxonomy and routing gate
  consensus_core/            # Dataclasses, event log, in-memory store

config/
  endpoint_config.py         # Default local judge endpoints (4 SLMs)
  llm_client.py              # Lightweight OpenAI-compatible HTTP client
  configs/                   # Per-experiment JSON configs

benchmark_dataset/
  build_agreement_dataset.py # Reproducible 100-Q benchmark builder
  source_datasets/           # Place MedQuAD / MedDialog / Medical Meadow CSVs here

rubrics/
  rubric1_pemat.json
  rubric2_healthbench.json
  rubric3_clinical_eval.json
  rubric4_prometheus.json
  rubric5_pemat_likert.json

experiments/
  demo_agreement_showcases.py   # No-LLM walkthrough of all agreement levels
  exp1_dataset_analysis.py      # Table 1: rubric results at default scale
  exp2_agreement_analysis.py    # Table 2: scoring-scale sensitivity
  exp3_rubric_sensitivity.py    # Table 3: agreement by clinical domain
  exp4_boxplot_agreement.py     # Box-plot figures (reads Exp2 output)

tests/
  test_core.py
```

---

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# No LLM required -- runs tests, builds benchmark, walks through agreement levels
python3 tests/test_core.py
python3 benchmark_dataset/build_agreement_dataset.py
python3 experiments/demo_agreement_showcases.py

# Full pipeline (requires judge endpoints on ports 8001-8004)
bash run_all.sh
```

---

## Experiments

| Experiment | Script | Requires LLMs? | Paper Table |
|---|---|---|---|
| Exp1 | `experiments/exp1_dataset_analysis.py` | No | Table 1 (default-scale results) |
| Exp2 | `experiments/exp2_agreement_analysis.py` | Yes | Table 2 (scale sensitivity) |
| Exp3 | `experiments/exp3_rubric_sensitivity.py` | Yes | Table 3 (domain breakdown) |
| Exp4 | `experiments/exp4_boxplot_agreement.py` | No (reads Exp2) | Figures |
| Demo | `experiments/demo_agreement_showcases.py` | No | -- |

---

## Key Findings

- Binary rubrics consistently yield higher inter-judge agreement (PEMAT 94.1%,
  HealthBench 91.3%) than Likert-scale rubrics (ClinicalEval 80.9%,
  Prometheus 74.0%, PEMAT-Likert 72.2%).
- PEMAT vs PEMAT-Likert (identical criteria, different scale) isolates the scale
  effect: switching from binary to Likert 1-5 drops mean agreement by 22 pp.
- Cardiology is the most consistently judged domain (36% Full Agreement,
  0 Full Disagreements); Pharmacology is the most contested (20% FA, 38%
  combined Split + Full Disagreement).
- Pharmacology contention is driven primarily by Likert-scale rubrics (ClinicalEval
  13, PEMAT-Likert 14 split/disagree cases); binary rubrics show only 2-3 even
  in this domain.

---

## Ethical Considerations

This repository is research infrastructure for evaluating clinical NLP systems.
It does not provide, endorse, or validate medical advice. All benchmark questions
are drawn from publicly available datasets. Any clinical deployment would require
physician validation and regulatory review.

All experiments use locally deployed models -- no patient-identifiable text is
transmitted to external services, consistent with HIPAA and institutional
data-governance principles.

To support reproducibility, code, prompts, and evaluation scripts are released
here. Model checkpoints are not redistributed and must be obtained from their
original sources in compliance with their respective licences.
