# Multi-SLM Medical AI Judge

Code for the paper *"Consensus-Driven Medical Answer Evaluation: A Multi-SLM Judge Framework for Privacy-Preserving, Local Deployment"* (EMNLP 2026 submission).

The core idea: instead of sending clinical QA answers to a cloud model for evaluation (HIPAA risk, no offline support), run a panel of small locally-deployed medical LLMs and only report a quality score when the panel sufficiently agrees. When they disagree, route to a human reviewer.

---

## Why bother with consensus?

A single judge model can be confidently wrong. Running four models and checking whether they land in the same region gives a cheap signal of when to trust the automated score vs. when to flag it. The threshold we use (Agr >= 0.75) was calibrated on the validation split - below that, precision drops enough that automated scores aren't worth trusting on their own.

---

## How it works

**Step 1 - Score.** Each of the four judge models scores the answer independently on all rubric criteria. Three-stage prompting handles models that fail to score all items in one batch (per-item retry, then a minimal single-criterion fallback).

**Step 2 - Compute Agr.** Pairwise agreement across all N judges (Eq.1 from the paper):

```
Agr = (2 / N(N-1)) * sum_{i<j} sum_k (1 - |s_ik - s_jk| / kappa_k)
```

Each criterion difference is divided by its max possible score so binary and Likert rubrics are comparable. Agr is in [0, 1].

**Step 3 - Route.**

| Level | Range | Action |
|---|---|---|
| Full Agreement | Agr >= 0.95 | report S |
| Majority Agreement | 0.75 - 0.95 | report S with note |
| Split | 0.50 - 0.75 | flag for human review |
| Disagree | < 0.50 | escalate to domain expert |

**Step 4 - Compute S (only when Agr >= 0.75).**

```
S = (1/N) * sum_i [ sum_k w_k * (s_ik / kappa_k) ] / [ sum_k w_k ]
```

If there's an outlier judge (one model clearly out of step with the other three) you can remove or downweight it before computing S.

---

## Judges

Four models, all locally deployable, spanning different training backgrounds:

| id | model | paper ref |
|---|---|---|
| medgemma | google/medgemma-4b-it | sellergren2025medgemma |
| biomistral | BioMistral/BioMistral-7B-DARE | labrak2024biomistral |
| meditron | epfl-llm/meditron-7b | chen2023meditron70b |
| medalpaca | medalpaca/medalpaca-7b | han2023medalpaca |

Each runs behind a local vLLM server. No data leaves your machine.

---

## Rubrics

Five rubrics - four from published work plus one controlled variant:

| id | scale | what it measures |
|---|---|---|
| R1 PEMAT | binary | patient understandability + actionability |
| R2 HealthBench | binary | clinical correctness, safety, escalation |
| R3 ClinicalEval | Likert 1-5 | accuracy, safety, relevance, completeness |
| R4 Prometheus | Likert 1-5 | factuality, coherence, instruction-following |
| R5 PEMAT-Likert | Likert 1-5 | same criteria as R1, different scale |

R1 vs R5 is the controlled pair for the scale-effect analysis in Exp3.

---

## Benchmark

100 questions from MedQuAD, MedDialog, and Medical Meadow. 20 per domain:
Cardiology, Pharmacology, Neurology, Pediatrics, Emergency.

Script to reproduce: `benchmark_dataset/build_agreement_dataset.py`

---

## Running it

```bash
pip install -r requirements.txt

# sanity check without any models
python tests/test_core.py
python benchmark_dataset/build_agreement_dataset.py
python experiments/demo_agreement_showcases.py

# full run (needs judge servers on ports 8001-8004)
bash run_all.sh
```

Each experiment reads a config JSON from `config/configs/`. The config specifies which judges, which rubrics, and the agreement thresholds. Default thresholds: FA=0.95, MA=0.75, SP=0.50.

---

## Repo layout

```
core/
  wrapper.py          - runs the full panel for one question
  rubric_engine.py    - Agr (Eq.1) + S (Eq.2) + outlier handling
  agreement.py        - agreement thresholds and routing
  model_adapters.py   - per-model prompt builders and parsers
  metrics.py          - logging / result collection

config/
  endpoint_config.py  - judge endpoint definitions
  configs/            - per-experiment JSON configs

benchmark_dataset/
  build_agreement_dataset.py

rubrics/              - five rubric JSON files
experiments/          - exp1 through exp4 + demo script
tests/
```

---

## Main results

Binary rubrics agree more than Likert ones. PEMAT (binary) hit 94.1% mean agreement; PEMAT-Likert (same criteria, 1-5 scale) dropped to 72.2% - a 22 pp gap from scale alone. Cardiology was the most consistently judged domain; Pharmacology was the hardest, mainly on Likert rubrics.

---

## Notes

This is research infrastructure, not a clinical tool. Nothing here should be used for actual patient care decisions. All benchmark data is from public datasets. Models are not redistributed - download them from HuggingFace under their respective licences.
