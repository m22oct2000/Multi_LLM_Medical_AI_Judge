# tests for core pipeline components - no LLM calls needed
# covers: rubric JSON loading, config validation, benchmark CSV,
#         heuristic scoring, exp1 placeholder, agreement math
#
# run with:  pytest tests/test_core.py -v
# or just:   python tests/test_core.py

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_PASS = []
_FAIL = []

def ok(name):
    _PASS.append(name)
    print(f'  PASS  {name}')

def fail(name, reason):
    _FAIL.append(name)
    print(f'  FAIL  {name}: {reason}')

def expect(cond, name, reason=''):
    ok(name) if cond else fail(name, reason or 'condition is False')


# 1. rubric JSON files
RUBRIC_FILES = [
    'rubrics/rubric1_pemat.json',
    'rubrics/rubric2_healthbench.json',
    'rubrics/rubric3_clinical_eval.json',
    'rubrics/rubric4_prometheus.json',
    'rubrics/rubric5_pemat_likert.json',
]
REQUIRED_RUBRIC_KEYS = {'id', 'name', 'source_paper', 'paradigm', 'items'}
REQUIRED_ITEM_KEYS   = {'id', 'name', 'description', 'scale', 'weight', 'why'}

def test_rubrics():
    print('\n[1] rubric JSON files')
    for path_str in RUBRIC_FILES:
        path = ROOT / path_str
        expect(path.exists(), f'rubric_exists:{path.name}')
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            fail(f'rubric_parse:{path.name}', str(e))
            continue
        missing = REQUIRED_RUBRIC_KEYS - set(data.keys())
        expect(not missing, f'rubric_top_keys:{path.name}', f'missing: {missing}')
        expect(isinstance(data.get('items'), list) and len(data['items']) >= 4,
               f'rubric_items_count:{path.name}', 'need >=4 items')
        for item in data.get('items', []):
            miss = REQUIRED_ITEM_KEYS - set(item.keys())
            expect(not miss, f'item_keys:{path.name}:{item.get("id","?")}', f'missing: {miss}')
        if data['id'] == 'rubric1_pemat':
            scales = {it['scale'] for it in data['items']}
            expect(scales == {'BINARY'}, 'pemat_all_binary', f'got {scales}')
        if data['id'] == 'rubric5_pemat_likert':
            scales = {it['scale'] for it in data['items']}
            expect(scales == {'LIKERT'}, 'pemat_likert_all_likert', f'got {scales}')
            r1 = json.loads((ROOT / 'rubrics/rubric1_pemat.json').read_text())
            ids1 = {it['id'] for it in r1['items']}
            ids5 = {it['id'] for it in data['items']}
            expect(ids1 == ids5, 'pemat_controlled_pair_same_ids',
                   f'rubric1 ids={ids1}, rubric5 ids={ids5}')


# 2. config JSON files
CONFIG_FILES = {
    'config/configs/config_exp1_dataset.json': ['experiment', 'domains', 'output_files'],
    'config/configs/config_exp2_agreement.json': ['judges', 'rubrics', 'benchmark_csv', 'domains', 'output_files'],
    'config/configs/config_exp3_rubric_sensitivity.json': ['judges', 'rubrics', 'scoring_variants', 'output_files'],
    'config/configs/config_exp4_boxplots.json': ['judges', 'rubrics', 'output_files'],
}

def test_configs():
    print('\n[2] config JSON files')
    for path_str, required_keys in CONFIG_FILES.items():
        path = ROOT / path_str
        expect(path.exists(), f'config_exists:{Path(path_str).name}')
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            fail(f'config_parse:{path_str}', str(e))
            continue
        for k in required_keys:
            expect(k in data, f'config_key:{Path(path_str).name}:{k}', f'{k} missing')
        if 'config_exp2' in path_str:
            expect(len(data.get('rubrics', [])) == 5, 'exp2_has_5_rubrics')
            expect(set(data.get('domains', [])) == {'Cardiology','Pharmacology','Neurology','Pediatrics','Emergency'},
                   'exp2_5_domains')
        if 'config_exp3' in path_str:
            rubrics = data.get('rubrics', [])
            expect(any('rubric5' in r for r in rubrics), 'exp3_has_rubric5')
            expect(set(data.get('scoring_variants', [])) == {'BINARY','LIKERT_1_5','SCALED_0_10'},
                   'exp3_3_scoring_variants')


# 3. benchmark CSV
BENCHMARK_CSV_REQUIRED_COLS = [
    'id', 'domain', 'question', 'reference_answer', 'source',
    'expected_class', 'rationale', 'observed_class', 'verified',
    'score_U1_plain', 'score_A1_action', 'score_HB3_emerg', 'score_CE5_clarity'
]
VALID_CLASSES = {'fully_agree', 'majority_agree', 'split', 'full_disagree'}

def test_benchmark_csv():
    print('\n[3] benchmark CSV')
    builder = ROOT / 'benchmark_dataset' / 'build_agreement_dataset.py'
    if not builder.exists():
        print('  SKIP  build_agreement_dataset.py not found')
        return
    result = subprocess.run([sys.executable, str(builder)], capture_output=True, text=True)
    if result.returncode != 0:
        fail('benchmark_csv_build',
             f'exited {result.returncode}: {result.stderr.strip()[-300:]}')
        return
    csv_path = ROOT / 'benchmark_dataset' / 'agreement_benchmark.csv'
    if not csv_path.exists():
        print('  SKIP  CSV not generated')
        return
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)
    missing = [c for c in BENCHMARK_CSV_REQUIRED_COLS if c not in cols]
    expect(not missing, 'benchmark_csv_columns', f'missing: {missing}')
    expect(len(rows) > 0, 'benchmark_csv_nonempty')
    for cls in VALID_CLASSES:
        n = sum(1 for r in rows if r.get('expected_class') == cls)
        expect(n >= 5, f'benchmark_min_5_per_class:{cls}', f'found {n}')
    bad = [r['expected_class'] for r in rows if r.get('expected_class') not in VALID_CLASSES]
    expect(not bad, 'benchmark_valid_classes', f'invalid: {bad[:3]}')


# 4. heuristic scoring unit tests
def test_heuristics():
    print('\n[4] heuristic scoring')
    sys.path.insert(0, str(ROOT / 'benchmark_dataset'))
    try:
        from build_agreement_dataset import (
            expected_agreement_class, score_plain_language,
            score_actionable_steps, score_emergency_flag, classify_domain,
        )
    except ImportError as e:
        fail('heuristics_import', str(e))
        return

    s = score_plain_language('pharmacokinetics acetylcholinesterase bioavailability pathophysiology')
    expect(s < 0.4, 'plain_language_jargon_low', f'got {s:.3f}')
    s2 = score_plain_language('Call your doctor if the patient is confused or falls.')
    expect(s2 > 0.7, 'plain_language_simple_high', f'got {s2:.3f}')
    s3 = score_actionable_steps('Call 911. Administer epinephrine. Monitor breathing. Apply pressure.')
    expect(s3 > 0.4, 'actionable_verbs_high', f'got {s3:.3f}')
    cls, _, _ = expected_agreement_class('patient is unresponsive and not breathing',
                                         'The patient should rest and drink fluids.')
    expect(cls in ('split', 'full_disagree'), 'emergency_no_flag_class', f'got {cls}')
    cls2, _, _ = expected_agreement_class('patient is unresponsive and not breathing',
                                          'Call 911 immediately. Begin CPR. Use AED if available.')
    expect(cls2 in ('fully_agree', 'majority_agree'), 'emergency_with_flag_class', f'got {cls2}')
    expect(classify_domain('STEMI management inferior ST elevation') == 'Cardiology', 'domain_cardiology')
    expect(classify_domain('metformin renal impairment contraindication') == 'Pharmacology', 'domain_pharmacology')
    expect(classify_domain('BLS cardiac arrest unresponsive') == 'Emergency', 'domain_emergency')
    expect(classify_domain('2-month vaccine DTaP ACIP schedule') == 'Pediatrics', 'domain_pediatrics')
    expect(classify_domain('subarachnoid hemorrhage thunderclap headache') == 'Neurology', 'domain_neurology')


# 5. exp1 placeholder
def test_exp1_placeholder():
    print('\n[5] exp1 placeholder fallback')
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'exp1', ROOT / 'experiments' / 'exp1_dataset_analysis.py'
        )
        exp1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp1)
        benchmark = exp1.build_benchmark([], max_per_domain=5)
        expect(len(benchmark) == 25, 'exp1_placeholder_25_questions', f'got {len(benchmark)}')
        expect({q['domain'] for q in benchmark} == {'Cardiology','Pharmacology','Neurology','Pediatrics','Emergency'},
               'exp1_placeholder_all_domains')
        for q in benchmark:
            expect('id' in q and 'text' in q and 'domain' in q,
                   f'exp1_question_schema:{q.get("id","?")}')
    except Exception as e:
        fail('exp1_placeholder_run', str(e))


# 6. agreement math - uses new compute_agr / summarize_agreement API
def test_agreement_math():
    print('\n[6] agreement math')
    try:
        from core.agreement import compute_agr, summarize_agreement, classify_agreement_level
    except ImportError as e:
        fail('agreement_import', str(e))
        return

    # perfect agreement: all judges give identical scores
    kappa = {'c1': 1.0, 'c2': 5.0, 'c3': 1.0}
    scores_identical = {
        'a': {'c1': 1.0, 'c2': 4.0, 'c3': 0.0},
        'b': {'c1': 1.0, 'c2': 4.0, 'c3': 0.0},
        'c': {'c1': 1.0, 'c2': 4.0, 'c3': 0.0},
        'd': {'c1': 1.0, 'c2': 4.0, 'c3': 0.0},
    }
    agr = compute_agr(scores_identical, kappa)
    expect(agr == 1.0, 'math_identical_agr_1', f'got {agr}')
    expect(classify_agreement_level(agr) == 'full_agree', 'math_full_agree', f'got {classify_agreement_level(agr)}')

    # one judge clearly out of step
    scores_outlier = {
        'a': {'c1': 1.0, 'c2': 4.0, 'c3': 1.0},
        'b': {'c1': 1.0, 'c2': 4.0, 'c3': 1.0},
        'c': {'c1': 1.0, 'c2': 4.0, 'c3': 1.0},
        'd': {'c1': 0.0, 'c2': 1.0, 'c3': 0.0},  # outlier
    }
    summary = summarize_agreement(scores_outlier, kappa)
    expect(summary['outlier_judge'] == 'd', 'math_outlier_detected',
           f'got {summary["outlier_judge"]}')

    # low agreement -> disagree level -> report_score False
    scores_disagree = {
        'a': {'c1': 1.0, 'c2': 5.0, 'c3': 1.0},
        'b': {'c1': 0.0, 'c2': 1.0, 'c3': 0.0},
    }
    summary2 = summarize_agreement(scores_disagree, kappa)
    expect(not summary2['report_score'], 'math_low_agr_no_report',
           f'agr={summary2["agr"]} report={summary2["report_score"]}')


if __name__ == '__main__':
    print('Multi-SLM Medical AI Judge -- core tests')
    print()
    test_rubrics()
    test_configs()
    test_benchmark_csv()
    test_heuristics()
    test_exp1_placeholder()
    test_agreement_math()

    total = len(_PASS) + len(_FAIL)
    print(f'\n{len(_PASS)}/{total} passed, {len(_FAIL)} failed')
    if _FAIL:
        print('failed:')
        for f_ in _FAIL:
            print(f'  x {f_}')
        sys.exit(1)
    else:
        print('all tests passed')
        sys.exit(0)
