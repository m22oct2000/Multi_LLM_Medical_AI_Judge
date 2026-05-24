# exp2: per-rubric agreement analysis
# runs all judges on every benchmark row under all 5 rubrics
# writes partial results after each rubric so exp4 can pick up mid-run
#
# env vars:
#   MAX_QUESTIONS  - cap total questions sampled evenly per domain (default 100, 0=all)
#   DATASET_PATH   - override dataset CSV path
#   EXP_CONFIG     - override config JSON path
#
# run:  python experiments/exp2_agreement_analysis.py
# output: results/exp2_agreement_results.json
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.consensus_core.models import Answer, Question, Rubric, RubricItem, new_id
from core.wrapper import ADRDJudgeRunner, PanelResult

import os as _os

CONFIG_PATH   = Path(_os.environ.get(
    'EXP_CONFIG',
    str(ROOT / 'config' / 'configs' / 'config_exp2_agreement.json'),
))
DATASET_PATH  = Path(_os.environ.get(
    'DATASET_PATH',
    str(ROOT / 'benchmark_dataset' / 'source_datasets' / 'benchmark_dataset_500.csv'),
))
MAX_QUESTIONS = int(_os.environ.get('MAX_QUESTIONS', '100'))


def load_rubric(path):
    with open(ROOT / path) as f:
        data = json.load(f)
    items = [
        RubricItem(
            id=it['id'], name=it['name'],
            description=it['description'],
            scale=it['scale'], weight=float(it['weight']),
            source_paper=data.get('source_paper', ''),
        )
        for it in data['items']
    ]
    return Rubric(
        id=data['id'], name=data['name'],
        source_paper=data['source_paper'],
        source_url=data.get('source_url'),
        items=items,
    )


def load_dataset(path, max_questions):
    if not path.exists():
        print(f'ERROR: dataset not found: {path}')
        sys.exit(1)
    df = pd.read_csv(path)
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ('question', 'text', 'input', 'prompt'):         col_map[col] = 'question'
        elif cl in ('reference_answer', 'answer', 'output'):      col_map[col] = 'reference_answer'
        elif cl == 'domain':                                       col_map[col] = 'domain'
        elif cl in ('id', 'question_id'):                         col_map[col] = 'id'
        elif cl == 'source':                                       col_map[col] = 'source'
        elif cl in ('expected_class', 'expected_agreement'):      col_map[col] = 'expected_class'
    df = df.rename(columns=col_map)
    if 'id' not in df.columns:
        df['id'] = [f'q_{i:04d}' for i in range(len(df))]
    if 'expected_class' not in df.columns:
        df['expected_class'] = ''
    if max_questions > 0 and max_questions < len(df):
        per_domain = max(1, max_questions // len(df['domain'].unique()))
        df = (
            df.groupby('domain', group_keys=False)
              .apply(lambda g: g.sample(n=min(per_domain, len(g)), random_state=42))
              .reset_index(drop=True)
        )
        print(f'sampled {len(df)} rows ({per_domain}/domain)')
    else:
        print(f'loaded {len(df)} rows from {path.name}')
    for dom, n in sorted(Counter(df['domain'].tolist()).items()):
        print(f'  {dom}: {n}')
    return df.to_dict('records')


def _print_rubric_summary(block):
    results = block['results']
    live    = [r for r in results if r.get('agreement_class') != 'skipped']
    skipped = len(results) - len(live)
    total   = len(live)
    counts  = Counter(r['agreement_class'] for r in live)
    mean_pw = sum(r['mean_pairwise_agreement'] for r in live) / total if total else 0
    print(f"\n{block['rubric_name']}:")
    for cls in ['fully_agree', 'majority_agree', 'split', 'full_disagree']:
        n = counts.get(cls, 0)
        print(f'  {cls:<20}: {n:3d}/{total} ({n/total*100:.1f}%)' if total else f'  {cls}: 0')
    if skipped:
        print(f'  skipped: {skipped}')
    print(f'  mean pairwise: {mean_pw:.1f}%')


def main():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    rows    = load_dataset(DATASET_PATH, MAX_QUESTIONS)
    rubrics = [load_rubric(r) for r in config['rubrics']]
    runner  = ADRDJudgeRunner(config_path=str(CONFIG_PATH))

    out_path = ROOT / config['output_files']['results_json']
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    already_done = set()
    if out_path.exists():
        try:
            all_results   = json.load(open(out_path))
            already_done  = {b['rubric_id'] for b in all_results}
            print(f'resuming: {len(already_done)} rubric(s) already done')
        except Exception:
            all_results = []

    total   = len(rows) * len(rubrics)
    done    = len(already_done) * len(rows)
    t_start = time.time()
    print(f'\n{len(rows)} questions x {len(rubrics)} rubrics = {total} rows')

    for rubric in rubrics:
        if rubric.id in already_done:
            print(f'skipping done rubric: {rubric.id}')
            continue

        print(f'\nrubric: {rubric.name}  ({rubric.source_paper})')
        rubric_results = []

        for row in rows:
            question = Question(
                id=str(row['id']),
                text=str(row.get('question', row.get('text', ''))),
                category=str(row.get('domain', '')),
                source=str(row.get('source', '')),
            )
            answer = Answer(
                id=new_id('ans'),
                text=str(row.get('reference_answer', '')),
                provider='reference',
            )
            expected = str(row.get('expected_class', ''))
            panel: PanelResult = runner.run(question, answer, rubric)
            observed = panel.agreement_class
            n_ran = len([jr for jr in panel.judge_results
                         if jr.raw_response.strip() or any(
                             str(s['score']).upper() not in ('', 'NA', 'N/A', 'NONE')
                             for s in jr.scores
                         )])
            if n_ran < len(runner.judges):
                print(f'  warning: only {n_ran}/{len(runner.judges)} judges ran')
            rubric_results.append(panel.to_dict())
            done += 1
            match = '\u2713' if observed == expected and expected else '-'
            print(f'  [{done}/{total}] {match} expected={expected} observed={observed} '
                  f'pw={panel.mean_pairwise_agreement:.1f}% '
                  f'judges={n_ran} t={time.time()-t_start:.0f}s')
            if panel.outlier_judge:
                print(f'  outlier: {panel.outlier_judge}')

        block = {
            'rubric_id':    rubric.id,
            'rubric_name':  rubric.name,
            'source_paper': rubric.source_paper,
            'results':      rubric_results,
        }
        all_results.append(block)
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'partial save -> {out_path}')
        _print_rubric_summary(block)

    print('\nagreement summary:')
    for block in all_results:
        _print_rubric_summary(block)
    print(f'\nresults -> {out_path}')


if __name__ == '__main__':
    main()
