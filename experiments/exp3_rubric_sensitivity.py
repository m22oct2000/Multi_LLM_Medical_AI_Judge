# exp3: rubric sensitivity - tests each rubric under 3 scoring variants
#   BINARY, LIKERT_1_5, SCALED_0_10
# run: python experiments/exp3_rubric_sensitivity.py
# output: results/exp3_sensitivity_results.json
from __future__ import annotations

import json
import sys
import statistics
from pathlib import Path
from typing import Any, Dict, List
from collections import Counter

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.consensus_core.models import Answer, Question, Rubric, RubricItem, new_id
from core.wrapper import ADRDJudgeRunner

import os as _os

CONFIG_PATH  = Path(_os.environ.get(
    'EXP_CONFIG',
    str(ROOT / 'config' / 'configs' / 'config_exp3_rubric_sensitivity.json'),
))
DATASET_PATH = Path(_os.environ.get(
    'DATASET_PATH',
    str(ROOT / 'benchmark_dataset' / 'source_datasets' / 'benchmark_dataset_500.csv'),
))

SCALE_RANGE = {
    'BINARY':      (0.0, 1.0),
    'LIKERT_1_5':  (1.0, 5.0),
    'SCALED_0_10': (0.0, 10.0),
}


def load_dataset(path, max_questions=0):
    if not path.exists():
        print(f'ERROR: dataset not found: {path}')
        sys.exit(1)
    df = pd.read_csv(path)
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ('question', 'text', 'input', 'prompt'):    col_map[col] = 'text'
        elif cl in ('reference_answer', 'answer', 'output'): col_map[col] = 'reference_answer'
        elif cl == 'domain':                                  col_map[col] = 'domain'
        elif cl in ('id', 'question_id'):                    col_map[col] = 'id'
    df = df.rename(columns=col_map)
    if 'id' not in df.columns:
        df['id'] = [f'q_{i:04d}' for i in range(len(df))]
    if max_questions and max_questions < len(df):
        if 'domain' in df.columns:
            per_domain = max(1, max_questions // len(df['domain'].unique()))
            df = pd.concat([
                grp.head(per_domain) for _, grp in df.groupby('domain')
            ]).head(max_questions).reset_index(drop=True)
        else:
            df = df.head(max_questions)
        print(f'sampled {len(df)} rows')
    rows = df.to_dict('records')
    print(f'loaded {len(rows)} rows from {path.name}')
    for dom, n in sorted(Counter(r.get('domain', '?') for r in rows).items()):
        print(f'  {dom}: {n}')
    return rows


def load_rubric_raw(path):
    with open(ROOT / path) as f:
        return json.load(f)


def build_rubric_variant(rubric_raw, scoring_variant):
    scale_map = {'BINARY': 'BINARY', 'LIKERT_1_5': 'LIKERT', 'SCALED_0_10': 'LIKERT'}
    scale     = scale_map[scoring_variant]
    lo, hi    = SCALE_RANGE[scoring_variant]
    suffix    = '' if scoring_variant == 'LIKERT_1_5' else f' Score {int(lo)}-{int(hi)}.'
    items = [
        RubricItem(
            id=it['id'], name=it['name'],
            description=it['description'] + suffix,
            scale=scale, weight=float(it.get('weight', 1.0)),
            source_paper=rubric_raw.get('source_paper', ''),
        )
        for it in rubric_raw['items']
    ]
    return Rubric(
        id=f"{rubric_raw['id']}_{scoring_variant.lower()}",
        name=f"{rubric_raw['name']} [{scoring_variant}]",
        source_paper=rubric_raw['source_paper'],
        items=items,
    )


def main():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    max_questions = int(config.get('max_questions', 0))
    rows          = load_dataset(DATASET_PATH, max_questions=max_questions)
    runner        = ADRDJudgeRunner(config_path=str(CONFIG_PATH))
    variants      = config['scoring_variants']
    total_rows    = len(rows) * len(config['rubrics']) * len(variants)
    print(f'{len(rows)} questions x {len(config["rubrics"])} rubrics x {len(variants)} variants = {total_rows} rows')

    all_results = []

    for rubric_path in config['rubrics']:
        rubric_raw = load_rubric_raw(rubric_path)
        variant_results = []

        for variant in variants:
            rubric    = build_rubric_variant(rubric_raw, variant)
            pw_scores = []
            print(f"\n{rubric_raw['name']} / {variant}")

            for i, qraw in enumerate(rows, 1):
                q = Question(
                    id=str(qraw['id']),
                    text=str(qraw.get('text', qraw.get('question', ''))),
                    category=str(qraw.get('domain', '')),
                )
                a = Answer(
                    id=new_id('ans'),
                    text=str(qraw.get('reference_answer', '')),
                    provider='reference',
                )
                panel = runner.run(q, a, rubric)
                if not panel.skipped:
                    pw_scores.append(panel.mean_pairwise_agreement)
                if i % 10 == 0:
                    print(f'  [{i}/{len(rows)}] done')

            mean_pw = statistics.mean(pw_scores) if pw_scores else 0.0
            std_pw  = statistics.stdev(pw_scores) if len(pw_scores) > 1 else 0.0
            print(f'  mean={mean_pw:.2f}%  std={std_pw:.2f}%  n={len(pw_scores)}')
            if std_pw < 5.0:
                print(f'  note: low std ({std_pw:.2f}%) - consider merging with adjacent variant')
            variant_results.append({
                'rubric_id':               rubric_raw['id'],
                'scoring_variant':         variant,
                'scale_range':             f'{SCALE_RANGE[variant][0]}-{SCALE_RANGE[variant][1]}',
                'mean_pairwise_agreement': round(mean_pw, 2),
                'std_pairwise_agreement':  round(std_pw, 2),
                'n_questions':             len(pw_scores),
                'per_question_scores':     [round(s, 2) for s in pw_scores],
            })

        all_results.append({
            'rubric_id':   rubric_raw['id'],
            'rubric_name': rubric_raw['name'],
            'variants':    variant_results,
        })

    out_path = ROOT / config['output_files']['results_json']
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nresults -> {out_path}')


if __name__ == '__main__':
    main()
