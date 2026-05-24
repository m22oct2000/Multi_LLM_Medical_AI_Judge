# exp1: dataset breakdown - prints Table 1 numbers and saves markdown + JSON
# run: python experiments/exp1_dataset_analysis.py
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os as _os

CONFIG_PATH  = Path(_os.environ.get(
    'EXP_CONFIG',
    str(ROOT / 'config' / 'configs' / 'config_exp1_dataset.json'),
))
DATASET_PATH = Path(_os.environ.get(
    'DATASET_PATH',
    str(ROOT / 'benchmark_dataset' / 'source_datasets' / 'benchmark_dataset_500.csv'),
))

DOMAIN_DEFINITIONS = {
    'Cardiology':   {'summary': 'STEMI management, 65-year-old, inferior ST elevation',
                     'example_q': 'A 65-year-old presents with inferior ST elevation. What is the management?'},
    'Pharmacology': {'summary': 'Metformin contraindications and renal safety criteria',
                     'example_q': 'What are the contraindications for metformin use in patients with renal impairment?'},
    'Neurology':    {'summary': 'Thunderclap headache workup, SAH rule-out protocol',
                     'example_q': 'A patient presents with a thunderclap headache. How do you rule out SAH?'},
    'Pediatrics':   {'summary': '2-month vaccination schedule, US CDC ACIP guidelines',
                     'example_q': 'What vaccines are recommended at the 2-month well-child visit per CDC ACIP?'},
    'Emergency':    {'summary': 'BLS protocol, unresponsive non-breathing patient',
                     'example_q': 'What are the steps for BLS in an unresponsive, non-breathing adult?'},
}


def build_benchmark(source_rows, max_per_domain=20):
    """Build a flat list of question dicts. Falls back to placeholder questions
    when source_rows is empty (used by tests)."""
    domains = list(DOMAIN_DEFINITIONS.keys())
    if not source_rows:
        placeholder = []
        for i, dom in enumerate(domains):
            for j in range(max_per_domain):
                placeholder.append({
                    'id':     f'{dom[:3].lower()}_{j:03d}',
                    'text':   DOMAIN_DEFINITIONS[dom]['example_q'],
                    'domain': dom,
                })
        return placeholder
    return [
        {'id': str(r.get('id', i)), 'text': str(r.get('question', r.get('text', ''))),
         'domain': str(r.get('domain', ''))}
        for i, r in enumerate(source_rows)
    ]


def generate_table_md(df):
    counts  = df['domain'].value_counts().sort_index()
    sources = df['source'].value_counts()
    lines = [
        '# Benchmark Dataset',
        '',
        '| Domain | Summary | # Questions |',
        '|---|---|---|',
    ]
    for domain, defn in DOMAIN_DEFINITIONS.items():
        n = counts.get(domain, 0)
        lines.append(f'| {domain} | {defn["summary"]} | {n} |')
    lines += [
        f'| **Total** | | **{len(df)}** |',
        '',
        '| Domain | Representative Question |',
        '|---|---|',
    ]
    for domain, defn in DOMAIN_DEFINITIONS.items():
        lines.append(f'| {domain} | {defn["example_q"]} |')
    lines += [
        '',
        '| Source | # Questions |',
        '|---|---|',
    ]
    for src, n in sources.items():
        lines.append(f'| {src} | {n} |')
    return '\n'.join(lines)


def main():
    if not DATASET_PATH.exists():
        print(f'ERROR: dataset not found: {DATASET_PATH}')
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    df = pd.read_csv(DATASET_PATH)
    print(f'exp1: loaded {len(df)} rows from {DATASET_PATH.name}')
    print(f'columns: {list(df.columns)}')

    counts  = df['domain'].value_counts().sort_index()
    sources = df['source'].value_counts()

    print('\ndomain breakdown:')
    for domain in DOMAIN_DEFINITIONS:
        print(f'  {domain:15s}: {counts.get(domain, 0):4d}')
    print(f'  {"total":15s}: {len(df):4d}')

    print('\nsource breakdown:')
    for src, n in sources.items():
        print(f'  {src:25s}: {n:4d}')

    table_md = generate_table_md(df)
    table_path = ROOT / 'benchmark_dataset' / 'dataset_table.md'
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, 'w') as f:
        f.write(table_md)
    print(f'table -> {table_path}')

    results_path = ROOT / config['output_files']['results_json']
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            'dataset_path':  str(DATASET_PATH),
            'total':         len(df),
            'domain_counts': counts.to_dict(),
            'source_counts': sources.to_dict(),
        }, f, indent=2)
    print(f'results -> {results_path}')


if __name__ == '__main__':
    main()
