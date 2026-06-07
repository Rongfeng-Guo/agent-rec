#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from genrec.memory.data_adapter import build_eval_samples, load_item_embeddings, load_item_metadata, load_item_sids, load_train_item_set
from run_oracle_route_memory_eval import build_memory, evaluate_mode

TOPKS = (10, 20, 50)


def summarize_match_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            'sample_count': 0,
            'Recall@10': 0.0,
            'Recall@20': 0.0,
            'Recall@50': 0.0,
            'NDCG@10': 0.0,
            'NDCG@20': 0.0,
            'NDCG@50': 0.0,
            'MRR@10': 0.0,
            'MRR@20': 0.0,
            'MRR@50': 0.0,
        }
    out: Dict[str, Any] = {'sample_count': len(rows)}
    for k in TOPKS:
        recalls = []
        ndcgs = []
        mrrs = []
        for row in rows:
            rank = row.get('match_rank')
            hit = rank is not None and int(rank) <= k
            recalls.append(1.0 if hit else 0.0)
            ndcgs.append(0.0 if not hit else 1.0 / math.log2(int(rank) + 1))
            mrrs.append(0.0 if not hit else 1.0 / float(rank))
        out[f'Recall@{k}'] = sum(recalls) / len(recalls)
        out[f'NDCG@{k}'] = sum(ndcgs) / len(ndcgs)
        out[f'MRR@{k}'] = sum(mrrs) / len(mrrs)
    return out


def group_rows(rows: Sequence[Mapping[str, Any]], sample_meta: Mapping[str, Mapping[str, str]]) -> Dict[Tuple[str, str], List[Mapping[str, Any]]]:
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        meta = sample_meta[str(row['sample_id'])]
        subset = meta['subset']
        domain = meta['domain']
        groups[(subset, 'ALL')].append(row)
        groups[(subset, domain)].append(row)
    return groups


def compute_oracle_rows(data_dir: str, item_embedding_path: str, item_sid_path: str, history_len: int) -> List[Dict[str, Any]]:
    embeddings = load_item_embeddings(data_dir, item_embedding_path)
    item_sids = load_item_sids(data_dir, item_sid_path)
    item_metadata = load_item_metadata(data_dir)
    train_item_set = load_train_item_set(data_dir)
    samples = build_eval_samples(data_dir, split='test', cold_only=False)
    sample_meta = {
        str(sample['sample_id']): {
            'subset': 'cold' if bool(sample.get('cold')) else 'warm',
            'domain': str(sample['domain']),
        }
        for sample in samples
    }

    method_specs = [
        {
            'method_key': 'metadata_global',
            'display_name': 'Metadata Global',
            'family': 'baseline',
            'selection_status': 'fixed_baseline',
            'claimable': True,
            'is_upper_bound': False,
            'notes': 'No-route metadata retrieval baseline.',
            'mode': 'metadata',
            'prefix_len': 1,
        },
        {
            'method_key': 'oracle_route_p1',
            'display_name': 'Oracle Route P1',
            'family': 'upper_bound',
            'selection_status': 'oracle_upper_bound',
            'claimable': False,
            'is_upper_bound': True,
            'notes': 'Target prefix-1 route injected; upper bound only.',
            'mode': 'oracle_route',
            'prefix_len': 1,
        },
        {
            'method_key': 'oracle_route_p2',
            'display_name': 'Oracle Route P2',
            'family': 'upper_bound',
            'selection_status': 'oracle_upper_bound',
            'claimable': False,
            'is_upper_bound': True,
            'notes': 'Target full prefix-2 route injected; upper bound only.',
            'mode': 'oracle_route',
            'prefix_len': 2,
        },
    ]

    rows: List[Dict[str, Any]] = []
    for spec in method_specs:
        memory, _ = build_memory(embeddings, item_sids, item_metadata, spec['prefix_len'])
        _, per_sample = evaluate_mode(
            mode=spec['mode'],
            prefix_len=spec['prefix_len'],
            samples=samples,
            embeddings=embeddings,
            item_sids=item_sids,
            train_item_set=train_item_set,
            memory=memory,
            topks=list(TOPKS),
            history_len=history_len,
            predicted_routes={},
        )
        grouped = group_rows(per_sample, sample_meta)
        for (subset, domain), subset_rows in sorted(grouped.items()):
            metric = summarize_match_rows(subset_rows)
            rows.append({
                **spec,
                'subset': subset,
                'domain': domain,
                **metric,
            })
    return rows


def load_summary_rows(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding='utf-8'))


def extract_eval_rows(
    summary_path: Path,
    summary_by_domain_path: Path,
    *,
    method_key: str,
    display_name: str,
    family: str,
    selection_status: str,
    claimable: bool,
    is_upper_bound: bool,
    notes: str,
    query_source: str,
    mode: str,
) -> List[Dict[str, Any]]:
    if not summary_path.exists() or not summary_by_domain_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for row in load_summary_rows(summary_path):
        if row.get('query_source') == query_source and row.get('mode') == mode:
            rows.append({
                'method_key': method_key,
                'display_name': display_name,
                'family': family,
                'selection_status': selection_status,
                'claimable': claimable,
                'is_upper_bound': is_upper_bound,
                'notes': notes,
                'subset': row['subset'],
                'domain': 'ALL',
                'sample_count': row['sample_count'],
                'Recall@10': row['Recall@10'],
                'Recall@20': row['Recall@20'],
                'Recall@50': row['Recall@50'],
                'NDCG@10': row['NDCG@10'],
                'NDCG@20': row['NDCG@20'],
                'NDCG@50': row['NDCG@50'],
                'MRR@10': row['MRR@10'],
                'MRR@20': row['MRR@20'],
                'MRR@50': row['MRR@50'],
            })
    for row in load_summary_rows(summary_by_domain_path):
        if row.get('query_source') == query_source and row.get('mode') == mode:
            rows.append({
                'method_key': method_key,
                'display_name': display_name,
                'family': family,
                'selection_status': selection_status,
                'claimable': claimable,
                'is_upper_bound': is_upper_bound,
                'notes': notes,
                'subset': row['subset'],
                'domain': row['domain'],
                'sample_count': row['sample_count'],
                'Recall@10': row['Recall@10'],
                'Recall@20': row['Recall@20'],
                'Recall@50': row['Recall@50'],
                'NDCG@10': row['NDCG@10'],
                'NDCG@20': row['NDCG@20'],
                'NDCG@50': row['NDCG@50'],
                'MRR@10': row['MRR@10'],
                'MRR@20': row['MRR@20'],
                'MRR@50': row['MRR@50'],
            })
    return rows


def render_markdown(rows: Sequence[Mapping[str, Any]], missing_baseline_note: str) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f'{v:.4f}'
        return str(v)

    official = [row for row in rows if row['claimable']]
    diagnostics = [row for row in rows if not row['claimable']]
    lines = [
        '# Official Retrieval Comparison',
        '',
        'This table uses the same `user_simulator/test` split, the same warm/cold partition, and the same Recall/NDCG/MRR metrics.',
        '',
        f'- Missing baseline note: {missing_baseline_note}',
        '',
        '## Official Claimable Rows',
        '',
        '| method | subset | domain | samples | Recall@10 | Recall@20 | Recall@50 | NDCG@50 | MRR@50 | notes |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in sorted(official, key=lambda x: (x['display_name'], x['subset'], x['domain'])):
        lines.append(
            f"| {row['display_name']} | {row['subset']} | {row['domain']} | {row['sample_count']} | {fmt(row['Recall@10'])} | {fmt(row['Recall@20'])} | {fmt(row['Recall@50'])} | {fmt(row['NDCG@50'])} | {fmt(row['MRR@50'])} | {row['notes']} |"
        )
    lines.extend([
        '',
        '## Diagnostic / Non-claim Rows',
        '',
        '| method | subset | domain | samples | Recall@10 | Recall@20 | Recall@50 | NDCG@50 | MRR@50 | notes |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---|',
    ])
    for row in sorted(diagnostics, key=lambda x: (x['display_name'], x['subset'], x['domain'])):
        lines.append(
            f"| {row['display_name']} | {row['subset']} | {row['domain']} | {row['sample_count']} | {fmt(row['Recall@10'])} | {fmt(row['Recall@20'])} | {fmt(row['Recall@50'])} | {fmt(row['NDCG@50'])} | {fmt(row['MRR@50'])} | {row['notes']} |"
        )
    claimable_cold = [row for row in official if row['subset'] == 'cold' and row['domain'] == 'ALL']
    best_claimable = max(claimable_cold, key=lambda x: x['Recall@50']) if claimable_cold else None
    lines.extend([
        '',
        '## Readout',
        '',
    ])
    if best_claimable:
        lines.append(
            f"- Best claimable cold row in current evidence: `{best_claimable['display_name']}` with Recall@50 `{best_claimable['Recall@50']:.4f}`."
        )
    diag_best = [row for row in diagnostics if row['subset'] == 'cold' and row['domain'] == 'ALL']
    if diag_best:
        best_diag = max(diag_best, key=lambda x: x['Recall@50'])
        lines.append(
            f"- Best diagnostic-only cold row: `{best_diag['display_name']}` with Recall@50 `{best_diag['Recall@50']:.4f}`."
        )
    lines.append('- No checked-in same-metric full-SID / generative-paper baseline was found in the current repo state, so surpassing that method is still unproven here.')
    return '\n'.join(lines) + '\n'


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        'method_key', 'display_name', 'family', 'selection_status', 'claimable', 'is_upper_bound', 'subset', 'domain', 'sample_count',
        'Recall@10', 'Recall@20', 'Recall@50', 'NDCG@10', 'NDCG@20', 'NDCG@50', 'MRR@10', 'MRR@20', 'MRR@50', 'notes'
    ]
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build an official like-for-like retrieval comparison table.')
    parser.add_argument('--data-dir', default='user_simulator')
    parser.add_argument('--item-embedding-path', default='outputs/oracle_route_memory/assets/metadata_embeddings/item_embeddings.npy')
    parser.add_argument('--item-sid-path', default='outputs/oracle_route_memory/assets/proxy_routes_b16_d2/item_sid_mapping.json')
    parser.add_argument('--history-len', type=int, default=5)
    parser.add_argument('--locked-eval-dir', default='outputs/oracle_route_memory/validation_fusion_locked_cold_gpu0')
    parser.add_argument('--diagnostic-eval-dir', default='outputs/oracle_route_memory/prefix1_query_fusion_roundrobin_20260607_002130')
    parser.add_argument('--locked-method-key', default='predicted_route_validation_selected')
    parser.add_argument('--locked-display-name', default='Predicted Route Validation-Selected')
    parser.add_argument('--locked-selection-status', default='validation_selected_tested_once')
    parser.add_argument('--locked-notes', default='Official validation-selected config tested once on cold/warm; this is the only predicted-route row allowed in the main claim table.')
    parser.add_argument('--diagnostic-method-key', default='predicted_route_diagnostic_fusion_lr_p1t4')
    parser.add_argument('--diagnostic-display-name', default='Predicted Route Diagnostic Fusion LR P1T4')
    parser.add_argument('--diagnostic-selection-status', default='diagnostic_cold_best')
    parser.add_argument('--diagnostic-notes', default='Diagnostic cold-best fusion; not validation-selected, so excluded from official claim table.')
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = compute_oracle_rows(args.data_dir, args.item_embedding_path, args.item_sid_path, args.history_len)
    rows.extend(
        extract_eval_rows(
            Path(args.locked_eval_dir) / 'summary.json',
            Path(args.locked_eval_dir) / 'summary_by_domain.json',
            method_key=args.locked_method_key,
            display_name=args.locked_display_name,
            family='predicted_route',
            selection_status=args.locked_selection_status,
            claimable=True,
            is_upper_bound=False,
            notes=args.locked_notes,
            query_source='selected_policy',
            mode='validation_selected',
        )
    )
    rows.extend(
        extract_eval_rows(
            Path(args.diagnostic_eval_dir) / 'summary.json',
            Path(args.diagnostic_eval_dir) / 'summary_by_domain.json',
            method_key=args.diagnostic_method_key,
            display_name=args.diagnostic_display_name,
            family='predicted_route',
            selection_status=args.diagnostic_selection_status,
            claimable=False,
            is_upper_bound=False,
            notes=args.diagnostic_notes,
            query_source='fusion',
            mode='fusion_lr_p1t4',
        )
    )

    missing_baseline_note = 'No same-metric full-SID / generative-paper baseline artifact was found in the current repo state on the same split and metrics.'
    payload = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'args': vars(args),
        'missing_baseline_note': missing_baseline_note,
        'rows': rows,
    }
    (output_dir / 'comparison.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    write_csv(output_dir / 'comparison.csv', rows)
    (output_dir / 'comparison.md').write_text(render_markdown(rows, missing_baseline_note), encoding='utf-8')
    print(json.dumps({'status': 'ok', 'output_dir': str(output_dir.resolve()), 'row_count': len(rows)}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
