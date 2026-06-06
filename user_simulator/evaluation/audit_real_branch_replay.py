from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fail-on-critical-error', action='store_true')
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})


def build_md(summary: dict) -> str:
    lines = [
        '# Real Branch Replay Audit',
        '',
        f"- status: `{summary['status']}`",
        f"- snapshot_count: `{summary['snapshot_count']}`",
        f"- branch_count: `{summary['branch_count']}`",
        f"- follow/ignore/over_apply: `{summary['follow_count']}/{summary['ignore_count']}/{summary['over_apply_count']}`",
        f"- positive/zero/negative uplift: `{summary['positive_uplift_count']}/{summary['zero_uplift_count']}/{summary['negative_uplift_count']}`",
        f"- identical_branch_groups: `{summary['identical_branch_groups']}`",
        '',
        '## Critical Errors',
    ]
    if not summary['critical_errors']:
        lines.append('- none')
    else:
        for error in summary['critical_errors']:
            lines.append(f'- {error}')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshots = read_jsonl(input_dir / 'replay_snapshots.jsonl')
    branches = read_jsonl(input_dir / 'branch_rollouts.jsonl')
    pairs = read_jsonl(input_dir / 'replay_pairs.jsonl')
    failures = read_jsonl(input_dir / 'replay_failures.jsonl')

    errors: list[dict] = []
    critical_errors: list[str] = []

    if not snapshots:
        critical_errors.append('snapshot count = 0')
    if not branches:
        critical_errors.append('branch count = 0')

    branch_by_snapshot: dict[str, list[dict]] = defaultdict(list)
    for row in branches:
        branch_by_snapshot[str(row.get('snapshot_id', ''))].append(row)
        if not row.get('provenance'):
            errors.append({'row_type': 'branch', 'snapshot_id': row.get('snapshot_id'), 'error': 'missing provenance'})
        if row.get('utility_breakdown') in (None, {}, ''):
            errors.append({'row_type': 'branch', 'snapshot_id': row.get('snapshot_id'), 'error': 'missing utility breakdown'})
        if row.get('utility_total') in (None, ''):
            errors.append({'row_type': 'branch', 'snapshot_id': row.get('snapshot_id'), 'error': 'missing utility total'})
        if not row.get('snapshot'):
            errors.append({'row_type': 'branch', 'snapshot_id': row.get('snapshot_id'), 'error': 'missing snapshot payload'})

    branch_types = Counter(row.get('branch_type', 'UNKNOWN') for row in branches)
    if any(branch_types.get(branch, 0) == 0 for branch in ['follow', 'ignore', 'over_apply']):
        critical_errors.append('one or more branches are completely missing')

    identical_groups = 0
    for snapshot_id, group in branch_by_snapshot.items():
        if len(group) >= 2:
            canonical = [json.dumps(row.get('trajectory', []), sort_keys=True, ensure_ascii=False) for row in group]
            if len(set(canonical)) == 1:
                identical_groups += 1
    if branches and identical_groups == len(branch_by_snapshot):
        critical_errors.append('all branch trajectories are identical')

    if not pairs:
        errors.append({'row_type': 'pair', 'error': 'no replay pairs'})
    if pairs and any(not row.get('provenance') for row in pairs):
        critical_errors.append('pair provenance missing')

    uplift_values = [float(row.get('uplift', 0.0)) for row in pairs if row.get('uplift') is not None]
    positive_uplift_count = sum(1 for value in uplift_values if value > 0)
    zero_uplift_count = sum(1 for value in uplift_values if value == 0)
    negative_uplift_count = sum(1 for value in uplift_values if value < 0)

    task_summary_rows = []
    for snapshot in snapshots:
        task_summary_rows.append(
            {
                'task_type': snapshot.get('task_type', 'UNKNOWN'),
                'provenance': snapshot.get('provenance', 'UNKNOWN'),
            }
        )
    task_counts = Counter(row['task_type'] for row in task_summary_rows)
    task_summary_csv = [{'task_type': task, 'count': count} for task, count in sorted(task_counts.items())]
    branch_summary_csv = []
    by_branch = Counter((row.get('branch_type', 'UNKNOWN'), row.get('status', 'UNKNOWN')) for row in branches)
    for (branch_type, status), count in sorted(by_branch.items()):
        branch_summary_csv.append({'branch_type': branch_type, 'status': status, 'count': count})
    pair_quality_csv = []
    by_pair_status = Counter(row.get('metadata', {}).get('pair_status', 'UNKNOWN') for row in pairs)
    for pair_status, count in sorted(by_pair_status.items()):
        pair_quality_csv.append({'pair_status': pair_status, 'count': count})
    uplift_summary_csv = [
        {
            'count': len(uplift_values),
            'mean': mean(uplift_values) if uplift_values else 0.0,
            'min': min(uplift_values) if uplift_values else 0.0,
            'max': max(uplift_values) if uplift_values else 0.0,
            'positive_count': positive_uplift_count,
            'zero_count': zero_uplift_count,
            'negative_count': negative_uplift_count,
        }
    ]

    summary = {
        'status': 'FAIL' if critical_errors else 'PARTIAL' if errors else 'PASS',
        'snapshot_count': len(snapshots),
        'branch_count': len(branches),
        'pair_count': len(pairs),
        'follow_count': branch_types.get('follow', 0),
        'ignore_count': branch_types.get('ignore', 0),
        'over_apply_count': branch_types.get('over_apply', 0),
        'recommend_count': sum(1 for row in snapshots if row.get('task_type') == 'recommend'),
        'ask_count': sum(1 for row in snapshots if row.get('task_type') == 'ask'),
        'search_count': sum(1 for row in snapshots if row.get('task_type') == 'search'),
        'generic_count': sum(1 for row in snapshots if row.get('task_type') not in {'recommend', 'ask', 'search'}),
        'successful_replay_count': sum(1 for row in branches if row.get('status') == 'COMPLETED'),
        'failed_replay_count': len(failures),
        'parse_failure_count': sum(
            1
            for row in branches
            for step in row.get('trajectory', [])
            if str(step.get('parser_status', 'OK')).upper() not in {'OK', 'RECOVERED'}
        ),
        'tool_failure_count': sum(
            1
            for row in branches
            for step in row.get('trajectory', [])
            if str(step.get('tool_status', 'NO_TOOL')).upper() == 'FAIL'
        ),
        'terminal_count': sum(
            1
            for row in branches
            for step in row.get('trajectory', [])
            if str(step.get('terminal_status', 'ACTIVE')).upper() != 'ACTIVE'
        ),
        'average_horizon': mean([len(row.get('trajectory', [])) for row in branches]) if branches else 0.0,
        'positive_uplift_count': positive_uplift_count,
        'zero_uplift_count': zero_uplift_count,
        'negative_uplift_count': negative_uplift_count,
        'identical_branch_count': identical_groups,
        'missing_utility_count': sum(1 for row in errors if 'utility' in row.get('error', '').lower()),
        'missing_breakdown_count': sum(1 for row in errors if 'breakdown' in row.get('error', '').lower()),
        'missing_provenance_count': sum(1 for row in errors if 'provenance' in row.get('error', '').lower()),
        'critical_errors': critical_errors,
        'row_errors': errors,
        'identical_branch_groups': identical_groups,
    }

    write_json(output_dir / 'audit.json', summary)
    (output_dir / 'audit.md').write_text(build_md(summary), encoding='utf-8')
    write_csv(output_dir / 'task_type_summary.csv', task_summary_csv, ['task_type', 'count'])
    write_csv(output_dir / 'branch_summary.csv', branch_summary_csv, ['branch_type', 'status', 'count'])
    write_csv(output_dir / 'uplift_summary.csv', uplift_summary_csv, ['count', 'mean', 'min', 'max', 'positive_count', 'zero_count', 'negative_count'])
    write_csv(output_dir / 'pair_quality_summary.csv', pair_quality_csv, ['pair_status', 'count'])
    write_jsonl(output_dir / 'row_errors.jsonl', errors)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.fail_on_critical_error and critical_errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
