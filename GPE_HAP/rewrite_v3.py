import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

from rewrite_func import refine_rec_trajectories, refine_ask_trajectories, refine_search_trajectories

try:
    from user_simulator.persona.model.model import OpenAIClient
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from user_simulator.persona.model.model import OpenAIClient


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all', help='Comma-separated modes: recommend, search, ask, or all')
    parser.add_argument('--domain', type=str, required=True, help='Domain name (e.g., Book, Game, Yelp), case-sensitive')
    parser.add_argument('--config_path', type=str, default='config/api_config.json', help='Path to api_config.json')
    parser.add_argument('--index_root', type=str, default='raw_data/emb', help='Root path to FAISS index folders')
    parser.add_argument('--model_path', type=str, default='crs/tools/all-MiniLM-L6-v2', help='Embedding model path')
    parser.add_argument('--sample_num', '--sample-limit', dest='sample_num', type=int, default=2, help='Number of samples to process')
    parser.add_argument('--task_limit', '--task-limit', dest='task_limit', type=int, default=None, help='Max number of tasks to refine across each mode')
    parser.add_argument('--input_root', type=str, default=None, help='Directory containing recommend_data.json / ask_data.json / search_data.json')
    parser.add_argument('--output_dir', type=str, default=None, help='Explicit output directory')
    parser.add_argument('--base_url', type=str, default=None, help='OpenAI-compatible base URL override')
    parser.add_argument('--api_key', type=str, default=None, help='OpenAI-compatible API key override')
    parser.add_argument('--model_name', type=str, default=None, help='Served model alias override')
    parser.add_argument('--mini_base_url', type=str, default=None, help='Mini-model base URL override')
    parser.add_argument('--mini_api_key', type=str, default=None, help='Mini-model API key override')
    parser.add_argument('--mini_model_name', type=str, default=None, help='Mini-model alias override')
    parser.add_argument('--no_potential', dest='potential', action='store_false', help='Disable potential function refinement')
    parser.add_argument('--no_valid', dest='valid', action='store_false', help='Disable validation')
    parser.set_defaults(potential=True, valid=True)
    return parser.parse_args()


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def load_config(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def build_client(config: dict, base_url: str | None, api_key: str | None, model_name: str | None):
    return OpenAIClient(
        base_url=base_url or config.get('base_url') or 'http://127.0.0.1:8000/v1',
        api_key=api_key or config.get('api_key') or 'EMPTY',
        model_path=model_name or config.get('model_path') or 'qwen2.5-3b-instruct',
    )


def resolve_input_root(repo_root: Path, domain: str, input_root: str | None) -> Path:
    if input_root:
        return Path(input_root)
    return repo_root / 'your' / 'work' / 'dir' / domain / 'bpo'


def resolve_dataset_path(input_root: Path, filename: str) -> Path:
    candidate = input_root / filename
    if candidate.exists():
        return candidate
    candidate = input_root / 'bpo' / filename
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f'Missing required input file: {filename} under {input_root}')


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    mode = args.mode.lower()
    if mode == 'all':
        active_modes = {'recommend', 'search', 'ask'}
    else:
        active_modes = {m.strip() for m in mode.split(',') if m.strip()}

    input_root = resolve_input_root(repo_root, args.domain, args.input_root)
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / 'outputs' / 'server184_gimo' / 'gpe_hap_smoke' / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config_path)
    config = load_config(config_path) if config_path.exists() else {}
    openai_cfg = config.get('openai', {})
    openai_mini_cfg = config.get('openai_mini', openai_cfg)

    openai_client = build_client(openai_cfg, args.base_url, args.api_key, args.model_name)
    openai_mini_client = build_client(openai_mini_cfg, args.mini_base_url or args.base_url, args.mini_api_key or args.api_key, args.mini_model_name or args.model_name)

    rec_tasks = []
    ask_tasks = []
    search_tasks = []
    if 'recommend' in active_modes:
        rec_data = load_json(resolve_dataset_path(input_root, 'recommend_data.json'))
        rec_tasks = [(i, d['Instruction'], d['output'], d['ground_truth'], d.get('is_correct'), d.get('is_recall')) for i, d in enumerate(rec_data)]
    if 'ask' in active_modes:
        ask_data = load_json(resolve_dataset_path(input_root, 'ask_data.json'))
        ask_tasks = [(i, d['Instruction'], d['output'], d['ground_truth']) for i, d in enumerate(ask_data)]
    if 'search' in active_modes:
        search_data = load_json(resolve_dataset_path(input_root, 'search_data.json'))
        search_tasks = [(i, d['Instruction'], d['output'], d['gt_title']) for i, d in enumerate(search_data)]

    if args.task_limit is not None:
        rec_tasks = rec_tasks[: args.task_limit]
        ask_tasks = ask_tasks[: args.task_limit]
        search_tasks = search_tasks[: args.task_limit]

    sft_results = []
    dpo_results = []
    all_logs = []

    if 'recommend' in active_modes and rec_tasks:
        rec_sft, rec_dpo, rec_logs = refine_rec_trajectories(rec_tasks, openai_mini_client, args.sample_num, args.potential, args.valid, max_workers=4)
        sft_results.extend(rec_sft)
        dpo_results.extend(rec_dpo)
        all_logs.extend(rec_logs)

    if 'ask' in active_modes and ask_tasks:
        ask_sft, ask_dpo, ask_logs = refine_ask_trajectories(args.domain, ask_tasks, openai_mini_client, args.sample_num, args.potential, args.valid, max_workers=4)
        sft_results.extend(ask_sft)
        dpo_results.extend(ask_dpo)
        all_logs.extend(ask_logs)

    if 'search' in active_modes and search_tasks:
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise RuntimeError('search mode requires faiss and sentence_transformers in the active environment') from exc
        index_path = Path(args.index_root) / args.domain / 'faiss_index.bin'
        metadata_path = Path(args.index_root) / args.domain / 'metadata.json'
        emb_model = SentenceTransformer(args.model_path)
        index = faiss.read_index(str(index_path))
        metadata = load_json(metadata_path)
        search_sft, search_dpo, search_logs = refine_search_trajectories(search_tasks, openai_mini_client, index, metadata, emb_model, args.sample_num, args.potential, args.valid, max_workers=4)
        sft_results.extend(search_sft)
        dpo_results.extend(search_dpo)
        all_logs.extend(search_logs)

    random.shuffle(sft_results)
    random.shuffle(dpo_results)

    (output_dir / f'{args.domain}_sto_sft_v1_sample{args.sample_num}.json').write_text(json.dumps(sft_results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output_dir / f'{args.domain}_sto_dpo_v1_sample{args.sample_num}.json').write_text(json.dumps(dpo_results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    refine_log_path = output_dir / f'{args.domain}_refine_log_sample{args.sample_num}.json'
    refine_log_path.write_text(json.dumps(all_logs, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'run_metadata.json').write_text(json.dumps({
        'status': 'ok',
        'domain': args.domain,
        'modes': sorted(active_modes),
        'input_root': str(input_root),
        'output_dir': str(output_dir),
        'sample_num': args.sample_num,
        'task_limit': args.task_limit,
        'log_count': len(all_logs),
        'sft_count': len(sft_results),
        'dpo_count': len(dpo_results),
        'base_url': args.base_url or openai_mini_cfg.get('base_url') or openai_cfg.get('base_url'),
        'model_name': args.model_name or args.mini_model_name or openai_mini_cfg.get('model_path') or openai_cfg.get('model_path'),
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'ok', 'output_dir': str(output_dir), 'refine_log': str(refine_log_path), 'log_count': len(all_logs)}, indent=2))


if __name__ == '__main__':
    main()
