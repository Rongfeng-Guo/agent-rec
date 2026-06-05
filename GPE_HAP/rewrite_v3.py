import json
import re
import os
import sys
import argparse
import random
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import faiss
from sentence_transformers import SentenceTransformer
# 获取父级目录路径
from rewrite_func import refine_rec_trajectories, refine_ask_trajectories, refine_search_trajectories
from refine_prompts_v2 import ask_potential_function_template, recommendation_potential_function_template, search_potential_function_template, ask_policy_improvement_template, recommendation_policy_improvement_template, search_policy_improvement_template, ask_potential_eval_template, recommendation_potential_eval_template
# 获取当前工作目录
current_dir = os.getcwd()

# 获取父级目录路径
parent_dir = os.path.abspath(os.path.join(current_dir, '../../'))

# 将父级目录添加到 sys.path
sys.path.insert(0, parent_dir)
from model.model import OpenAIClient

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all", help="Comma-separated modes: recommend, search, ask, or all")
    parser.add_argument("--domain", type=str, required=True, help="Domain name (e.g., Book, Game, Yelp), case-sensitive")
    parser.add_argument("--config_path", type=str, default="../../config/api_config.json", help="Path to api_config.json")
    parser.add_argument("--index_root", type=str, default="../../raw_data/emb", help="Root path to FAISS index folders")
    parser.add_argument("--model_path", type=str, default="../../crs/tools/all-MiniLM-L6-v2", help="Embedding model path")
    parser.add_argument("--sample_num", type=int, default=2, help="Number of samples to process")
    parser.add_argument("--task_limit", type=int, default=None, help="Max number of tasks to refine (across each mode)")
    parser.add_argument("--data_root", type=str, default=None, help="Directory containing recommend_data.json / ask_data.json / search_data.json")
    parser.add_argument("--output_root", type=str, default=None, help="Base directory for timestamped rollout outputs")
    parser.add_argument("--output_dir", type=str, default=None, help="Explicit output directory. Overrides --output_root.")
    parser.add_argument("--max_workers", type=int, default=50, help="Worker count for refinement jobs")
    parser.add_argument("--no_potential", dest="potential", action="store_false", help="Disable potential function refinement")
    parser.add_argument("--no_valid", dest="valid", action="store_false", help="Disable validation")
    return parser.parse_args()




def load_json(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def env_or_value(env_name: str, value: str | None) -> str | None:
    env_value = os.environ.get(env_name)
    if env_value is not None and env_value.strip():
        return env_value
    return value


def resolve_data_root(args) -> str:
    data_root = env_or_value("GPE_HAP_INPUT", args.data_root)
    if data_root:
        return data_root
    return os.path.join("your", "work", "dir", args.domain, "bpo")


def resolve_output_dir(args, domain: str) -> str:
    if args.output_dir:
        return args.output_dir
    output_root = env_or_value("GPE_HAP_OUTPUT_DIR", args.output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_root:
        return os.path.join(output_root, timestamp)
    return os.path.join(f"../{domain}/bpo", timestamp)


def load_task_jsons(data_root: str):
    required = {
        "recommend": os.path.join(data_root, "recommend_data.json"),
        "ask": os.path.join(data_root, "ask_data.json"),
        "search": os.path.join(data_root, "search_data.json"),
    }
    missing = [path for path in required.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing task data files: {missing}")
    return {name: load_json(path) for name, path in required.items()}


def build_client(config: dict, prefix: str) -> OpenAIClient:
    base_url = env_or_value(f"{prefix}_BASE_URL", config.get("base_url"))
    api_key = env_or_value(f"{prefix}_API_KEY", config.get("api_key"))
    model_path = env_or_value(f"{prefix}_MODEL_PATH", config.get("model_path"))
    return OpenAIClient(
        base_url=base_url,
        api_key=api_key,
        model_path=model_path,
    )


# def is_valid_format(text):
#     pattern = r"^(Ask\[[^\[\]]+\]|Recommend\[[^\[\]]+\]|Response\[[^\[\]]+\]|Search\[[^\[\]]+\])$"
#     return bool(text and re.match(pattern, text))



def is_valid_format(text, debug=False):
    if not text:
        if debug:
            print("❌ Empty text")
        return False

    pattern = r"^(Ask\[.*\]|Recommend\[.*\]|Response\[.*\]|Search\[.*\])$"
    match = re.match(pattern, text, re.DOTALL)
    
    if not match:
        if debug:
            print("❌ Regex did not match.")
            print(f"↪ TEXT START:\n{text[:100]}...")
            print(f"↪ TEXT END:\n{text[-100:]}")
            print(f"↪ LENGTH: {len(text)}")
            if text.count("[") != text.count("]"):
                print("❗ Brackets count mismatch:", text.count("["), "!=", text.count("]"))
            if not text.strip().endswith("]"):
                print("❗ Does not end with ]")
            if not text.strip().startswith(("Ask[", "Recommend[", "Response[", "Search[")):
                print("❗ Does not start with expected prefix")
        return False

    return True

def get_strategy(text):
    match = re.match(r"^(Ask|Recommend|Response|Search)\[.*\]$", text)
    return match.group(1) if match else None

def main():
    args = parse_args()
    mode = args.mode.lower()
    domain = args.domain
    sample_num = args.sample_num
    potential = args.potential
    valid = args.valid
    print(valid)
    if mode == "all":
        active_modes = {"recommend", "search", "ask"}
    else:
        active_modes = set([m.strip() for m in mode.split(",")])

    index_path = os.path.join(args.index_root, domain, "faiss_index.bin")
    metadata_path = os.path.join(args.index_root, domain, "metadata.json")
    data_root = resolve_data_root(args)

    emb_model = SentenceTransformer(args.model_path)
    index = faiss.read_index(index_path)

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    task_data = load_task_jsons(data_root)
    rec_data = task_data["recommend"]
    ask_data = task_data["ask"]
    search_data = task_data["search"]

    rec_tasks = [(i, d["Instruction"], d["output"], d["ground_truth"], d["is_correct"], d["is_recall"]) for i, d in enumerate(rec_data)]
    search_tasks = [(i, d["Instruction"], d["output"], d["gt_title"]) for i, d in enumerate(search_data)]
    ask_tasks = [(i, d["Instruction"], d["output"], d["ground_truth"]) for i, d in enumerate(ask_data)]

    output_dir = resolve_output_dir(args, domain)
    os.makedirs(output_dir, exist_ok=True)

    config = load_config(args.config_path)["openai"]
    openai_client = build_client(config, "OPENAI")

    mini_config = load_config(args.config_path)["openai_mini"]
    openai_mini_client = build_client(mini_config, "OPENAI_MINI")

    sft_results = []
    dpo_results = []
    all_logs = []

    if args.task_limit:
        rec_tasks = rec_tasks[:args.task_limit]
        ask_tasks = ask_tasks[:args.task_limit]
        search_tasks = search_tasks[:args.task_limit]

    if "recommend" in active_modes:
        rec_sft, rec_dpo, rec_logs = refine_rec_trajectories(rec_tasks, openai_mini_client, sample_num, potential, valid, max_workers=args.max_workers)
        sft_results.extend(rec_sft)
        dpo_results.extend(rec_dpo)
        all_logs.extend(rec_logs)

    if "ask" in active_modes:
        ask_sft, ask_dpo, ask_logs = refine_ask_trajectories(domain, ask_tasks, openai_mini_client, sample_num, potential, valid, max_workers=args.max_workers)
        sft_results.extend(ask_sft)
        dpo_results.extend(ask_dpo)
        all_logs.extend(ask_logs)

    if "search" in active_modes:
        search_sft, search_dpo, search_logs = refine_search_trajectories(
            search_tasks, openai_mini_client, index, metadata, emb_model, sample_num, potential, valid, max_workers=args.max_workers
        )
        sft_results.extend(search_sft)
        dpo_results.extend(search_dpo)
        all_logs.extend(search_logs)

    random.shuffle(sft_results)
    random.shuffle(dpo_results)

    with open(os.path.join(output_dir, f'{domain}_sto_sft_v1_sample{sample_num}.json'), 'w', encoding='utf-8') as f:
        json.dump(sft_results, f, ensure_ascii=False, indent=4)

    with open(os.path.join(output_dir, f'{domain}_sto_dpo_v1_sample{sample_num}.json'), 'w', encoding='utf-8') as f:
        json.dump(dpo_results, f, ensure_ascii=False, indent=4)

    with open(os.path.join(output_dir, f'{domain}_refine_log_sample{sample_num}.json'), 'w', encoding='utf-8') as f:
        json.dump(all_logs, f, ensure_ascii=False, indent=2)
    print(f"📝 Saved trace logs to: {os.path.join(output_dir, f'{domain}_refine_log_sample{sample_num}.json')}")

if __name__ == "__main__":
    main()
