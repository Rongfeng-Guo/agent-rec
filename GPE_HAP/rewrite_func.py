import json
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
import sys
from faiss_rank_utils import get_gt_rank
import os
from collections import Counter
# 设置 CUDA 设备为 GPU 0（可以是 0, 1, 2,...）
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 只使用 GPU 0
from sentence_transformers import SentenceTransformer
import faiss
import random
from refine_prompts_v2 import ask_potential_function_template, recommendation_potential_function_template, search_potential_function_template, ask_policy_improvement_template, recommendation_policy_improvement_template, search_policy_improvement_template, ask_potential_eval_template, recommendation_potential_eval_template, ask_policy_improvement_abpotential_template, recommendation_policy_improvement_abpotential_template, search_policy_improvement_abpotential_template
# 获取当前工作目录
current_dir = os.getcwd()

# 获取父级目录路径
parent_dir = os.path.abspath(os.path.join(current_dir, '../../'))

# 将父级目录添加到 sys.path
sys.path.insert(0, parent_dir)
from collections import defaultdict
from datetime import datetime  # Import datetime for timestamp functionality
from model.model import OpenAIClient
import re

def parse_action(string):
    # 支持两种格式：
    # 1. "Action 2: Search[circus setting mysterious books with quirky characters and supernatural elements]"
    # 2. "Search[circus setting mysterious books with quirky characters and supernatural elements]"
    pattern = r'^(?:Action\s+\d+:\s+)?(\w+)\[(.+)\]$'
    
    match = re.match(pattern, string)
    if match:
        action_type = match.group(1).lower()  # 提取动作类型并转换为小写
        argument = match.group(2)  # 提取参数，保留原始大小写
        return action_type, argument
    else:
        return None

# 从 JSONL 文件加载数据（逐行加载）
def load_jsonl(file_path: str, num_lines=None):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if num_lines is not None and i >= num_lines:
                break
            data.append(json.loads(line))
    return data

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def parse_json(response):
    if response.startswith("```json"):
        response = response[len("```json"):].strip()
    if response.endswith("```"):
        response = response[:-len("```")].strip()
    return json.loads(response)


def refine_single_rec_trajectory(openai_client, scratchpad_input, original_response, ground_truth, sample_num, potential=True, valid=True):
    def _build_recommendation_potential_function_prompt():
        prompt = recommendation_potential_function_template.format(
            Scratchpad=scratchpad_input,
            Original_response=original_response,
            Ground_truth=ground_truth,
            Sample_num=sample_num
        )
        return prompt
    
    def _build_recommendation_policy_improvement_abpotential_prompt(generative_reward):
        prompt = recommendation_policy_improvement_abpotential_template.format(
            Scratchpad=scratchpad_input,
            Original_response=original_response,
            Sample_num=sample_num
        )
        return prompt
    
    def _build_recommendation_policy_improvement_prompt(generative_reward):
        prompt = recommendation_policy_improvement_template.format(
            Scratchpad=scratchpad_input,
            Original_response=original_response,
            Generative_reward=generative_reward,
            Sample_num=sample_num
        )
        return prompt
    
    def _build_recommendation_potential_eval_prompt(refinements):
        """
        构建推荐潜在函数评估的 prompt。
        """
        prompt = recommendation_potential_eval_template.format(
            Scratchpad=scratchpad_input,
            Original_response=original_response,
            Refinement_output=refinements,
            Ground_truth=ground_truth,
            Sample_num=sample_num
        )
        return prompt
    """
    使用 openai_client 对单个对话进行refine。

    Parameters:
    - openai_client: The OpenAI API client.
    - original_text (str): 原始对话字符串
    - refine_prompt (str): refine的prompt模板

    Returns:
    - str: refine 后的文本
    """
    log = {
        "task_type": "recommend",
        "input": scratchpad_input,
        "original_response": original_response,
        "ground_truth": ground_truth
    }
    if potential:
        try:
            potential_prompt = _build_recommendation_potential_function_prompt()
            potential_reward = openai_client.get_single_chat_completion(potential_prompt).strip()
            log["potential_reward_prompt"] = potential_prompt
            log["potential_reward_output"] = potential_reward
        except Exception as e:
            log["potential_reward_error"] = str(e)
            return None, True, log
        
        try:
            policy_prompt = _build_recommendation_policy_improvement_prompt(potential_reward)
            improved_policy = openai_client.get_single_chat_completion(policy_prompt).strip()
            log["policy_improvement_prompt"] = policy_prompt
            log["policy_improvement_output"] = improved_policy
        except Exception as e:
            log["policy_improvement_error"] = str(e)
            return None, True, log
    else:
        try:
            policy_prompt = _build_recommendation_policy_improvement_abpotential_prompt(original_response)
            improved_policy = openai_client.get_single_chat_completion(policy_prompt).strip()
            log["policy_improvement_prompt"] = policy_prompt
            log["policy_improvement_output"] = improved_policy
        except Exception as e:
            log["policy_improvement_error"] = str(e)
            return None, True, log

    if valid:
        try:
            eval_prompt = _build_recommendation_potential_eval_prompt(improved_policy)
            eval_response = openai_client.get_single_chat_completion(eval_prompt).strip()
            parsed_eval = parse_json(eval_response)

            best_refinement = parsed_eval.get("refinement_output")
            is_original_best = not parsed_eval.get("is_better", False)

            log["eval_prompt"] = eval_prompt
            log["eval_response"] = eval_response
            log["best_refinement"] = best_refinement
            log["is_original_best"] = is_original_best

            return best_refinement, is_original_best, log
        except Exception as e:
            log["evaluation_error"] = str(e)
            return None, True, log
    else:
        # 如果不需要验证，直接返回改进后的策略
        return parse_json(improved_policy).get("refinement_output", None)[0], True, log



def refine_single_search_trajectory(openai_client, input, original_response, ground_truth, sample_num, potential=True):
    def _build_search_potential_function_prompt():
        return search_potential_function_template.format(
            Scratchpad=input,
            Original_response=original_response,
            Ground_truth=ground_truth,
            Sample_num=sample_num
        )
    
    def _build_search_policy_improvement_abpotential_prompt():
        return search_policy_improvement_abpotential_template.format(
            Scratchpad=input,
            Original_response=original_response,
            Sample_num=sample_num
        )

    def _build_search_policy_improvement_prompt(generative_reward):
        return search_policy_improvement_template.format(
            Scratchpad=input,
            Original_response=original_response,
            Generative_reward=generative_reward,
            Sample_num=sample_num
        )

    log = {
        "task_type": "search",
        "input": input,
        "original_response": original_response,
        "ground_truth": ground_truth
    }

    if potential:
        try:
            potential_prompt = _build_search_potential_function_prompt()
            potential_reward = openai_client.get_single_chat_completion(potential_prompt).strip()
            log["potential_reward_prompt"] = potential_prompt
            log["potential_reward_output"] = potential_reward
        except Exception as e:
            log["potential_reward_error"] = str(e)
            return None, log

        try:
            policy_prompt = _build_search_policy_improvement_prompt(potential_reward)
            improved_policy = openai_client.get_single_chat_completion(policy_prompt).strip()
            log["policy_improvement_prompt"] = policy_prompt
            log["policy_improvement_output"] = improved_policy
            return improved_policy, log
        except Exception as e:
            log["policy_improvement_error"] = str(e)
            return None, log
    else:
        try:
            policy_prompt = _build_search_policy_improvement_abpotential_prompt()
            improved_policy = openai_client.get_single_chat_completion(policy_prompt).strip()
            log["policy_improvement_prompt"] = policy_prompt
            log["policy_improvement_output"] = improved_policy
            return improved_policy, log
        except Exception as e:
            log["policy_improvement_error"] = str(e)
            return None, log


def refine_single_ask_trajectory(domain, openai_client, input, original_response, ground_truth, sample_num, potential=True, valid=True):
    def _build_ask_potential_function_prompt():
        return ask_potential_function_template.format(
            Domain=domain,
            Scratchpad=input,
            Original_response=original_response,
            Ground_truth=ground_truth,
            Sample_num=sample_num
        )
    
    def _build_ask_policy_improvement_abpotential_prompt():
        return ask_policy_improvement_abpotential_template.format(
            Domain=domain,
            Scratchpad=input,
            Original_response=original_response,
            Sample_num=sample_num
        )

    def _build_ask_policy_improvement_prompt(generative_reward):
        return ask_policy_improvement_template.format(
            Domain=domain,
            Scratchpad=input,
            Original_response=original_response,
            Generative_reward=generative_reward,
            Sample_num=sample_num
        )

    def _build_ask_potential_eval_prompt(refinements):
        return ask_potential_eval_template.format(
            Domain=domain,
            Scratchpad=input,
            Original_response=original_response,
            Refinement_output=refinements,
            Ground_truth=ground_truth,
            Sample_num=sample_num
        )

    log = {
        "task_type": "ask",
        "input": input,
        "original_response": original_response,
        "ground_truth": ground_truth
    }
    #raise NotImplementedError("to do")
    if potential:
        try:
            potential_prompt = _build_ask_potential_function_prompt()
            potential_reward = openai_client.get_single_chat_completion(potential_prompt).strip()
            log["potential_reward_prompt"] = potential_prompt
            log["potential_reward_output"] = potential_reward
        except Exception as e:
            log["potential_reward_error"] = str(e)
            return None, True, log

        try:
            policy_prompt = _build_ask_policy_improvement_prompt(potential_reward)
            improved_policy = openai_client.get_single_chat_completion(policy_prompt).strip()
            log["policy_improvement_prompt"] = policy_prompt
            log["policy_improvement_output"] = improved_policy
        except Exception as e:
            log["policy_improvement_error"] = str(e)
            return None, True, log
    else:
        try:
            policy_prompt = _build_ask_policy_improvement_abpotential_prompt()
            improved_policy = openai_client.get_single_chat_completion(policy_prompt).strip()
            log["policy_improvement_prompt"] = policy_prompt
            log["policy_improvement_output"] = improved_policy
        except Exception as e:
            log["policy_improvement_error"] = str(e)
            return None, True, log
    if valid:
        try:
            eval_prompt = _build_ask_potential_eval_prompt(improved_policy)
            eval_response = openai_client.get_single_chat_completion(eval_prompt).strip()
            parsed_eval = parse_json(eval_response)

            best_refinement = parsed_eval.get("refinement_output")
            is_original_best = not parsed_eval.get("is_better", False)

            log["eval_prompt"] = eval_prompt
            log["eval_response"] = eval_response
            log["best_refinement"] = best_refinement
            log["is_original_best"] = is_original_best

            return best_refinement, is_original_best, log
        except Exception as e:
            log["evaluation_error"] = str(e)
            return None, True, log
    else:
        # 如果不需要验证，直接返回改进后的策略
        return parse_json(improved_policy).get("refinement_output", None)[0], True, log


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
    """提取策略部分（Ask、Recommend、Response、Search）"""
    if not is_valid_format(text):
        return None
    # 提取策略的名字
    match = re.match(r"^(Ask|Recommend|Response|Search)\[.*\]$", text)
    return match.group(1) if match else None


def refine_rec_trajectories(tasks, openai_client, sample_num, potential=True, valid=True, max_workers=5):
    """
    并发对多个对话进行 refine 处理，并统计失败情况。
    返回三个值：sft结果、dpo结果、日志trace列表
    """
    sft_results = []
    dpo_results = []
    logs = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_user = {
            executor.submit(refine_single_rec_trajectory, openai_client, t[1], t[2], t[3], sample_num, potential, valid): t
            for t in tasks
        }

        for future in tqdm(as_completed(future_to_user), total=len(future_to_user), desc="Refining Dialogues"):
            task = future_to_user[future]
            input = task[1]
            original_output = task[2]

            try:
                best_refinement, original_better, trace = future.result()
                logs.append(trace)

                if best_refinement and is_valid_format(best_refinement) and not original_better:
                    sft_results.append({
                        "system": "You are a helpful assistant",
                        "instruction": input,
                        "input": "",
                        "output": best_refinement
                    })
                    dpo_results.append({
                        "system": "You are a helpful assistant",
                        "instruction": input,
                        "input": "",
                        "chosen": best_refinement,
                        "rejected": original_output
                    })
            except Exception as e:
                logs.append({"task": task, "error": str(e)})
                continue

    return sft_results, dpo_results, logs


def validate_ask(openai_client, scratchpad, original, refinements, ground_truth, sample_num):
    """
    验证 Ask 改写的有效性，返回最佳 Ask[...] 响应，以及是否 original 最佳。

    返回：
        (best_response: str, is_original_best: bool)
    """
    def _eval_ask_refinement():
        """
        使用 OpenAI API 评估 Ask 改写的质量。
        """
        prompt = ask_potential_eval_template.format(
            Scratchpad=scratchpad,
            Original_response=original,
            Refinement_output=refinements,
            Ground_truth=ground_truth,
            Sample_num=sample_num
        )
        return prompt
    
    eval_prompt = _eval_ask_refinement()
    try:
        response = openai_client.get_single_chat_completion(eval_prompt).strip()
    except Exception as e:
        print(f"Ask Evaluation Error: {e}")
        return None, False
    
    parsed_evaluation = parse_json(response)

    return parsed_evaluation.get("refinement_output"), not parsed_evaluation.get("is_better", False)



def validate_recommendation(openai_client, scratchpad, original, refinements, ground_truth, sample_num):
    """
    验证 Ask 改写的有效性，返回最佳 Ask[...] 响应，以及是否 original 最佳。

    返回：
        (best_response: str, is_original_best: bool)
    """
    def _eval_rec_refinement():
        """
        使用 OpenAI API 评估 Ask 改写的质量。
        """
        prompt = recommendation_potential_eval_template.format(
            Scratchpad=scratchpad,
            Original_response=original,
            Refinement_output=refinements,
            Ground_truth=ground_truth,
            Sample_num=sample_num
        )
        return prompt
    
    eval_prompt = _eval_rec_refinement()
    try:
        response = openai_client.get_single_chat_completion(eval_prompt).strip()
    except Exception as e:
        print(f"Ask Evaluation Error: {e}")
        return None, False
    
    parsed_evaluation = parse_json(response)

    return parsed_evaluation.get("refinement_output"), not parsed_evaluation.get("is_better", False)



def validate_query(original, refinements, ground_truth, index, metadata, model, threshold=1):
    def extract_query(action_str):
        return parse_action(action_str)[1] if action_str else ""

    log = {
        "task_type": "search",
        "original_action": original,
        "ground_truth": ground_truth,
        "refinements": refinements,
    }

    original_q = extract_query(original)
    if not original_q:
        log["error"] = "Original query extraction failed"
        return None, True, log

    refined_queries = []
    for refinement in refinements:
        if isinstance(refinement, str):
            ref_q = extract_query(refinement)
            refined_queries.append(ref_q)
        else:
            log["error"] = f"Invalid refinement format: {refinement}"
            return None, True, log

    original_rank, ori_sim = get_gt_rank(original_q, ground_truth, model, index, metadata)
    log["original_query"] = original_q
    log["original_rank"] = original_rank

    log["refined_queries"] = refined_queries
    log["refined_ranks"] = []

    best_rank = original_rank
    best_query = original_q
    best_response = original

    for i, ref_q in enumerate(refined_queries):
        rank, sim = get_gt_rank(ref_q, ground_truth, model, index, metadata)
        log["refined_ranks"].append({"query": ref_q, "rank": rank})
        if rank < best_rank:
            best_rank = rank
            best_query = ref_q
            best_response = refinements[i]

    if best_response == original:
        log["is_original_best"] = True
        log["best_refinement"] = best_response
        return best_response, True, log
    else:
        if original_rank - best_rank < threshold:
            log["is_original_best"] = True
            log["best_refinement"] = original
            return original, True, log
        else:
            log["is_original_best"] = False
            log["best_refinement"] = best_response
            return best_response, False, log



def refine_search_trajectories(tasks, openai_client, index, meta_data, emb_model, sample_num, potential=True, valid=True, max_workers=5):
    sft_results = []
    dpo_results = []
    logs = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_user = {
            executor.submit(refine_single_search_trajectory, openai_client, t[1], t[2], t[3], sample_num, potential): t
            for t in tasks
        }

        for future in tqdm(as_completed(future_to_user), total=len(future_to_user), desc="Refining Dialogues"):
            task = future_to_user[future]
            input = task[1]
            original_output = task[2]
            ground_truth = task[3]

            try:
                res, gen_log = future.result()
                if res is None:
                    continue

                refinements = parse_json(res)["refinement_output"]
                if valid:
                    best_refinement, original_better, val_log = validate_query(
                        original_output, refinements, ground_truth, index, meta_data, emb_model, threshold=1
                    )

                    combined_log = {
                        **gen_log,
                        **val_log,
                        "input": input,
                        "original_response": original_output
                    }
                    logs.append(combined_log)
                else:
                    best_refinement = refinements[0] if refinements else None
                    original_better = False
                    logs.append({
                        "task": task,
                        "input": input,
                        "original_response": original_output,
                        "best_refinement": best_refinement,
                        "is_original_best": original_better
                    })
                if not original_better and is_valid_format(best_refinement):
                    sft_results.append({
                        "system": "You are a helpful assistant",
                        "instruction": input,
                        "input": "",
                        "output": best_refinement
                    })
                    dpo_results.append({
                        "system": "You are a helpful assistant",
                        "instruction": input,
                        "input": "",
                        "chosen": best_refinement,
                        "rejected": original_output
                    })
            except Exception as e:
                logs.append({"task": task, "error": str(e)})
                continue

    return sft_results, dpo_results, logs


def refine_ask_trajectories(domain, tasks, openai_client, sample_num, potential=True, valid=True, max_workers=5):
    sft_results = []
    dpo_results = []
    logs = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_user = {
            executor.submit(refine_single_ask_trajectory, domain, openai_client, t[1], t[2], t[3], sample_num, potential, valid): t
            for t in tasks
        }

        for future in tqdm(as_completed(future_to_user), total=len(future_to_user), desc="Refining Dialogues"):
            task = future_to_user[future]
            input = task[1]
            original_output = task[2]

            try:
                best_refinement, original_better, trace = future.result()
                logs.append(trace)

                if best_refinement and is_valid_format(best_refinement) and not original_better:
                    sft_results.append({
                        "system": "You are a helpful assistant",
                        "instruction": input,
                        "input": "",
                        "output": best_refinement
                    })
                    dpo_results.append({
                        "system": "You are a helpful assistant",
                        "instruction": input,
                        "input": "",
                        "chosen": best_refinement,
                        "rejected": original_output
                    })
            except Exception as e:
                logs.append({"task": task, "error": str(e)})
                continue

    return sft_results, dpo_results, logs


def load_json(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


