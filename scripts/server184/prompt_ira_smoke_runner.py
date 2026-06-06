from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI
from user_simulator.persona.model.model import JsonResponseError
from user_simulator.user_agent_env_v1 import UserAgentEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default='Book')
    parser.add_argument('--persona-path', default=None)
    parser.add_argument('--user-id', type=int, default=0)
    parser.add_argument('--item-id', type=int, default=0)
    parser.add_argument('--config-path', required=True)
    parser.add_argument('--format-path', required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--api-key', default='EMPTY')
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--memory-mode', default='critiquescope')
    return parser.parse_args()


def write_run_metadata(output_dir: Path, payload: dict) -> None:
    (output_dir / 'run_metadata.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bpo_dir = output_dir / 'bpo'
    bpo_dir.mkdir(parents=True, exist_ok=True)

    persona_path = args.persona_path or f'user_simulator/task/{args.domain}_test.jsonl'
    request = {
        'domain': args.domain,
        'persona_path': persona_path,
        'user_id': args.user_id,
        'item_id': args.item_id,
        'base_url': args.base_url,
        'model_name': args.model_name,
        'memory_mode': args.memory_mode,
    }
    (output_dir / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    try:
        env = UserAgentEnv(
            persona_path=persona_path,
            user_id=args.user_id,
            item_id=args.item_id,
            config_path=args.config_path,
            format_path=args.format_path,
            domain=args.domain,
            model_type='vllm',
            memory_mode=args.memory_mode,
        )
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)

        first_turn = env.step(None)
        first_user = json.loads(first_turn['user_response'])

        ask_prompt = (
            f"You are a conversational recommendation assistant for {args.domain}. "
            "The user has just opened a conversation. Ask exactly one concise clarification question "
            "that helps identify a suitable item. Return exactly one action in the form Ask[...]."
            f"\nUser message: {first_user['response']}"
        )
        ask_resp = client.chat.completions.create(
            model=args.model_name,
            messages=[
                {'role': 'system', 'content': 'You are a precise conversational recommender.'},
                {'role': 'user', 'content': ask_prompt},
            ],
            temperature=0,
        ).choices[0].message.content.strip()

        second_turn = env.step(ask_resp)
        second_user = json.loads(second_turn['user_response'])
        history_before_recommend = str(env.get_dialogue_history())

        recommend_prompt = (
            f"You are a conversational recommendation assistant for {args.domain}. "
            "Based on the dialogue history below, provide exactly one recommendation action in the form "
            "Recommend[<item title> | <brief reason>]. Do not output anything else."
            f"\nDialogue history:\n{history_before_recommend}"
        )
        recommend_resp = client.chat.completions.create(
            model=args.model_name,
            messages=[
                {'role': 'system', 'content': 'You are a precise conversational recommender.'},
                {'role': 'user', 'content': recommend_prompt},
            ],
            temperature=0,
        ).choices[0].message.content.strip()

        third_turn = env.step(recommend_resp)
        feedback_user = json.loads(third_turn['user_response'])
        target_item = env.item

        ask_record = [{
            'Instruction': f"User opening request:\n{first_user['response']}",
            'output': ask_resp,
            'ground_truth': target_item,
        }]
        recommend_record = [{
            'Instruction': history_before_recommend,
            'output': recommend_resp,
            'ground_truth': target_item,
            'is_correct': target_item.get('ItemName', '').lower() in recommend_resp.lower(),
            'is_recall': target_item.get('ItemName', '').lower() in recommend_resp.lower(),
            'user_feedback': feedback_user['response'],
            'recommendation_satisfaction': third_turn['recommendation_satisfaction'],
            'action_satisfaction': third_turn['action_satisfaction'],
            'expression_satisfaction': third_turn['expression_satisfaction'],
        }]

        (bpo_dir / 'ask_data.json').write_text(json.dumps(ask_record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        (bpo_dir / 'recommend_data.json').write_text(json.dumps(recommend_record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

        response = {
            'first_user': first_user,
            'ask_prompt': ask_prompt,
            'ask_response': ask_resp,
            'second_user': second_user,
            'recommend_prompt': recommend_prompt,
            'recommend_response': recommend_resp,
            'feedback_user': feedback_user,
            'feedback_metrics': {
                'recommendation_satisfaction': third_turn['recommendation_satisfaction'],
                'action_satisfaction': third_turn['action_satisfaction'],
                'expression_satisfaction': third_turn['expression_satisfaction'],
            },
            'history_after_feedback': str(env.get_dialogue_history()),
        }
        (output_dir / 'response.json').write_text(json.dumps(response, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        write_run_metadata(output_dir, {
            'status': 'COMPLETED_REAL_PROMPT_IRA_SMOKE',
            'timestamp': datetime.now().isoformat(),
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'endpoint': args.base_url,
            'model_alias': args.model_name,
            'domain': args.domain,
            'ask_data_path': str(bpo_dir / 'ask_data.json'),
            'recommend_data_path': str(bpo_dir / 'recommend_data.json'),
        })
        print(json.dumps({'status': 'COMPLETED_REAL_PROMPT_IRA_SMOKE', 'output_dir': str(output_dir)}, ensure_ascii=False, indent=2))
    except JsonResponseError as exc:
        blocked_status = 'BLOCKED_JSON_SCHEMA_MISMATCH' if exc.status == 'SCHEMA_MISMATCH' else 'BLOCKED_INCOMPLETE_MODEL_JSON'
        failure = {
            'status': blocked_status,
            'json_status': exc.status,
            'error': str(exc),
            'timestamp': datetime.now().isoformat(),
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'endpoint': args.base_url,
            'model_alias': args.model_name,
            'domain': args.domain,
        }
        write_run_metadata(output_dir, failure)
        (output_dir / 'response.json').write_text(json.dumps({'error': str(exc), 'json_status': exc.status}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        raise
    except Exception as exc:
        failure = {
            'status': 'BLOCKED_VLLM',
            'error': str(exc),
            'exception_type': type(exc).__name__,
            'timestamp': datetime.now().isoformat(),
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'endpoint': args.base_url,
            'model_alias': args.model_name,
            'domain': args.domain,
            'traceback': traceback.format_exc(),
        }
        write_run_metadata(output_dir, failure)
        (output_dir / 'response.json').write_text(json.dumps(failure, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        raise


if __name__ == '__main__':
    main()
