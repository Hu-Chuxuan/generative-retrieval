"""
GPT-4o agent for WebShop.

Usage:
    python run_gpt4o.py --num_episodes 5 --output_dir trajs_gpt4o

Requires:
    pip install openai
    OPENAI_API_KEY environment variable set
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
from web_agent_site.utils import DEBUG_PROD_SIZE

client = OpenAI()

SYSTEM_PROMPT = """You are a shopping agent navigating an e-commerce website.
Your goal is to find and purchase a product that matches the given instruction.

At each step you will see the current page content and available actions.
You must respond with exactly one action in one of these two formats:
  search[query]     — submit a search query (only when on the search page)
  click[button]     — click a button or product link

Rules:
- On the search page, use search[...] to find products.
- On results/product pages, use click[...] to navigate or select options.
- When you find the right product and have selected the correct options, click[buy now].
- Do not output anything except the action."""


def build_user_message(obs: str, available_actions: dict) -> str:
    if available_actions["has_search_bar"]:
        actions_str = "Available: search[your query]"
    else:
        clickables = available_actions["clickables"]
        actions_str = "Available clicks: " + ", ".join(f'click[{c}]' for c in clickables)
    return f"{obs}\n\n{actions_str}"


def get_action(obs: str, available_actions: dict, history: list) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": build_user_message(obs, available_actions)})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.0,
        max_tokens=64,
    )
    action = response.choices[0].message.content.strip()
    return action, messages[-1]["content"]


def run_episode(env, episode_idx: int, max_steps: int = 15) -> dict:
    obs, _ = env.reset(session=episode_idx)
    goal = env.server.goals[episode_idx]

    history = []
    steps = []
    total_reward = 0.0

    for step in range(max_steps):
        available_actions = env.get_available_actions()
        action, user_msg = get_action(obs, available_actions, history)

        # append to history for multi-turn context
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": action})

        steps.append({"step": step, "obs": obs, "action": action})

        obs, reward, done, info = env.step(action)
        total_reward = reward

        if done:
            break

    return {
        "episode": episode_idx,
        "goal": goal,
        "steps": steps,
        "reward": total_reward,
        "num_steps": len(steps),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="trajs_gpt4o")
    parser.add_argument("--max_steps", type=int, default=15)
    args = parser.parse_args()

    out_dir = Path(__file__).parent / args.output_dir
    out_dir.mkdir(exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"{run_id}.jsonl"

    env = WebAgentTextEnv(observation_mode="text", num_products=DEBUG_PROD_SIZE)

    rewards = []
    with open(out_file, "w") as f:
        for i in range(args.start_idx, args.start_idx + args.num_episodes):
            print(f"Episode {i} ...", end=" ", flush=True)
            result = run_episode(env, i, max_steps=args.max_steps)
            rewards.append(result["reward"])
            f.write(json.dumps(result) + "\n")
            f.flush()
            print(f"reward={result['reward']:.3f}  steps={result['num_steps']}")

    env.close()
    avg = sum(rewards) / len(rewards) if rewards else 0
    print(f"\nDone. Avg reward: {avg:.3f}  Trajectories saved to: {out_file}")


if __name__ == "__main__":
    main()
