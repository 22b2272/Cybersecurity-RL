"""Train and evaluate DDQN on the action-dependent synthetic IDS simulator."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from cybersecurity_ml.ids.ddqn import ACTION_NAMES, DDQNAgent
from cybersecurity_ml.ids.lab_simulator import ActionDependentThreatEnv, FEATURE_NAMES


def evaluate(agent: DDQNAgent | None, episodes: int, seed: int, fixed_action: int | None = None) -> dict[str, float]:
    env = ActionDependentThreatEnv(seed=seed)
    returns = []
    threat_detection = 0
    threat_count = 0
    false_blocks = 0
    benign_count = 0
    for _ in range(episodes):
        state = env.reset()
        total = 0.0
        done = False
        while not done:
            action = fixed_action if agent is None else agent.select_action(state, explore=False)
            state, reward, done, info = env.step(action)
            total += reward
            if info["label"]:
                threat_count += 1
                threat_detection += action != 0
            else:
                benign_count += 1
                false_blocks += action == 2
        returns.append(total)
    return {
        "episodes": float(episodes),
        "mean_episode_reward": float(np.mean(returns)),
        "threat_detection_rate": threat_detection / max(threat_count, 1),
        "false_blocks_per_benign_step": false_blocks / max(benign_count, 1),
    }


def train(episodes: int = 300, seed: int = 42, episode_steps: int = 32) -> tuple[DDQNAgent, dict]:
    env = ActionDependentThreatEnv(episode_steps=episode_steps, seed=seed)
    agent = DDQNAgent(
        state_size=len(FEATURE_NAMES), batch_size=64, memory_size=25_000,
        target_update_steps=200, seed=seed,
    )
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, explore=True)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
    eval_episodes = max(50, min(episodes, 500))
    metrics = {
        "ddqn": evaluate(agent, episodes=eval_episodes, seed=seed + 1),
        "always_allow": evaluate(None, episodes=eval_episodes, seed=seed + 1, fixed_action=0),
        "always_alert": evaluate(None, episodes=eval_episodes, seed=seed + 1, fixed_action=1),
        "always_block": evaluate(None, episodes=eval_episodes, seed=seed + 1, fixed_action=2),
    }
    return agent, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--episode-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/ids_simulator_ddqn.joblib"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    agent, metrics = train(args.episodes, args.seed, args.episode_steps)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "agent": agent,
        "feature_names": FEATURE_NAMES,
        "action_names": ACTION_NAMES,
        "training": {"episodes": args.episodes, "seed": args.seed, "simulator": "synthetic-v1"},
        "evaluation": metrics,
    }, args.model_out)
    print(f"Synthetic simulator evaluation: {metrics}")
    print(f"Saved simulator policy to {args.model_out}")


if __name__ == "__main__":
    main()
