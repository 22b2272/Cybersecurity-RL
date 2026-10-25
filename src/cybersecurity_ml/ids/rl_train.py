"""Train and evaluate a DDQN IDS response policy on NSL-KDD replay data."""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from cybersecurity_ml.ids.data import build_preprocessor, load_kdd, split_features_and_labels
from cybersecurity_ml.ids.ddqn import ACTION_NAMES, DDQNAgent
from cybersecurity_ml.ids.environment import DEFAULT_REWARDS, IDSResponseEnv


def _dense_float32(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def train_policy(
    train_path: str | Path,
    steps: int = 10_000,
    batch_size: int = 64,
    memory_size: int = 25_000,
    episode_steps: int = 512,
    seed: int = 42,
) -> tuple[DDQNAgent, object]:
    """Fit the train-only preprocessor and learn response actions by DDQN replay."""
    if steps < 1:
        raise ValueError("steps must be positive")
    data = load_kdd(train_path)
    features, labels = split_features_and_labels(data)
    preprocessor = build_preprocessor()
    states = _dense_float32(preprocessor.fit_transform(features))
    environment = IDSResponseEnv(states, labels.to_numpy(), episode_steps=episode_steps, seed=seed)
    agent = DDQNAgent(
        state_size=states.shape[1],
        batch_size=batch_size,
        memory_size=memory_size,
        seed=seed,
    )
    state = environment.reset()
    for step in range(1, steps + 1):
        action = agent.select_action(state, explore=True)
        next_state, reward, done, _ = environment.step(action)
        agent.remember(state, action, reward, next_state, done)
        loss = agent.learn()
        if done:
            state = environment.reset()
        else:
            state = next_state
        if step % 1000 == 0:
            loss_text = "warming up" if loss is None else f"loss={loss:.4f}"
            print(f"Step {step}/{steps}: epsilon={agent.epsilon:.3f}, {loss_text}")
    return agent, preprocessor


def evaluate_policy(agent: DDQNAgent, preprocessor, test_path: str | Path) -> dict[str, object]:
    """Evaluate intrusion detection and response choices on held-out labeled rows."""
    data = load_kdd(test_path)
    features, labels = split_features_and_labels(data)
    states = _dense_float32(preprocessor.transform(features))
    actions = agent.predict_actions(states)
    detections = (actions != 0).astype(np.int8)
    y_true = labels.to_numpy()
    print("Threat detection metrics (alert or block = detected):")
    print(classification_report(y_true, detections, target_names=["normal", "attack"], zero_division=0))
    print("Confusion matrix [normal, attack]:")
    matrix = confusion_matrix(y_true, detections, labels=[0, 1])
    print(matrix)
    action_counts = {name: int(np.sum(actions == index)) for index, name in enumerate(ACTION_NAMES)}
    action_by_class = {
        label_name: {
            action_name: int(np.sum((y_true == label) & (actions == action_index)))
            for action_index, action_name in enumerate(ACTION_NAMES)
        }
        for label, label_name in ((0, "normal"), (1, "attack"))
    }
    print(f"Response action counts: {action_counts}")
    print(f"Response actions by true class: {action_by_class}")
    chosen_rewards = DEFAULT_REWARDS[y_true, actions]
    mean_reward = float(chosen_rewards.mean())
    print(f"Mean response reward: {mean_reward:.4f}")
    return {
        "confusion_matrix": matrix,
        "action_counts": action_counts,
        "action_by_class": action_by_class,
        "mean_reward": mean_reward,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Path to KDDTrain+.txt")
    parser.add_argument("--test", type=Path, required=True, help="Path to KDDTest+.txt")
    parser.add_argument("--steps", type=int, default=10_000, help="DDQN environment steps")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--memory-size", type=int, default=25_000)
    parser.add_argument("--episode-steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/ids_ddqn.joblib"))
    args = parser.parse_args()
    agent, preprocessor = train_policy(
        args.train, args.steps, args.batch_size, args.memory_size, args.episode_steps, args.seed
    )
    metrics = evaluate_policy(agent, preprocessor, args.test)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"agent": agent, "preprocessor": preprocessor, "action_names": ACTION_NAMES,
         "seed": args.seed, "metrics": metrics},
        args.model_out,
    )
    print(f"Saved DDQN policy to {args.model_out}")


if __name__ == "__main__":
    main()
