"""A small NumPy Double DQN agent for IDS response actions."""
from collections import deque

import numpy as np

ACTION_NAMES = ("allow", "alert", "block")


class QNetwork:
    """Two-layer MLP with Huber loss, Adam updates, and clipped gradients."""

    def __init__(self, input_size: int, action_size: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.parameters = {
            "w1": rng.normal(0, np.sqrt(2 / input_size), (input_size, 128)).astype(np.float32),
            "b1": np.zeros(128, dtype=np.float32),
            "w2": rng.normal(0, np.sqrt(1 / 128), (128, action_size)).astype(np.float32),
            "b2": np.zeros(action_size, dtype=np.float32),
        }
        self.moments = {key: np.zeros_like(value) for key, value in self.parameters.items()}
        self.variances = {key: np.zeros_like(value) for key, value in self.parameters.items()}
        self.optimizer_step = 0

    def predict(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        hidden = np.maximum(states @ self.parameters["w1"] + self.parameters["b1"], 0)
        return hidden @ self.parameters["w2"] + self.parameters["b2"]

    def train(self, states: np.ndarray, targets: np.ndarray, learning_rate: float = 1e-3) -> float:
        states = np.asarray(states, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        z1 = states @ self.parameters["w1"] + self.parameters["b1"]
        hidden = np.maximum(z1, 0)
        predictions = hidden @ self.parameters["w2"] + self.parameters["b2"]
        error = predictions - targets
        absolute = np.abs(error)
        loss = np.where(absolute <= 1, 0.5 * error**2, absolute - 0.5).mean()
        batch_size = max(len(states), 1)
        gradient_output = np.clip(error, -1, 1) / batch_size
        gradient_w2 = hidden.T @ gradient_output
        gradient_b2 = gradient_output.sum(axis=0)
        gradient_hidden = gradient_output @ self.parameters["w2"].T
        gradient_z1 = gradient_hidden * (z1 > 0)
        gradients = {
            "w1": states.T @ gradient_z1,
            "b1": gradient_z1.sum(axis=0),
            "w2": gradient_w2,
            "b2": gradient_b2,
        }
        norm = np.sqrt(sum(np.square(gradient).sum() for gradient in gradients.values()))
        if norm > 10:
            gradients = {key: value * (10 / norm) for key, value in gradients.items()}
        self.optimizer_step += 1
        beta1, beta2 = 0.9, 0.999
        for key, gradient in gradients.items():
            self.moments[key] = beta1 * self.moments[key] + (1 - beta1) * gradient
            self.variances[key] = beta2 * self.variances[key] + (1 - beta2) * gradient**2
            moment_hat = self.moments[key] / (1 - beta1**self.optimizer_step)
            variance_hat = self.variances[key] / (1 - beta2**self.optimizer_step)
            self.parameters[key] -= learning_rate * moment_hat / (np.sqrt(variance_hat) + 1e-8)
        return float(loss)

    def copy_from(self, other: "QNetwork") -> None:
        for key in self.parameters:
            self.parameters[key] = other.parameters[key].copy()


class DDQNAgent:
    """Double DQN with replay memory and a periodically copied target network."""

    def __init__(
        self,
        state_size: int,
        action_size: int = len(ACTION_NAMES),
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        memory_size: int = 25_000,
        target_update_steps: int = 200,
        seed: int = 42,
    ) -> None:
        if action_size != len(ACTION_NAMES):
            raise ValueError(f"IDS response agent requires {len(ACTION_NAMES)} actions")
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_update_steps = target_update_steps
        self.rng = np.random.default_rng(seed)
        self.memory: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=memory_size)
        self.online = QNetwork(state_size, action_size, seed)
        self.target = QNetwork(state_size, action_size, seed + 1)
        self.target.copy_from(self.online)
        self.learning_steps = 0

    def q_values(self, state: np.ndarray) -> np.ndarray:
        return self.online.predict(np.asarray(state, dtype=np.float32).reshape(1, -1))[0]

    def select_action(self, state: np.ndarray, explore: bool = False) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_size))
        return int(np.argmax(self.q_values(state)))

    def remember(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ) -> None:
        self.memory.append((
            np.asarray(state, dtype=np.float32).copy(), int(action), float(reward),
            np.asarray(next_state, dtype=np.float32).copy(), bool(done),
        ))

    def learn(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None
        indices = self.rng.choice(len(self.memory), size=self.batch_size, replace=False)
        batch = [self.memory[index] for index in indices]
        states = np.stack([item[0] for item in batch])
        actions = np.asarray([item[1] for item in batch], dtype=np.int64)
        rewards = np.asarray([item[2] for item in batch], dtype=np.float32)
        next_states = np.stack([item[3] for item in batch])
        dones = np.asarray([item[4] for item in batch], dtype=np.float32)

        # Double DQN: online network selects; target network evaluates the next action.
        next_actions = np.argmax(self.online.predict(next_states), axis=1)
        next_values = self.target.predict(next_states)[np.arange(self.batch_size), next_actions]
        td_targets = rewards + self.gamma * (1 - dones) * next_values
        targets = self.online.predict(states)
        targets[np.arange(self.batch_size), actions] = td_targets
        loss = self.online.train(states, targets, self.learning_rate)

        self.learning_steps += 1
        if self.learning_steps % self.target_update_steps == 0:
            self.target.copy_from(self.online)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss

    def predict_actions(self, states: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        return np.concatenate([
            np.argmax(self.online.predict(states[start : start + batch_size]), axis=1)
            for start in range(0, len(states), batch_size)
        ]).astype(np.int64)
