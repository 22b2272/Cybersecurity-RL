"""Offline response environment for labeled intrusion-detection records."""
import numpy as np

# Rows are true class [normal, attack], columns are actions [allow, alert, block].
DEFAULT_REWARDS = np.asarray(
    [[1.0, -0.25, -1.0], [-1.0, 0.25, 1.0]], dtype=np.float32
)


class IDSResponseEnv:
    """Replay labeled observations and reward proportionate IDS response choices.

    This is an offline replay environment, not a live network simulator. Actions
    affect the immediate reward; the next recorded observation is exogenous.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        episode_steps: int = 512,
        rewards: np.ndarray = DEFAULT_REWARDS,
        seed: int = 42,
    ) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int8)
        if self.features.ndim != 2 or len(self.features) != len(self.labels) or len(self.labels) == 0:
            raise ValueError("features must be a non-empty 2D array aligned with labels")
        if not np.isin(self.labels, [0, 1]).all():
            raise ValueError("labels must be binary: normal=0, attack=1")
        if episode_steps < 1:
            raise ValueError("episode_steps must be positive")
        self.rewards = np.asarray(rewards, dtype=np.float32)
        if self.rewards.shape != (2, 3):
            raise ValueError("rewards must have shape (2 classes, 3 response actions)")
        self.episode_steps = min(episode_steps, len(self.labels))
        self.rng = np.random.default_rng(seed)
        self.indices = np.empty(0, dtype=np.int64)
        self.position = 0

    def reset(self) -> np.ndarray:
        """Begin a shuffled episode and return its first observation."""
        self.indices = self.rng.permutation(len(self.labels))[: self.episode_steps]
        self.position = 0
        return self.features[self.indices[self.position]]

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, int]]:
        """Apply allow/alert/block and return next state, reward, done, metadata."""
        if not 0 <= int(action) < 3:
            raise ValueError("action must be 0=allow, 1=alert, or 2=block")
        row_index = int(self.indices[self.position])
        label = int(self.labels[row_index])
        reward = float(self.rewards[label, int(action)])
        self.position += 1
        done = self.position >= len(self.indices)
        next_state = (
            np.zeros(self.features.shape[1], dtype=np.float32)
            if done
            else self.features[self.indices[self.position]]
        )
        return next_state, reward, done, {"label": label}
