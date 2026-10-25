from __future__ import annotations

"""Synthetic action-dependent IDS simulator for safe policy experiments.

This environment is intentionally synthetic. It models how allow/alert/block can
change a persistent threat session; it is not a network emulator or a firewall.
"""

import numpy as np

from cybersecurity_ml.ids.ddqn import ACTION_NAMES

FEATURE_NAMES = (
    "threat_level",
    "request_rate",
    "failed_auth_rate",
    "port_scan_rate",
    "previous_action",
    "time_remaining",
)


class ActionDependentThreatEnv:
    """Small stochastic MDP where response choices change future threat state."""

    def __init__(self, episode_steps: int = 32, seed: int = 42) -> None:
        if episode_steps < 2:
            raise ValueError("episode_steps must be at least 2")
        self.episode_steps = episode_steps
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.threat_level = 0.0
        self.previous_action = 0

    def _observation(self) -> np.ndarray:
        threat = self.threat_level
        return np.asarray([
            threat,
            np.clip(0.15 + 0.8 * threat + self.rng.normal(0, 0.04), 0, 1),
            np.clip(0.05 + 0.9 * threat + self.rng.normal(0, 0.05), 0, 1),
            np.clip(0.02 + 0.85 * threat + self.rng.normal(0, 0.06), 0, 1),
            self.previous_action / (len(ACTION_NAMES) - 1),
            max(0.0, 1.0 - self.t / self.episode_steps),
        ], dtype=np.float32)

    def reset(self) -> np.ndarray:
        """Start a scenario with a randomly selected benign or active session."""
        self.t = 0
        self.previous_action = 0
        self.threat_level = (
            float(self.rng.uniform(0.65, 0.95))
            if self.rng.random() < 0.35
            else float(self.rng.uniform(0.0, 0.25))
        )
        return self._observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, int | float]]:
        """Apply one response and return its effect on the next simulated state."""
        action = int(action)
        if action not in range(len(ACTION_NAMES)):
            raise ValueError("action must be 0=allow, 1=alert, or 2=block")
        malicious = int(self.threat_level >= 0.55)
        rewards = ((1.0, -0.3, -1.2), (-1.0, 0.35, 1.0))
        reward = rewards[malicious][action]

        # Responses alter the persistent session. Blocking strongly suppresses it;
        # alerting slows it; allowing it to continue gives it room to escalate.
        if malicious:
            effect = (0.12, -0.22, -0.62)[action]
        else:
            effect = (0.0, 0.0, 0.0)[action]
        arrival = 0.28 if action == 0 else (0.12 if action == 1 else 0.03)
        new_threat = self.threat_level + effect + self.rng.normal(0, 0.035)
        if self.rng.random() < arrival:
            new_threat += float(self.rng.uniform(0.12, 0.32))
        self.threat_level = float(np.clip(new_threat, 0.0, 1.0))
        self.previous_action = action
        self.t += 1
        done = self.t >= self.episode_steps
        next_state = np.zeros(len(FEATURE_NAMES), dtype=np.float32) if done else self._observation()
        return next_state, float(reward), done, {
            "label": malicious,
            "threat_level": self.threat_level,
        }
