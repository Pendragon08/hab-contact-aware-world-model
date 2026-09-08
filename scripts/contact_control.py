"""Causal action modifiers; gripper absolute-position control is never scaled."""
import numpy as np


class ContactControl:
    def __init__(self, method, mapping, config, max_steps=200):
        if method not in ("sac", "conservative", "force_rule"):
            raise ValueError(method)
        self.method, self.cfg, self.max_steps = method, config, max_steps
        self.mapping = {k: tuple(v) for k, v in mapping.items()}
        assert self.mapping == {"arm": (0, 7), "gripper": (7, 8), "body": (8, 11), "base": (11, 13)}, self.mapping
        self.motion = np.array(list(range(7)) + [10, 11, 12])
        self.previous = None

    def __call__(self, action, previous_force, previous_cumulative, step):
        x = np.asarray(action).copy()
        assert x.shape == (13,) and np.isfinite(x).all()
        assert 1 <= step <= self.max_steps
        force, cumulative = float(previous_force), float(previous_cumulative)
        assert np.isfinite([force, cumulative]).all() and min(force, cumulative) >= 0
        scale = 1.0
        if self.method != "sac":
            motion = x[self.motion].copy()
            if self.previous is not None:
                alpha = self.cfg["smoothing_alpha"]
                motion = alpha * motion + (1-alpha) * self.previous
            self.previous = motion.copy()
            if self.method == "conservative":
                scale = self.cfg["conservative_scale"]
            else:
                remaining = max(0., self.cfg["cumulative_limit"] - cumulative)
                per_step_budget = remaining / (self.max_steps-step+1)
                scale = float(np.clip(min(1., self.cfg["force_soft_threshold"] / max(force, 1e-6),
                                          per_step_budget / max(force, 1e-6)),
                                      self.cfg["minimum_scale"], 1.))
            x[self.motion] = scale * motion
        # Match official stationary-head wrapper, without mutating actor output.
        x[8:10] = 0
        x = np.clip(x, -1, 1)
        assert x[7] == action[7], "Gripper must not be scaled or filtered"
        return x, {"scale": scale, "input_force": force, "input_cumulative_force": cumulative}
