"""Curriculum observations, exploration and auditable replay for the DQN."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from rl.curriculum import CurriculumEnv, STAGES
from rl.interaction_dqn import UniformReplay
from rl.wrappers import NormalizePositions


EVENTS = ("collection", "lemon", "wall", "useful_wall", "after_wall", "return", "life_lost", "victory")


class History(gym.Wrapper):
    """Preserve the DQN input layout while the number of active ghosts varies."""

    def __init__(self, env):
        super().__init__(env)
        base = env.unwrapped
        self.visits = np.zeros((1, base.rows, base.cols), dtype=np.float32)
        self.motion = np.zeros(8 + base.ghost_count * 2, dtype=np.float32)
        self.observation_space = spaces.Dict({
            **env.observation_space.spaces,
            "visits": spaces.Box(0, 1, self.visits.shape, dtype=np.float32),
            "motion": spaces.Box(-1, 1, self.motion.shape, dtype=np.float32),
        })

    def observe(self, obs):
        raw = self.unwrapped._last_raw_observation
        row, col = self.unwrapped._grid_position(raw["pacman"])
        self.visits[0, row, col] += 1
        return {**obs, "visits": np.clip(np.log1p(self.visits)/np.log(101), 0, 1).astype(np.float32),
                "motion": self.motion.copy()}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.visits.fill(0)
        self.motion.fill(0)
        return self.observe(obs), info

    def step(self, action):
        base = self.unwrapped
        before = base._last_raw_observation
        obs, reward, terminal, truncated, info = self.env.step(action)
        after = base._last_raw_observation
        self.motion[:6] = self.motion[2:8].copy()
        self.motion[6:8] = np.clip((np.array(after["pacman"])-before["pacman"])/60, -1, 1)
        for slot, old, new in zip(base._ghost_slots, before["ghosts"], after["ghosts"]):
            self.motion[8+2*slot:10+2*slot] = np.clip((np.array(new)-old)/60, -1, 1)
        return self.observe(obs), reward, terminal, truncated, info


def make_curriculum_env(ghosts, render_mode=None):
    return History(NormalizePositions(CurriculumEnv(ghosts, max_episode_steps=3000,
                                                    render_mode=render_mode)))


class BurstExploration:
    """Repeat a uniformly drawn legal direction for a short burst while it remains legal."""

    def __init__(self, rng):
        self.rng = rng
        self.remaining = 0
        self.direction = 0

    def choose(self, model, packed, mask, epsilon):
        if self.remaining and mask[self.direction]:
            self.remaining -= 1
            return self.direction, True
        self.remaining = 0
        if self.rng.random() < epsilon:
            self.direction = int(self.rng.choice(np.flatnonzero(mask)))
            self.remaining = int(self.rng.integers(2, 7))
            return self.direction, True
        return model.greedy_action(packed, mask), False


class AuditedReplay(UniformReplay):
    def __init__(self, capacity, example, rng):
        super().__init__(capacity, example, rng)
        self.events = np.zeros((capacity, len(EVENTS)), dtype=np.bool_)
        self.insertions = np.zeros(capacity, dtype=np.int64)
        self.evictions = np.zeros(len(EVENTS), dtype=np.int64)
        self.sampled_events = np.zeros(len(EVENTS), dtype=np.int64)
        self.sampled = 0
        self.replaced = 0

    def add(self, *transition, events, step):
        index = self.index
        if self.size == self.capacity:
            self.replaced += 1
            self.evictions += self.events[index]
        super().add(*transition)
        self.events[index] = events
        self.insertions[index] = step

    def arrays(self):
        return (*self.current, *self.following, self.actions, self.rewards, self.next_masks, self.terminated)

    def draw(self, count):
        indices = self.rng.integers(self.size, size=count)
        self.sampled_events += self.events[indices].sum(axis=0)
        self.sampled += count
        return [field[indices] for field in self.arrays()]

    def snapshot(self):
        return dict(size=self.size, resident=dict(zip(EVENTS, self.events[:self.size].sum(axis=0).tolist())),
                    overwritten=self.replaced, overwritten_events=dict(zip(EVENTS, self.evictions.tolist())),
                    sampled=self.sampled, sampled_events=dict(zip(EVENTS, self.sampled_events.tolist())),
                    oldest=int(self.insertions[:self.size].min()), newest=int(self.insertions[:self.size].max()))

    def save(self, path, rng):
        # The main buffer remains uniform. The frozen subset preserves rare outcomes
        # alongside a uniform sample of this learner's own other transitions.
        rare = np.flatnonzero(self.events[:self.size, [1, 3, 7]].any(axis=1))
        protected = rng.choice(rare, min(4096, len(rare)), replace=False)
        ordinary = np.setdiff1d(np.arange(self.size), protected, assume_unique=True)
        sample = rng.choice(ordinary, min(16384-len(protected), len(ordinary)), replace=False)
        keep = np.concatenate((protected, sample))
        fields = [field[keep] for field in self.arrays()]
        with Path(path).open("xb") as stream:
            np.savez_compressed(stream, *fields, events=self.events[keep], decisions=self.insertions[keep])
        return dict(kept=len(keep), rare_kept=len(protected))


class RetainedReplay:
    def __init__(self, paths, rng):
        self.rng = rng
        self.stages = []
        self.sampled = [0] * len(paths)
        self.sampled_events = [np.zeros(len(EVENTS), dtype=np.int64) for _ in paths]
        for path in paths:
            with np.load(path, allow_pickle=False) as stream:
                self.stages.append(([stream[f"arr_{i}"] for i in range(10)], stream["events"]))

    def batch(self, current, count=64):
        retained_count = 16 if self.stages else 0
        fields = current.draw(count-retained_count)
        if self.stages:
            allocations = self.rng.multinomial(retained_count, [1/len(self.stages)] * len(self.stages))
            for j, ((older, tags), amount) in enumerate(zip(self.stages, allocations)):
                if amount:
                    indices = self.rng.integers(len(tags), size=amount)
                    fields = [np.concatenate((new, past[indices])) for new, past in zip(fields, older)]
                    self.sampled[j] += int(amount)
                    self.sampled_events[j] += tags[indices].sum(axis=0)
        tensors = [torch.as_tensor(value) for value in fields]
        return tensors[:3], tensors[3:6], *tensors[6:]

    def snapshot(self):
        return [dict(ghosts=ghosts, sampled=count, events=dict(zip(EVENTS, tags.tolist())))
                for ghosts, count, tags in zip(STAGES, self.sampled, self.sampled_events)]
