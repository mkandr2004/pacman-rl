import unittest

import numpy as np
import torch

from rl.curriculum import STAGES
from rl.dqn_curriculum import AuditedReplay, BurstExploration, EVENTS, RetainedReplay, make_curriculum_env
from rl.imitation_env import make_imitation_env
from rl.interaction_dqn import pack_observation


class CurriculumDQNTests(unittest.TestCase):
    def test_same_full_level_and_network_inputs(self):
        normal = make_imitation_env(danger=False)
        curriculum = make_curriculum_env(7)
        try:
            a = normal.reset(seed=27)[0]
            b = curriculum.reset(seed=27)[0]
            rng = np.random.default_rng(60)
            for _ in range(24):
                for left, right in zip(pack_observation(a), pack_observation(b)):
                    np.testing.assert_array_equal(left, right)
                mask = normal.unwrapped.action_masks()
                np.testing.assert_array_equal(mask, curriculum.unwrapped.action_masks())
                action = int(rng.choice(np.flatnonzero(mask)))
                a, reward, terminal, truncated, _ = normal.step(action)
                b, actual, done, cutoff, _ = curriculum.step(action)
                self.assertEqual((reward, terminal, truncated), (actual, done, cutoff))
        finally:
            normal.close()
            curriculum.close()

    def test_all_stages_support_the_same_observation_dimensions(self):
        for count in STAGES:
            env = make_curriculum_env(count)
            try:
                obs, _ = env.reset(seed=42)
                base = env.unwrapped
                self.assertEqual(len(base._last_raw_observation["ghosts"]), count)
                self.assertEqual((base._last_raw_observation["points_remaining"],
                                  base._last_raw_observation["coins_remaining"]), (81, 11))
                for _ in range(4):
                    mask = base.action_masks()
                    obs, *_ = env.step(int(np.flatnonzero(mask)[0]))
                    self.assertTrue(env.observation_space.contains(obs))
                    self.assertEqual(pack_observation(obs)[2].shape, (47,))
            finally:
                env.close()

    def test_burst_obeys_changed_mask(self):
        class Greedy:
            def greedy_action(self, packed, mask):
                return int(np.flatnonzero(mask)[0])
        explorer = BurstExploration(np.random.default_rng(10))
        first, _ = explorer.choose(Greedy(), (), np.array([False, True, False, False]), 1.)
        self.assertEqual(first, 1)
        second, random = explorer.choose(Greedy(), (), np.array([False, False, True, False]), 0.)
        self.assertEqual(second, 2)
        self.assertFalse(random)

    def test_replay_reports_overwrite_and_retained_mixing(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        packed = tuple(np.zeros(1, dtype=np.float32) for _ in range(3))
        mask = np.ones(4, dtype=np.bool_)
        replay = AuditedReplay(2, packed, np.random.default_rng(1))
        for step in (1, 2, 3):
            tags = np.zeros(len(EVENTS), dtype=np.bool_)
            tags[1] = step == 1
            replay.add(packed, mask, 0, float(step), packed, mask, False, False, events=tags, step=step)
        self.assertEqual(replay.snapshot()["overwritten_events"]["lemon"], 1)
        self.assertEqual(replay.snapshot()["resident"]["lemon"], 0)
        self.assertEqual(replay.snapshot()["oldest"], 2)
        with TemporaryDirectory() as root:
            path = Path(root) / "prior.npz"
            replay.save(path, np.random.default_rng(8))
            retained = RetainedReplay([path], np.random.default_rng(9))
            *_, terminal = retained.batch(replay)
            self.assertEqual(terminal.shape, torch.Size([64]))
            self.assertEqual(replay.sampled, 48)
            self.assertEqual(retained.sampled, [16])
