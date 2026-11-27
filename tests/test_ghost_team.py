import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from rl.ghost_team_env import REWARD_PROFILES, GhostTeamEnv, TeamVecEnv, ghost_rewards


class Opponent:
    def predict(self, observation, *, deterministic, action_masks):
        return int(np.flatnonzero(action_masks)[0]), None


class GhostTeamTest(unittest.TestCase):
    def test_capture_credit_is_local_and_team_return_is_shared(self):
        rewards, credit, shared = ghost_rewards((0,1),[(0,10),(4,70)],True,False)
        self.assertEqual(credit,[1,0])
        self.assertAlmostEqual(shared,.5)
        np.testing.assert_allclose(rewards,[1.1,.5])
        rewards, _, _ = ghost_rewards((0,1),[],False,False)
        np.testing.assert_array_equal(rewards,[0.,0.])

    def test_balanced_reward_credits_a_fast_capture_and_ignores_classic_captures(self):
        reward, credit, shared = ghost_rewards(
            (0, 1), [(1, 10), (4, 70)], True, False,
            config=REWARD_PROFILES["balanced"], fast_slots=(1,))
        self.assertEqual(credit, [0, 1])
        self.assertAlmostEqual(shared, .45)
        np.testing.assert_allclose(reward, [.45, 1.35])
        reward, _, _ = ghost_rewards((0, 1), [(0, 10)], False, False,
                                    config=REWARD_PROFILES["balanced"], fast_slots=(1,))
        np.testing.assert_allclose(reward, [.65, .05])

    def test_team_moves_two_ghosts_in_one_game_tick(self):
        env=GhostTeamEnv(Opponent(),2)
        self.addCleanup(env.close)
        observation,_=env.reset(seed=130)
        self.assertTrue(all(env.observation_space.contains({k:v[i] for k,v in observation.items()})
                            for i in range(2)))
        np.testing.assert_array_equal(observation["scene"][0],observation["scene"][1])
        self.assertFalse(np.array_equal(observation["agent"][0],observation["agent"][1]))
        actions=np.argmax(env.action_masks(),axis=1)
        _,reward,ended,truncated,info=env.step(actions)
        self.assertFalse(ended or truncated)
        self.assertEqual(info["simulated_ms"],40)
        self.assertEqual(len(reward),2)
        self.assertEqual(set(info["ghost_moved_frames"]),set(env.slots))
        self.assertTrue(all(v>0 for v in info["ghost_moved_frames"].values()))

    def test_team_vec_timeout_has_distinct_terminal_observations(self):
        match=GhostTeamEnv(Opponent(),2,max_episode_steps=1)
        vec=TeamVecEnv(match,7000)
        self.addCleanup(vec.close)
        vec.reset()
        masks=np.stack(vec.env_method("action_masks"))
        _,reward,dones,info=vec.step(np.argmax(masks,axis=1))
        np.testing.assert_array_equal(dones,[True,True])
        np.testing.assert_array_equal(reward,[0.,0.])
        self.assertTrue(all(row["TimeLimit.truncated"] and not row["ghost_win"] for row in info))
        first,last=[row["terminal_observation"] for row in info]
        self.assertFalse(np.array_equal(first["agent"],last["agent"]))
        self.assertEqual(vec.next_seed,7002)

    def test_same_policy_input_for_two_four_and_seven(self):
        expected=None
        for count in (2,4,7):
            with self.subTest(count=count):
                env=GhostTeamEnv(Opponent(),count)
                try:
                    observation,_=env.reset(seed=131)
                    if expected is None:
                        expected=env.observation_space
                    self.assertEqual(env.observation_space,expected)
                    self.assertEqual(env.action_masks().shape,(count,5))
                    self.assertEqual(len(set(env.slots)),count)
                    env.step(np.argmax(env.action_masks(),axis=1))
                    self.assertTrue(env.observation_space.contains({k:v[-1] for k,v in observation.items()}))
                finally:
                    env.close()
