import unittest
import numpy as np
from rl.navigation_lab import ScenarioEnv, HistoryEnv
from rl.pacman_env import PacmanEnv
from rl.wrappers import NormalizePositions


class NavigationLabTests(unittest.TestCase):
    def run_actions(self, factory):
        env = factory()
        try:
            obs, _ = env.reset(seed=1000)
            masks = []
            for action in [3]*10+[1]*10+[0,1]*4:
                masks.append(env.unwrapped.action_masks().copy())
                obs, reward, _, _, info = env.step(action)
                self.assertTrue(env.observation_space.contains(obs))
            return obs, masks, reward, info
        finally:
            env.close()

    def test_history_does_not_change_transitions_rewards_or_masks(self):
        a, ma, ra, ia = self.run_actions(lambda:NormalizePositions(PacmanEnv()))
        b, mb, rb, ib = self.run_actions(lambda:NormalizePositions(HistoryEnv()))
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(ma, mb)
        self.assertEqual(ra, rb)
        self.assertEqual(ia, ib)

    def test_ghost_free_scenario_has_truthful_empty_grid_and_fixed_shape(self):
        for factory in (ScenarioEnv, HistoryEnv):
            env = NormalizePositions(factory(prefix=[3]*10, ghosts=False))
            try:
                obs, _ = env.reset(seed=1000)
                self.assertTrue(env.observation_space.contains(obs))
                self.assertEqual(obs['grid'][5].sum(), 0)
                self.assertEqual(obs['ghost_positions'].sum(), 0)
                obs, _, _, _, _ = env.step(3)
                self.assertTrue(env.observation_space.contains(obs))
                self.assertEqual(len(env.unwrapped._last_raw_observation['ghosts']), 0)
            finally:
                env.close()

    def test_identical_prefix_means_identical_physical_scenario(self):
        states=[]
        for factory in (ScenarioEnv, HistoryEnv):
            env = factory(prefix=[3]*10+[1]*10)
            try:
                env.reset(seed=1000)
                states.append(env._last_raw_observation)
                self.assertEqual(env._elapsed_steps, 0)
            finally:
                env.close()
        self.assertEqual(states[0], states[1])

    def test_setup_prefix_does_not_consume_scenario_time_budget(self):
        env = ScenarioEnv(prefix=[3]*10,max_episode_steps=2)
        try:
            env.reset(seed=1000)
            _, _, _, truncated, _ = env.step(3)
            self.assertFalse(truncated)
            _, _, _, truncated, _ = env.step(3)
            self.assertTrue(truncated)
        finally:
            env.close()

    def test_lemon_fixtures_use_finite_normal_power_and_real_map(self):
        for fixture in ('lemon_upper','lemon_lower'):
            env=HistoryEnv(fixture=fixture,ghosts=False)
            try:
                obs,_=env.reset(seed=1000)
                self.assertTrue(env.observation_space.contains(obs))
                raw=env._last_raw_observation
                self.assertTrue(0 < raw['effects'][2] < 1)
                self.assertEqual(len(raw['bonus_positions'][2]),1)
                self.assertEqual(env._elapsed_steps,0)
                self.assertEqual(len(raw['ghosts']),0)
            finally:
                env.close()

    def test_prefix_history_matches_actual_replayed_history(self):
        prefix=[3]*10+[1]*10
        env=HistoryEnv()
        try:
            env.reset(seed=1000)
            for action in prefix:
                obs,*_=env.step(action)
            visits,motion=obs['visits'],obs['motion']
        finally:
            env.close()
        env=HistoryEnv(prefix=prefix)
        try:
            obs,_=env.reset(seed=1000)
            np.testing.assert_array_equal(visits,obs['visits'])
            np.testing.assert_array_equal(motion,obs['motion'])
        finally:
            env.close()

    def test_wall_connectivity_metric_measures_access_not_direction(self):
        from scripts.navigation_study import reachable_collectibles
        walls=tuple((c*60,r*60) for r in range(3) for c in range(5)
                    if r in (0,2) or c in (0,4) or (r,c)==(1,2))
        raw=dict(screen_size=(300,180),pacman=(60,60),walls=walls,points=((180,60),),coins=())
        self.assertEqual(reachable_collectibles(raw),0)
        opened={**raw,'walls':tuple(w for w in walls if w!=(120,60))}
        self.assertEqual(reachable_collectibles(opened),1)

    def test_legal_u_turn_remains_available(self):
        env=HistoryEnv()
        try:
            env.reset(seed=1000)
            env.step(3)
            self.assertTrue(env.action_masks()[2])
            self.assertTrue(env.action_masks()[3])
        finally:
            env.close()

    def test_spatial_policy_supports_rollout_and_training_minibatches(self):
        import torch
        from sb3_contrib import MaskablePPO
        from rl.spatial_policy import SpatialHistoryExtractor
        torch.set_num_threads(1)
        env = NormalizePositions(HistoryEnv(max_episode_steps=16))
        try:
            model = MaskablePPO('MultiInputPolicy', env, n_steps=8, batch_size=4, n_epochs=1,
                policy_kwargs=dict(features_extractor_class=SpatialHistoryExtractor,
                                   net_arch=dict(pi=[16],vf=[16])), device='cpu', seed=42)
            before = {key:value.detach().clone() for key,value in model.policy.state_dict().items()}
            model.learn(8)
            self.assertTrue(any(not torch.equal(before[key],value)
                                for key,value in model.policy.state_dict().items()))
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
