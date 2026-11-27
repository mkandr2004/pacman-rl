import random
import unittest

from rl.curriculum import CurriculumEnv, STAGES
from rl.ghost_forecast import danger_map, forecast_ghosts
from rl.pacman_env import PacmanEnv


class CurriculumForecastTests(unittest.TestCase):
    def test_curriculum_keeps_shape_and_changes_only_active_ghosts(self):
        for stage in STAGES:
            env = CurriculumEnv(stage,danger_channel=True)
            try:
                obs,_ = env.reset(seed=12)
                self.assertTrue(env.observation_space.contains(obs))
                self.assertEqual(len(env._last_raw_observation["ghosts"]),stage)
                self.assertEqual(env._last_raw_observation["points_remaining"],81)
                self.assertEqual(env._last_raw_observation["coins_remaining"],11)
                self.assertEqual(bool(obs["danger"].any()),bool(stage))
                obs,*_ = env.step(3)
                self.assertTrue(env.observation_space.contains(obs))
            finally:
                env.close()

    def test_seven_ghosts_produce_identical_engine_transitions(self):
        def episode(env):
            try:
                obs,_=env.reset(seed=555)
                states=[]
                for action in [3,1,2,0]*12:
                    obs,r,t,u,i=env.step(action)
                    states.append((env._last_raw_observation,r,t,u,i))
                    if t or u:
                        break
                return states
            finally:
                env.close()
        self.assertEqual(episode(PacmanEnv()),episode(CurriculumEnv(7,danger_channel=True)))

    def test_forecast_reuses_exact_ghost_transitions_and_is_read_only(self):
        env = PacmanEnv(frames_per_action=1)
        try:
            obs,_=env.reset(seed=123)
            game=env._game
            player=game._screen_field_mapper.get_pacman().get_rect().copy()
            snapshots=game.observation()
            rng=random.getstate()
            # LEFT is physically blocked at the start, so the player's rectangle
            # stays fixed. Equal seeds are used only for this parity test.
            forecasts=forecast_ghosts(game,[player]*90,seed=123)
            self.assertEqual(snapshots,game.observation())
            self.assertEqual(rng,random.getstate())
            for rectangles in forecasts:
                actual=game._screen_field_mapper.get_ghosts()
                self.assertEqual([g.get_rect() for g in actual],list(rectangles))
                env.step(0)
        finally:
            env.close()


if __name__=="__main__":
    unittest.main()
