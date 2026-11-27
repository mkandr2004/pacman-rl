import unittest
from scripts.select_navigation_checkpoint import should_promote


class SelectionTest(unittest.TestCase):
    def setUp(self):
        self.baseline = dict(wins=0, collection=.2, return_count=100,
                             return_ratio=.5, safe=True)

    def test_identical_weights_performance_is_not_an_improvement(self):
        self.assertFalse(should_promote(self.baseline, dict(self.baseline)))

    def test_safety_collection_and_loop_regressions_are_rejected(self):
        for change in (dict(safe=False), dict(collection=.1),
                       dict(return_count=120), dict(return_ratio=.6)):
            with self.subTest(change=change):
                self.assertFalse(should_promote(self.baseline, {**self.baseline, **change}))

    def test_more_collection_with_fewer_loops_is_accepted(self):
        candidate = {**self.baseline, "collection": .3,
                     "return_count": 90, "return_ratio": .4}
        self.assertTrue(should_promote(self.baseline, candidate))

    def test_actual_wins_take_priority_over_loop_count(self):
        candidate = {**self.baseline, "wins": 1, "collection": .5,
                     "return_count": 150, "return_ratio": .6}
        self.assertTrue(should_promote(self.baseline, candidate))
