"""Small deterministic tests for the custom sequence logic."""

import unittest

import numpy as np

from sequence_v2 import slot_teams
from sequence_v4 import duration_cleanup, runs
from occlusion_bridge import bridge_occlusions
from sequence_v13 import constrained_fusion


class SequenceTests(unittest.TestCase):
    def test_official_metadata_has_all_slots(self):
        teams = slot_teams(__import__("sequence_v2").META)
        self.assertEqual(len(teams), 46)
        self.assertEqual(set(teams.tolist()), {0, 1})

    def test_short_run_between_same_state_is_removed(self):
        states = np.array([0] * 20 + [1] * 4 + [0] * 20, dtype=np.uint8)
        score = np.zeros((len(states), 3), dtype=np.float32)
        cleaned = duration_cleanup(states, score)
        np.testing.assert_array_equal(cleaned, np.zeros(len(states), dtype=np.uint8))

    def test_segment_boundaries_are_half_open(self):
        states = np.array([0, 0, 1, 1, 1, 2], dtype=np.uint8)
        starts, ends = runs(states)
        np.testing.assert_array_equal(starts, [0, 2, 5])
        np.testing.assert_array_equal(ends, [2, 5, 6])

    def test_occlusion_bridge_requires_matching_strong_flanks(self):
        states = np.array([0] * 4 + [2] * 3 + [0] * 4, dtype=np.uint8)
        dh = np.array([1.] * 4 + [np.nan] * 3 + [1.] * 4)
        da = np.array([4.] * 4 + [np.nan] * 3 + [4.] * 4)
        home = np.array([.8] * 4 + [0.] * 3 + [.8] * 4)
        away = np.zeros(len(states))
        bridged, spans = bridge_occlusions(
            states, dh, da, home, away, max_gap=3, flank=2, min_anchor=.3)
        np.testing.assert_array_equal(bridged, np.zeros(len(states), dtype=np.uint8))
        self.assertEqual(len(spans), 1)

    def test_visual_fusion_bridges_dead_uncertainty_without_island(self):
        base = np.array(list("AAADDDHHHH"))
        visual = np.array(list("AAADDDAHHH"))
        tracks = __import__("pandas").DataFrame({
            "frame": [6], "track_id": [0], "trajectory_valid": [True]})
        fused, audit = constrained_fusion(base, visual, tracks, max_bridge=10)
        np.testing.assert_array_equal(fused, np.array(list("AAAAAAAHHH")))
        self.assertEqual(len(audit), 1)

    def test_visual_fusion_rejects_unanchored_island(self):
        base = np.array(list("HHHDDDHHH"))
        visual = np.array(list("HHHDADHHH"))
        tracks = __import__("pandas").DataFrame({
            "frame": [4], "track_id": [0], "trajectory_valid": [True]})
        fused, audit = constrained_fusion(base, visual, tracks, max_bridge=2)
        np.testing.assert_array_equal(fused, base)
        self.assertEqual(len(audit), 0)


if __name__ == "__main__":
    unittest.main()
