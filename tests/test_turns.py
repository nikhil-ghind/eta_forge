import unittest

from eta_forge.graph.turns import (
    DEFAULT_TURN_PENALTY_S,
    class_mix,
    classify_turn,
    route_turn_profile,
    turn_penalty_s,
)
from eta_forge.types import HighwayClass, Location, RouteLeg, TurnType


def leg(
    segment_id=1,
    bearing_in=0.0,
    bearing_out=0.0,
    length_m=100.0,
    highway_class=HighwayClass.RESIDENTIAL,
    signal=False,
):
    return RouteLeg(
        segment_id=segment_id,
        source_vertex=segment_id,
        target_vertex=segment_id + 1,
        length_m=length_m,
        highway_class=highway_class,
        speed_kph=30.0,
        duration_s=length_m / (30.0 / 3.6),
        bearing_in_deg=bearing_in,
        bearing_out_deg=bearing_out,
        has_signal_at_end=signal,
    )


class TestClassifyTurn(unittest.TestCase):
    def test_straight(self):
        self.assertIs(classify_turn(90.0, 90.0), TurnType.STRAIGHT)
        self.assertIs(classify_turn(90.0, 105.0), TurnType.STRAIGHT)
        self.assertIs(classify_turn(90.0, 75.0), TurnType.STRAIGHT)

    def test_slight(self):
        self.assertIs(classify_turn(0.0, 30.0), TurnType.SLIGHT_RIGHT)
        self.assertIs(classify_turn(0.0, 330.0), TurnType.SLIGHT_LEFT)

    def test_full_turns(self):
        self.assertIs(classify_turn(0.0, 90.0), TurnType.RIGHT)
        self.assertIs(classify_turn(0.0, 270.0), TurnType.LEFT)

    def test_sharp(self):
        self.assertIs(classify_turn(0.0, 140.0), TurnType.SHARP_RIGHT)
        self.assertIs(classify_turn(0.0, 220.0), TurnType.SHARP_LEFT)

    def test_uturn(self):
        self.assertIs(classify_turn(0.0, 180.0), TurnType.UTURN)
        self.assertIs(classify_turn(0.0, 175.0), TurnType.UTURN)

    def test_wraps_across_north(self):
        # The seam case: 350 then 10 is a slight drift, not a u-turn.
        self.assertIs(classify_turn(350.0, 10.0), TurnType.STRAIGHT)
        self.assertIs(classify_turn(355.0, 35.0), TurnType.SLIGHT_RIGHT)

    def test_boundaries(self):
        self.assertIs(classify_turn(0.0, 20.0), TurnType.STRAIGHT)
        self.assertIs(classify_turn(0.0, 20.1), TurnType.SLIGHT_RIGHT)
        self.assertIs(classify_turn(0.0, 45.0), TurnType.SLIGHT_RIGHT)
        self.assertIs(classify_turn(0.0, 45.1), TurnType.RIGHT)
        self.assertIs(classify_turn(0.0, 120.0), TurnType.RIGHT)
        self.assertIs(classify_turn(0.0, 120.1), TurnType.SHARP_RIGHT)
        self.assertIs(classify_turn(0.0, 160.0), TurnType.SHARP_RIGHT)
        self.assertIs(classify_turn(0.0, 160.1), TurnType.UTURN)


class TestTurnPenalty(unittest.TestCase):
    def test_left_costs_more_than_right(self):
        # Right-hand traffic: a left crosses oncoming lanes.
        self.assertGreater(
            turn_penalty_s(TurnType.LEFT), turn_penalty_s(TurnType.RIGHT)
        )
        self.assertGreater(
            turn_penalty_s(TurnType.SHARP_LEFT), turn_penalty_s(TurnType.SHARP_RIGHT)
        )

    def test_straight_is_free(self):
        self.assertEqual(turn_penalty_s(TurnType.STRAIGHT), 0.0)

    def test_config_override(self):
        self.assertEqual(turn_penalty_s(TurnType.LEFT, {"left": 30.0}), 30.0)

    def test_partial_override_falls_back(self):
        self.assertEqual(
            turn_penalty_s(TurnType.RIGHT, {"left": 30.0}),
            DEFAULT_TURN_PENALTY_S[TurnType.RIGHT],
        )


class TestRouteTurnProfile(unittest.TestCase):
    def test_straight_route_has_no_turns(self):
        legs = [leg(1, 0.0, 0.0), leg(2, 0.0, 0.0), leg(3, 0.0, 0.0)]
        profile = route_turn_profile(legs)
        self.assertEqual(profile.total_turns, 0)
        self.assertEqual(profile.total_penalty_s, 0.0)

    def test_counts_each_manoeuvre(self):
        legs = [
            leg(1, 0.0, 0.0),
            leg(2, 90.0, 90.0),     # right
            leg(3, 0.0, 0.0),       # left
            leg(4, 180.0, 180.0),   # u-turn from 0 -> 180
        ]
        profile = route_turn_profile(legs)
        self.assertEqual(profile.counts[TurnType.RIGHT.value], 1)
        self.assertEqual(profile.counts[TurnType.LEFT.value], 1)
        self.assertEqual(profile.counts[TurnType.UTURN.value], 1)
        self.assertEqual(profile.left_turns, 1)
        self.assertEqual(profile.right_turns, 1)
        self.assertEqual(profile.major_turns, 3)
        self.assertEqual(profile.uturns, 1)

    def test_uses_exit_and_entry_bearings_not_chords(self):
        # A leg that curves 0 -> 80 followed by one entering at 80 is straight
        # through, even though the two chord bearings differ by 80 degrees.
        legs = [leg(1, 0.0, 80.0), leg(2, 80.0, 80.0)]
        profile = route_turn_profile(legs)
        self.assertEqual(profile.total_turns, 0)

    def test_signals_are_counted_and_charged(self):
        legs = [leg(1, 0.0, 0.0, signal=True), leg(2, 0.0, 0.0, signal=True)]
        profile = route_turn_profile(legs, signal_penalty_s=8.0)
        self.assertEqual(profile.signal_count, 2)
        self.assertEqual(profile.total_penalty_s, 16.0)

    def test_class_changes(self):
        legs = [
            leg(1, 0.0, 0.0, highway_class=HighwayClass.RESIDENTIAL),
            leg(2, 0.0, 0.0, highway_class=HighwayClass.PRIMARY),
            leg(3, 0.0, 0.0, highway_class=HighwayClass.PRIMARY),
            leg(4, 0.0, 0.0, highway_class=HighwayClass.MOTORWAY),
        ]
        self.assertEqual(route_turn_profile(legs).class_changes, 2)

    def test_max_turn_deg(self):
        legs = [leg(1, 0.0, 0.0), leg(2, 100.0, 100.0), leg(3, 110.0, 110.0)]
        self.assertAlmostEqual(route_turn_profile(legs).max_turn_deg, 100.0)

    def test_single_leg_and_empty(self):
        self.assertEqual(route_turn_profile([leg(1)]).total_turns, 0)
        self.assertEqual(route_turn_profile([]).total_turns, 0)

    def test_feature_dict_is_flat_and_named(self):
        legs = [leg(1, 0.0, 0.0), leg(2, 90.0, 90.0)]
        features = route_turn_profile(legs).to_dict()
        self.assertEqual(features["turn_right"], 1.0)
        self.assertIn("turn_penalty_s", features)
        self.assertIn("signal_count", features)
        self.assertTrue(all(isinstance(v, float) for v in features.values()))


class TestClassMix(unittest.TestCase):
    def test_weighted_by_distance_not_leg_count(self):
        legs = [leg(i, length_m=20.0, highway_class=HighwayClass.SERVICE)
                for i in range(1, 11)]
        legs.append(leg(11, length_m=3000.0, highway_class=HighwayClass.MOTORWAY))
        mix = class_mix(legs)
        self.assertGreater(mix["motorway"], 0.9)
        self.assertLess(mix["service"], 0.1)

    def test_sums_to_one(self):
        legs = [
            leg(1, length_m=100.0, highway_class=HighwayClass.PRIMARY),
            leg(2, length_m=300.0, highway_class=HighwayClass.RESIDENTIAL),
        ]
        self.assertAlmostEqual(sum(class_mix(legs).values()), 1.0, places=9)

    def test_covers_every_class_key(self):
        mix = class_mix([leg(1)])
        self.assertEqual(set(mix), {c.value for c in HighwayClass})

    def test_empty_route(self):
        mix = class_mix([])
        self.assertEqual(sum(mix.values()), 0.0)


if __name__ == "__main__":
    unittest.main()
