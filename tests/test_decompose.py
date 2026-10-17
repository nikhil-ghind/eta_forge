import math
import unittest

from eta_forge.decompose import compose_quantiles, decompose, naive_sum, share_of_total
from eta_forge.types import QuantilePair, ValidationError

PREP = QuantilePair(600.0, 1000.0)      # spread 400
TRAVEL = QuantilePair(900.0, 1200.0)    # spread 300
HANDOFF = QuantilePair(90.0, 210.0)     # spread 120


class TestCompose(unittest.TestCase):
    def test_medians_add(self):
        total = compose_quantiles([PREP, TRAVEL, HANDOFF], 0.35)
        self.assertAlmostEqual(total.p50, 1590.0)

    def test_full_correlation_reproduces_the_naive_sum(self):
        total = compose_quantiles([PREP, TRAVEL, HANDOFF], 1.0)
        self.assertAlmostEqual(total.p90, naive_sum([PREP, TRAVEL, HANDOFF]).p90)

    def test_zero_correlation_adds_in_quadrature(self):
        total = compose_quantiles([PREP, TRAVEL, HANDOFF], 0.0)
        self.assertAlmostEqual(total.spread, math.sqrt(400 ** 2 + 300 ** 2 + 120 ** 2))

    def test_tail_grows_with_correlation(self):
        spreads = [compose_quantiles([PREP, TRAVEL, HANDOFF], rho).spread for rho in (0.0, 0.35, 0.7, 1.0)]
        self.assertEqual(spreads, sorted(spreads))

    def test_composed_tail_never_exceeds_the_naive_sum(self):
        for rho in (0.0, 0.2, 0.5, 0.9):
            self.assertLessEqual(
                compose_quantiles([PREP, TRAVEL, HANDOFF], rho).p90,
                naive_sum([PREP, TRAVEL, HANDOFF]).p90 + 1e-9,
            )

    def test_single_component_is_unchanged(self):
        total = compose_quantiles([TRAVEL], 0.35)
        self.assertEqual((total.p50, total.p90), (TRAVEL.p50, TRAVEL.p90))

    def test_zero_components_add_nothing(self):
        total = compose_quantiles([TRAVEL, QuantilePair.zero(), QuantilePair.zero()], 0.35)
        self.assertAlmostEqual(total.p90, TRAVEL.p90)
        self.assertEqual(compose_quantiles([], 0.5), QuantilePair.zero())

    def test_rho_bounds(self):
        with self.assertRaises(ValidationError):
            compose_quantiles([PREP], 1.5)
        with self.assertRaises(ValidationError):
            compose_quantiles([PREP], -0.1)


class TestDecompose(unittest.TestCase):
    def test_breakdown_and_tail_saving(self):
        parts = decompose(
            prep=PREP, travel=TRAVEL, handoff=HANDOFF,
            baseline_travel_s=820.0, residual_p50_s=80.0, residual_p90_s=380.0, rho=0.35,
        )
        self.assertEqual(parts.breakdown.travel, TRAVEL)
        self.assertEqual(parts.breakdown.baseline_travel_s, 820.0)
        self.assertGreater(parts.tail_saving_s, 0.0)
        shares = share_of_total(parts)
        self.assertAlmostEqual(sum(shares.values()), 1.0)
        self.assertAlmostEqual(shares["travel"], 900.0 / 1590.0)

    def test_breakdown_serialises(self):
        parts = decompose(
            prep=QuantilePair.zero(), travel=TRAVEL, handoff=QuantilePair.zero(),
            baseline_travel_s=900.0, residual_p50_s=0.0, residual_p90_s=300.0, rho=0.35,
        )
        payload = parts.breakdown.to_dict()
        self.assertEqual(payload["travel"], {"p50": 900.0, "p90": 1200.0})
        self.assertEqual(payload["prep"], {"p50": 0.0, "p90": 0.0})


if __name__ == "__main__":
    unittest.main()
