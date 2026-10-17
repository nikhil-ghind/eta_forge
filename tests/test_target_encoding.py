import unittest

from eta_forge.features.h3_cells import CellVocabulary, cell_for, fallback_cell
from eta_forge.features.target_encoding import TargetEncoder
from eta_forge.types import Location


class TestTargetEncoder(unittest.TestCase):
    def test_smoothed_mean(self):
        enc = TargetEncoder(smoothing=2.0).fit(["a", "a", "b"], [10.0, 20.0, 60.0])
        self.assertAlmostEqual(enc.global_mean, 30.0)
        # (30 + 2 * 30) / (2 + 2)
        self.assertAlmostEqual(enc.transform_one("a"), 22.5)
        # (60 + 2 * 30) / (1 + 2)
        self.assertAlmostEqual(enc.transform_one("b"), 40.0)

    def test_unseen_and_missing_keys_get_the_global_mean(self):
        enc = TargetEncoder(smoothing=5.0).fit(["a"], [12.0])
        self.assertEqual(enc.transform_one("zzz"), 12.0)
        self.assertEqual(enc.transform_one(None), 12.0)

    def test_large_counts_converge_on_the_key_mean(self):
        enc = TargetEncoder(smoothing=10.0).fit(["a"] * 10_000 + ["b"], [5.0] * 10_000 + [500.0])
        self.assertAlmostEqual(enc.transform_one("a"), 5.0, delta=0.1)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            TargetEncoder(smoothing=0.0)
        with self.assertRaises(ValueError):
            TargetEncoder().fit(["a"], [1.0, 2.0])

    def test_past_mode_uses_only_earlier_folds(self):
        keys = ["a", "a", "a", "a"]
        targets = [10.0, 10.0, 100.0, 100.0]
        enc = TargetEncoder(smoothing=1.0)
        encoded = enc.fit_transform_oof(keys, targets, [[0, 1], [2, 3]], mode="past")
        # Fold 0 has no history: both rows get fold 0's mean and no per-key signal.
        self.assertEqual(encoded[0], encoded[1])
        self.assertAlmostEqual(encoded[0], 10.0)
        # Fold 1 sees only fold 0: (20 + 1 * 10) / (2 + 1) = 10, never the 100s.
        self.assertAlmostEqual(encoded[2], 10.0)
        self.assertAlmostEqual(encoded[3], 10.0)

    def test_kfold_mode_never_sees_the_rows_own_target(self):
        keys = ["a", "a", "b", "b"]
        folds = [[0], [1], [2], [3]]
        base = TargetEncoder(smoothing=3.0).fit_transform_oof(keys, [1.0, 2.0, 3.0, 4.0], folds, mode="kfold")
        changed = TargetEncoder(smoothing=3.0).fit_transform_oof(
            keys, [999.0, 2.0, 3.0, 4.0], folds, mode="kfold"
        )
        self.assertEqual(base[0], changed[0])

    def test_oof_still_fits_the_full_encoder(self):
        enc = TargetEncoder(smoothing=1.0)
        enc.fit_transform_oof(["a", "b"], [1.0, 3.0], [[0], [1]])
        self.assertEqual(enc.count("a"), 1)
        self.assertAlmostEqual(enc.global_mean, 2.0)

    def test_round_trip(self):
        enc = TargetEncoder(smoothing=4.0).fit(["x", "y", "x"], [1.0, 2.0, 3.0])
        restored = TargetEncoder.from_dict(enc.to_dict())
        self.assertEqual(restored.transform(["x", "y", "q"]), enc.transform(["x", "y", "q"]))

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            TargetEncoder().fit_transform_oof(["a"], [1.0], [[0]], mode="random")


class TestCells(unittest.TestCase):
    def test_fallback_cells_are_deterministic_and_local(self):
        a = Location(40.7580, -73.9855)
        b = Location(40.7581, -73.9856)    # ~14 m away
        far = Location(40.6413, -73.7781)  # JFK
        self.assertEqual(fallback_cell(a, 8), fallback_cell(a, 8))
        self.assertEqual(fallback_cell(a, 8), fallback_cell(b, 8))
        self.assertNotEqual(fallback_cell(a, 8), fallback_cell(far, 8))
        self.assertTrue(fallback_cell(a, 8).startswith("g8:"))

    def test_finer_resolution_splits_cells(self):
        points = [Location(40.75 + i * 0.001, -73.98) for i in range(20)]
        coarse = {cell_for(p, 7, scheme="grid") for p in points}
        fine = {cell_for(p, 9, scheme="grid") for p in points}
        self.assertGreater(len(fine), len(coarse))

    def test_vocabulary_floor_and_round_trip(self):
        vocab = CellVocabulary(min_count=2).fit(["a", "a", "b", "c", "c", "c"])
        self.assertEqual(vocab.encode("b"), 0)
        self.assertGreater(vocab.encode("a"), 0)
        self.assertNotEqual(vocab.encode("a"), vocab.encode("c"))
        self.assertEqual(vocab.encode(None), 0)
        restored = CellVocabulary.from_dict(vocab.to_dict())
        self.assertEqual(restored.encode("c"), vocab.encode("c"))


if __name__ == "__main__":
    unittest.main()
