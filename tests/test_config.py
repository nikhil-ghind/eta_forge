import os
import tempfile
import unittest

from eta_forge.config import Config, ConfigError, load_config


class TestDefaults(unittest.TestCase):
    def test_defaults_validate(self):
        cfg = Config().validate()
        self.assertEqual(cfg.database.port, 5432)
        self.assertEqual(cfg.model.quantiles, [0.5, 0.9])
        self.assertEqual(cfg.profiles.tod_buckets, 24)

    def test_to_dict_is_nested_and_plain(self):
        data = Config().to_dict()
        self.assertIsInstance(data["graph"]["default_speed_kph"], dict)
        self.assertEqual(data["service"]["model_alias"], "production")


class TestEnvLayer(unittest.TestCase):
    def test_double_underscore_form(self):
        cfg = load_config(
            env={
                "ETA_FORGE_DATABASE__HOST": "postgis.internal",
                "ETA_FORGE_DATABASE__MAX_POOL_SIZE": "32",
            }
        )
        self.assertEqual(cfg.database.host, "postgis.internal")
        self.assertEqual(cfg.database.max_pool_size, 32)

    def test_single_underscore_form_resolves_known_section(self):
        cfg = load_config(env={"ETA_FORGE_SERVICE_PORT": "9000"})
        self.assertEqual(cfg.service.port, 9000)

    def test_unknown_prefix_ignored(self):
        cfg = load_config(env={"ETA_FORGE_NOT_A_SECTION_AT_ALL": "1", "PATH": "/bin"})
        self.assertEqual(cfg.service.port, 8080)

    def test_booleans_and_json_lists(self):
        cfg = load_config(
            env={
                "ETA_FORGE_FEATURES__USE_WEATHER": "false",
                "ETA_FORGE_MODEL__QUANTILES": "[0.5, 0.8, 0.95]",
            }
        )
        self.assertFalse(cfg.features.use_weather)
        self.assertEqual(cfg.model.quantiles, [0.5, 0.8, 0.95])


class TestCliOverrides(unittest.TestCase):
    def test_override_applies(self):
        cfg = load_config(env={}, overrides=["profiles.tod_buckets=12"])
        self.assertEqual(cfg.profiles.tod_buckets, 12)

    def test_override_beats_env(self):
        cfg = load_config(
            env={"ETA_FORGE_SERVICE__PORT": "9000"},
            overrides=["service.port=9999"],
        )
        self.assertEqual(cfg.service.port, 9999)

    def test_malformed_override(self):
        with self.assertRaises(ConfigError):
            load_config(env={}, overrides=["service.port"])
        with self.assertRaises(ConfigError):
            load_config(env={}, overrides=["port=1"])

    def test_unknown_key_is_an_error(self):
        with self.assertRaises(ConfigError) as ctx:
            load_config(env={}, overrides=["service.porrt=1"])
        self.assertIn("porrt", str(ctx.exception))

    def test_type_mismatch_is_an_error(self):
        with self.assertRaises(ConfigError):
            load_config(env={}, overrides=["service.port=not-a-number"])


class TestYamlLayer(unittest.TestCase):
    def setUp(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML not installed")

    def test_file_then_env_then_cli(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(
                "database:\n  host: from-file\n  port: 6000\n"
                "service:\n  port: 7000\n"
            )
            path = handle.name
        try:
            cfg = load_config(
                path,
                env={"ETA_FORGE_DATABASE__PORT": "6100"},
                overrides=["service.port=7100"],
            )
            self.assertEqual(cfg.database.host, "from-file")
            self.assertEqual(cfg.database.port, 6100)
            self.assertEqual(cfg.service.port, 7100)
        finally:
            os.unlink(path)

    def test_shipped_config_loads(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, "configs", "eta_forge.yaml")
        cfg = load_config(path, env={})
        self.assertEqual(cfg.profiles.timezone, "America/New_York")
        self.assertEqual(cfg.data.bbox().min_lat, 40.4774)

    def test_missing_file(self):
        with self.assertRaises(ConfigError):
            load_config("/nonexistent/eta.yaml", env={})


class TestSectionValidation(unittest.TestCase):
    def test_tod_buckets_must_divide_24(self):
        with self.assertRaises(ConfigError):
            load_config(env={}, overrides=["profiles.tod_buckets=7"])

    def test_split_fractions_must_leave_a_test_set(self):
        with self.assertRaises(ConfigError):
            load_config(
                env={},
                overrides=["model.train_fraction=0.9", "model.validation_fraction=0.2"],
            )

    def test_quantiles_must_be_ascending(self):
        with self.assertRaises(ConfigError):
            load_config(env={}, overrides=["model.quantiles=[0.9, 0.5]"])

    def test_coverage_band_ordering(self):
        with self.assertRaises(ConfigError):
            load_config(env={}, overrides=["model.promote_coverage_band=[0.95, 0.8]"])

    def test_h3_resolutions_must_be_ordered(self):
        with self.assertRaises(ConfigError):
            load_config(env={}, overrides=["features.h3_resolution_coarse=10"])

    def test_pool_sizes(self):
        with self.assertRaises(ConfigError):
            load_config(
                env={},
                overrides=["database.min_pool_size=8", "database.max_pool_size=4"],
            )

    def test_mape_floor_must_be_positive(self):
        with self.assertRaises(ConfigError):
            load_config(env={}, overrides=["eval.mape_floor_s=0"])

    def test_tail_correlation_bounds(self):
        with self.assertRaises(ConfigError):
            load_config(env={}, overrides=["service.component_tail_correlation=1.5"])


if __name__ == "__main__":
    unittest.main()
