import unittest
from datetime import date, datetime, timezone

from eta_forge.features.calendar import (
    HolidayCalendar,
    calendar_features,
    cyclical,
    us_federal_holidays,
)
from eta_forge.types import ValidationError

UTC = timezone.utc
BUILTIN = HolidayCalendar("US", use_package=False)


class TestHolidays(unittest.TestCase):
    def test_floating_holidays_2024(self):
        days = us_federal_holidays(2024)
        for expected in (
            date(2024, 1, 15),   # MLK: third Monday of January
            date(2024, 2, 19),   # Presidents' Day
            date(2024, 5, 27),   # Memorial Day: last Monday of May
            date(2024, 9, 2),    # Labor Day
            date(2024, 10, 14),  # Columbus Day
            date(2024, 11, 28),  # Thanksgiving: fourth Thursday
        ):
            self.assertIn(expected, days)

    def test_fixed_holidays_and_observance(self):
        self.assertIn(date(2024, 7, 4), us_federal_holidays(2024))
        # 4 July 2026 is a Saturday: observed Friday the 3rd.
        self.assertIn(date(2026, 7, 3), us_federal_holidays(2026))
        # 25 December 2022 is a Sunday: observed Monday the 26th.
        self.assertIn(date(2022, 12, 26), us_federal_holidays(2022))

    def test_new_year_observed_in_the_previous_year(self):
        # 1 January 2022 was a Saturday: observed Friday 31 December 2021.
        self.assertTrue(BUILTIN.is_holiday(date(2021, 12, 31)))

    def test_juneteenth_only_from_2021(self):
        self.assertIn(date(2021, 6, 18), us_federal_holidays(2021))  # Saturday -> Friday
        self.assertNotIn(date(2019, 6, 19), us_federal_holidays(2019))

    def test_ordinary_day(self):
        self.assertFalse(BUILTIN.is_holiday(date(2024, 3, 5)))
        self.assertEqual(BUILTIN.source, "builtin")


class TestCalendarFeatures(unittest.TestCase):
    def test_midnight_neighbours_are_close_on_the_circle(self):
        before = cyclical(23.98, 24.0)
        after = cyclical(0.02, 24.0)
        self.assertAlmostEqual(before[0], after[0] * -1, places=2)
        self.assertAlmostEqual(before[1], after[1], places=3)

    def test_weekday_morning_rush_in_local_time(self):
        # 13:00 UTC on Tuesday 5 March 2024 is 08:00 in New York.
        f = calendar_features(datetime(2024, 3, 5, 13, 0, tzinfo=UTC), holidays=BUILTIN)
        self.assertEqual(f["local_hour"], 8.0)
        self.assertEqual(f["dow"], 1.0)
        self.assertEqual(f["is_rush_hour"], 1.0)
        self.assertEqual(f["is_weekend"], 0.0)

    def test_weekend_has_no_rush_hour(self):
        f = calendar_features(datetime(2024, 3, 9, 13, 0, tzinfo=UTC), holidays=BUILTIN)
        self.assertEqual(f["is_weekend"], 1.0)
        self.assertEqual(f["is_rush_hour"], 0.0)

    def test_holiday_weekday_has_no_rush_hour(self):
        # 08:00 EDT on Thursday 4 July 2024.
        f = calendar_features(datetime(2024, 7, 4, 12, 0, tzinfo=UTC), holidays=BUILTIN)
        self.assertEqual(f["is_holiday"], 1.0)
        self.assertEqual(f["is_rush_hour"], 0.0)

    def test_hour_encoding_is_on_the_unit_circle(self):
        f = calendar_features(datetime(2024, 3, 5, 18, 45, tzinfo=UTC), holidays=BUILTIN)
        self.assertAlmostEqual(f["hour_sin"] ** 2 + f["hour_cos"] ** 2, 1.0)

    def test_naive_timestamp_rejected(self):
        with self.assertRaises(ValidationError):
            calendar_features(datetime(2024, 3, 5, 8, 0))


if __name__ == "__main__":
    unittest.main()
