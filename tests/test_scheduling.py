import unittest
from webrock.schedule_utils import parse_cron_field, parse_named_field, calculate_next_run, MONTH_LOOKUP, DOW_LOOKUP
from datetime import datetime


class TestParseCronField(unittest.TestCase):

    def test_parse_star(self):
        raw = "*"
        assert parse_cron_field(raw) is None

    def test_parse_empty_string(self):
        raw = ""
        assert parse_cron_field(raw) is None

    def test_parse_single_value(self):
        raw = "6"
        assert parse_cron_field(raw) == [6]

    def test_parse_comma_separated_value(self):
        raw = "1,2,3"
        assert parse_cron_field(raw) == [1, 2, 3]

    def test_parse_range(self):
        raw = "5-7"
        assert parse_cron_field(raw) == [5, 6, 7]

    def test_parse_mixed(self):
        raw = "59, 47-49"
        assert parse_cron_field(raw) == [47, 48, 49, 59]


class TestParseMonthField(unittest.TestCase):

    def test_parse_month_star(self):
        raw = "*"
        assert parse_named_field(raw, MONTH_LOOKUP) is None

    def test_parse_month_empty_string(self):
        raw = ""
        assert parse_named_field(raw, MONTH_LOOKUP) is None

    def test_parse_month_name(self):
        raw = "January"
        self.assertEqual(parse_named_field(raw, MONTH_LOOKUP), [1])

        raw_short = "Aug"
        self.assertEqual(parse_named_field(raw_short, MONTH_LOOKUP), [8])

    def test_parse_weird_capitalization(self):
        raw = "dEcEmBeR"
        assert parse_named_field(raw, MONTH_LOOKUP) == [12]

    def test_parse_month_number(self):
        raw = "8"
        assert parse_named_field(raw, MONTH_LOOKUP) == [8]

    def test_parse_month_comma_separated(self):
        raw = "Jan, 2, Mar, April"
        assert parse_named_field(raw, MONTH_LOOKUP) == [1, 2, 3, 4]

    def test_parse_month_range(self):
        num_raw = "1-3"
        assert parse_named_field(num_raw, MONTH_LOOKUP) == [1, 2, 3]
        name_raw = "jan-Mar"
        assert parse_named_field(name_raw, MONTH_LOOKUP) == [1, 2, 3]
        mixed_raw = "mar-5"
        assert parse_named_field(mixed_raw, MONTH_LOOKUP) == [3, 4, 5]

    def test_parse_mixed(self):
        raw = "1-3, 4, aug-9 , dec"
        assert parse_named_field(raw, MONTH_LOOKUP) == [1, 2, 3, 4, 8, 9, 12]


class TestCalculateNextRun(unittest.TestCase):

    def dt(self, y, m, d, H, M):
        """Helper: produce a UTC timestamp integer."""
        return int(datetime(y, m, d, H, M).timestamp())

    # ---------------------------
    # ONCE
    # ---------------------------
    def test_once_returns_timestamp(self):
        ts = datetime(2025, 1, 10, 12, 30)
        ts_string = "2025-01-10 12:30:00"
        schedule = {
            "type": "once",
            "timestamp": ts_string
        }
        self.assertEqual(calculate_next_run(schedule), int(ts.timestamp()))

    # ---------------------------
    # INTERVAL
    # ---------------------------
    def test_interval_adds_seconds(self):
        last = self.dt(2025, 1, 1, 10, 0)
        schedule = {
            "type": "interval",
            "last_run": last,
            "seconds": 60
        }
        next_run = calculate_next_run(schedule)
        self.assertEqual(next_run, last + 60)

    # ---------------------------
    # CRON – minutes & hours
    # ---------------------------
    def test_cron_simple_match_next_minute(self):
        # Last run was 10:05, cron = every minute
        last = self.dt(2025, 1, 1, 10, 5)
        schedule = {
            "type": "cron",
            "last_run": last,
            "minutes": "*",
            "hours": "*",
            "days_of_month": "*",
            "days_of_week": "*",
            "months": "*"
        }
        next_run = calculate_next_run(schedule)
        # Should be 10:06
        self.assertEqual(next_run, self.dt(2025, 1, 1, 10, 6))

    def test_cron_specific_minute_in_same_hour(self):
        # Cron: minute=15, any hour
        last = self.dt(2025, 1, 1, 10, 10)
        schedule = {
            "type": "cron",
            "last_run": last,
            "minutes": "15",
            "hours": "*",
            "days_of_month": "*",
            "days_of_week": "*",
            "months": "*"
        }
        # Next 15th minute is 10:15
        self.assertEqual(
            calculate_next_run(schedule),
            self.dt(2025, 1, 1, 10, 15)
        )

    def test_cron_next_hour_when_minute_passed(self):
        # Cron: minute=5
        last = self.dt(2025, 1, 1, 10, 10)
        schedule = {
            "type": "cron",
            "last_run": last,
            "minutes": "5",
            "hours": "*",
            "days_of_month": "*",
            "days_of_week": "*",
            "months": "*"
        }
        # Next matching time is 11:05
        self.assertEqual(
            calculate_next_run(schedule),
            self.dt(2025, 1, 1, 11, 5)
        )

    # ---------------------------
    # CRON – hours & ranges
    # ---------------------------
    def test_cron_hour_range(self):
        # Cron: hours=1-3, minute=0
        last = self.dt(2025, 1, 1, 0, 30)
        schedule = {
            "type": "cron",
            "last_run": last,
            "minutes": "0",
            "hours": "1-3",
            "days_of_month": "*",
            "days_of_week": "*",
            "months": "*"
        }
        # Next time is Jan 1 01:00
        next_run = calculate_next_run(schedule)
        self.assertEqual(
            next_run,
            self.dt(2025, 1, 1, 1, 0)
        )

        schedule["last_run"] = next_run
        next_run = calculate_next_run(schedule)
        self.assertEqual(
            next_run,
            self.dt(2025, 1, 1, 2, 0)
        )
    # ---------------------------
    # CRON – Day-of-week / Day-of-month AND logic
    # ---------------------------
    def test_cron_dom_or_dow(self):
        # Cron: run when day of month=5 AND dow=Monday
        # Last run is Feb 3, 2025 (Monday)
        last = self.dt(2025, 2, 3, 10, 0)  # Feb 3 2025 is Monday
        schedule = {
            "type": "cron",
            "last_run": last,
            "minutes": "0",
            "hours": "12",
            "days_of_month": "5",
            "days_of_week": "1",  # Monday = 0, Tuesday = 1
            "months": "*"
        }

        # Next Tuesday at 12:00 is Feb 4, 2025
        expected = self.dt(2025, 8, 5, 12, 0)
        self.assertEqual(calculate_next_run(schedule), expected)

    # ---------------------------
    # Months – numbers and names
    # ---------------------------
    def test_cron_named_month(self):
        # Cron: month=Feb, hour=0, minute=0
        last = self.dt(2025, 1, 31, 23, 0)
        schedule = {
            "type": "cron",
            "last_run": last,
            "minutes": "0",
            "hours": "0",
            "days_of_month": "1",
            "days_of_week": "*",
            "months": "feb"
        }
        # Next run is Feb 1 2025 00:00
        self.assertEqual(
            calculate_next_run(schedule),
            self.dt(2025, 2, 1, 0, 0)
        )

    def test_cron_month_range(self):
        # Cron: month=Mar-Apr, dom=1, 00:00
        last = self.dt(2025, 1, 1, 0, 0)
        schedule = {
            "type": "cron",
            "last_run": last,
            "minutes": "0",
            "hours": "0",
            "days_of_month": "1",
            "days_of_week": "*",
            "months": "mar-apr"
        }
        # Next run should be March 1, 2025
        self.assertEqual(
            calculate_next_run(schedule),
            self.dt(2025, 3, 1, 0, 0)
        )

    # ---------------------------
    # Lists & combined fields
    # ---------------------------
    def test_cron_list_minutes(self):
        # Cron: minutes = 10,20,30; hour=12
        last = self.dt(2025, 1, 1, 11, 59)
        schedule = {
            "type": "cron",
            "last_run": last,
            "minutes": "10,20,30",
            "hours": "12",
            "days_of_month": "*",
            "days_of_week": "*",
            "months": "*"
        }
        self.assertEqual(
            calculate_next_run(schedule),
            self.dt(2025, 1, 1, 12, 10)
        )

    # ---------------------------
    # Safety-stop
    # ---------------------------
    def test_cron_impossible_schedule(self):
        # Cron: Feb 30 is never valid
        last = self.dt(2025, 2, 1, 0, 0)
        schedule = {
            "type": "cron",
            "last_run": last,
            "minutes": "0",
            "hours": "0",
            "days_of_month": "30",
            "days_of_week": "8",
            "months": "feb"
        }

        with self.assertRaises(RuntimeError):
            calculate_next_run(schedule)
