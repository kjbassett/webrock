import unittest
from webrock.schedule_utils import parse_cron_field, parse_month_field


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
        assert parse_month_field(raw) is None

    def test_parse_month_empty_string(self):
        raw = ""
        assert parse_month_field(raw) is None

    def test_parse_month_name(self):
        raw = "January"
        assert parse_month_field(raw) == [1]

        raw_short = "Aug"
        assert parse_month_field(raw_short) == [8]

    def test_parse_weird_capitalization(self):
        raw = "dEcEmBeR"
        assert parse_month_field(raw) == [12]

    def test_parse_month_number(self):
        raw = "8"
        assert parse_month_field(raw) == [8]

    def test_parse_month_comma_separated(self):
        raw = "Jan, 2, Mar, April"
        assert parse_month_field(raw) == [1, 2, 3, 4]

    def test_parse_month_range(self):
        num_raw = "1-3"
        assert parse_month_field(num_raw) == [1, 2, 3]
        name_raw = "jan-Mar"
        assert parse_month_field(name_raw) == [1, 2, 3]
        mixed_raw = "mar-5"
        assert parse_month_field(mixed_raw) == [3, 4, 5]

    def test_parse_mixed(self):
        raw = "1-3, 4, aug-9 , dec"
        print(parse_month_field(raw))
        assert parse_month_field(raw) == [1, 2, 3, 4, 8, 9, 12]
