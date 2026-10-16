"""Unit tests for time-safe split behavior (uses only synthetic ratings)."""

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.split_data import (
    build_report,
    chronological_split,
    heldout_coverage,
    load_ratings,
    write_csv,
)


class ChronologicalSplitTests(unittest.TestCase):
    def setUp(self):
        # Reversed order: sorting must be global, not per user.
        self.rows = [
            (12, 6, 5, 60), (11, 6, 4, 60),
            (11, 1, 5, 10), (12, 1, 2, 10),
            (11, 2, 4, 20), (12, 2, 5, 20),
            (11, 3, 5, 30), (12, 3, 4, 30),
            (11, 4, 4, 40), (12, 4, 5, 40),
            (11, 5, 3, 50), (12, 5, 4, 50),
        ]

    def test_ties_not_split_and_no_time_leakage(self):
        train, validation, test = chronological_split(self.rows)
        self.assertEqual(len(train) + len(validation) + len(test), len(self.rows))
        self.assertLess(max(r[3] for r in train), min(r[3] for r in validation))
        self.assertLess(max(r[3] for r in validation), min(r[3] for r in test))
        self.assertEqual(len(set(r[3] for r in train) & set(r[3] for r in validation)), 0)
        self.assertEqual(len(set(r[3] for r in validation) & set(r[3] for r in test)), 0)
        report = build_report(train, validation, test)
        self.assertTrue(report["strict_timestamp_order"])

    def test_split_requires_three_timestamp_groups(self):
        with self.assertRaises(ValueError):
            chronological_split([(1, 1, 4, 10), (1, 2, 5, 20)])

    def test_warm_start_coverage(self):
        history = [(1, 10, 4, 1), (2, 20, 5, 1)]
        heldout = [(1, 20, 5, 2), (3, 20, 4, 2), (1, 30, 4, 2)]
        stats = heldout_coverage(history, heldout)
        self.assertEqual(stats["known_user_and_movie_ratings"], 1)
        self.assertEqual(stats["ratings_with_unseen_user"], 1)
        self.assertEqual(stats["ratings_with_unseen_movie"], 1)

    def test_csv_output_and_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "ratings.dat"
            raw.write_text("1::2::4::10\n2::3::5::20\n", encoding="utf-8")
            self.assertEqual(load_ratings(raw), [(1, 2, 4, 10), (2, 3, 5, 20)])
            output = root / "splits" / "train.csv"
            write_csv(output, [(1, 2, 4, 10)])
            with output.open("r", encoding="utf-8", newline="") as handle:
                actual = list(csv.DictReader(handle))
            self.assertEqual(actual[0]["movie_id"], "2")


if __name__ == "__main__":
    unittest.main()
