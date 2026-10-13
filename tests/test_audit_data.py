"""Small local tests, independent of the MovieLens download."""

import tempfile
import unittest
from pathlib import Path

from scripts.audit_data import audit_ratings


class AuditDataTests(unittest.TestCase):
    def test_counts_duplicates_invalid_rows_and_sparsity(self):
        example = (
            "1::10::5::100\n"
            "1::20::4::101\n"
            "2::10::3::102\n"
            "2::10::2::103\n"  # Duplicate user/movie pair
            "not a movie rating\n"  # Invalid format
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ratings.dat"
            path.write_text(example, encoding="utf-8")
            result = audit_ratings(path)

        self.assertEqual(result["rows_read"], 5)
        self.assertEqual(result["valid_distinct_ratings"], 3)
        self.assertEqual(result["duplicate_user_movie_pairs"], 1)
        self.assertEqual(result["invalid_rows"], 1)
        self.assertEqual(result["distinct_users"], 2)
        self.assertEqual(result["distinct_rated_movies"], 2)
        self.assertEqual(result["observed_matrix_density"], 0.75)
        self.assertEqual(result["rating_distribution"]["5"], 1)

    def test_rejects_out_of_range_ratings(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ratings.dat"
            path.write_text("1::10::6::100\n1::10::5::101\n", encoding="utf-8")
            result = audit_ratings(path)
        self.assertEqual(result["invalid_rows"], 1)
        self.assertEqual(result["valid_distinct_ratings"], 1)


if __name__ == "__main__":
    unittest.main()
