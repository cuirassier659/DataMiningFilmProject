"""Small deterministic tests for Day 3"""

import math
import tempfile
import unittest
from pathlib import Path

from scripts.popularity_baseline import (
    PopularityBaseline,
    build_report,
    evaluate_ranking,
    evaluate_rating_mean,
    load_ratings_csv,
    ranking_metrics,
)


class PopularityBaselineTests(unittest.TestCase):
    def setUp(self):
        # Movies 10 and 20 each have 2 training ratings, 30 has one.
        self.train = [
            (1, 10, 5, 1), (1, 20, 2, 1),
            (2, 10, 4, 1), (2, 30, 4, 1),
            (3, 20, 5, 1),
        ]

    def test_train_counts_ties_and_seen_movies(self):
        model = PopularityBaseline().fit(self.train)
        self.assertEqual(model.rank_order, [10, 20, 30])
        self.assertEqual(model.recommend(1, 10), [30])
        self.assertEqual(model.recommend(2, 10), [20])

    def test_no_heldout_label_leakage(self):
        validation = [(2, 20, 5, 2), (2, 999, 5, 2), (99, 20, 5, 2)]
        model = PopularityBaseline().fit(self.train)
        result = evaluate_ranking(model, validation)
        self.assertEqual(result["known_user_and_movie_ratings"], 1)
        self.assertEqual(result["ratings_with_unseen_user"], 1)
        self.assertEqual(result["ratings_with_unseen_movie"], 1)
        self.assertEqual(result["eligible_positive_ratings"], 1)
        self.assertEqual(result["evaluated_users_with_positive"], 1)
        self.assertEqual(result["ndcg_at_10"], 1.0)
        self.assertEqual(result["recall_at_10"], 1.0)
        self.assertEqual(model.counts[20], 2)
        self.assertEqual(model.recommend(2, 10), [20])

    def test_binary_ndcg_places_early_hits_higher(self):
        ndcg, recall = ranking_metrics([20, 30, 40], {20, 40})
        expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(ndcg, expected)
        self.assertEqual(recall, 1.0)
        later_ndcg, _ = ranking_metrics([30, 20, 40], {20, 40})
        self.assertLess(later_ndcg, ndcg)

    def test_only_eligible_positive_users(self):
        validation = [(2, 20, 3, 2), (99, 20, 5, 2)]
        result = evaluate_ranking(PopularityBaseline().fit(self.train), validation)
        self.assertEqual(result["evaluated_users_with_positive"], 0)
        self.assertIsNone(result["ndcg_at_10"])

    def test_movie_means_use_training_only(self):
        validation = [(2, 20, 5, 2), (99, 20, 1, 2), (2, 999, 1, 2)]
        result = evaluate_rating_mean(self.train, validation)
        self.assertEqual(result["evaluated_warm_start_ratings"], 1)
        self.assertAlmostEqual(result["movie_mean_rmse"], 1.5)
        self.assertAlmostEqual(result["global_mean_rmse"], 1.0)

    def test_csv_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "train.csv"
            csv_path.write_text("user_id,movie_id,rating,timestamp\n1,10,4,5\n",
                                encoding="utf-8")
            self.assertEqual(load_ratings_csv(csv_path), [(1, 10, 4, 5)])
        report = build_report(self.train, [(2, 20, 5, 2)])
        self.assertFalse(report["test_data_used"])
        self.assertEqual(report["candidate_movies"], 3)


if __name__ == "__main__":
    unittest.main()
