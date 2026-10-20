"""popularity ranking and mean-rating baselines.
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_ratings_csv(path):
    """Return (user_id, movie_id, rating, timestamp) integer tuples."""
    if not path.is_file():
        raise FileNotFoundError("Missing split file: {}".format(path))
    records = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = {"user_id", "movie_id", "rating", "timestamp"}
        if reader.fieldnames is None or not expected.issubset(set(reader.fieldnames)):
            raise ValueError("Split CSV missing required columns: {}".format(path))
        for row in reader:
            records.append(tuple(int(row[key]) for key in
                                 ("user_id", "movie_id", "rating", "timestamp")))
    return records


class PopularityBaseline:
    """Global movie popularity, excluding a user's TRAIN-rated movies."""

    def __init__(self):
        self.counts = Counter()
        self.seen = defaultdict(set)
        self.rank_order = []
        self.known_users = set()

    def fit(self, train_rows):
        if not train_rows:
            raise ValueError("Cannot fit an empty training dataset")
        for user_id, movie_id, rating, timestamp in train_rows:
            self.counts[movie_id] += 1
            self.seen[user_id].add(movie_id)
            self.known_users.add(user_id)
        # Deterministic tie-breaking: lower movie ID comes first.
        self.rank_order = sorted(self.counts, key=lambda m: (-self.counts[m], m))
        return self

    def recommend(self, user_id, k=10):
        """Top k TRAIN-catalog movies not rated by user in TRAIN."""
        if k < 1:
            raise ValueError("k must be positive")
        seen = self.seen.get(user_id, set())
        result = []
        for movie_id in self.rank_order:
            if movie_id not in seen:
                result.append(movie_id)
                if len(result) == k:
                    break
        return result


def ranking_metrics(recommendations, relevant_movies, k=10):
    """Binary relevance DCG, normalized DCG, and recall for one user."""
    if not relevant_movies:
        raise ValueError("Ranking metrics require at least one relevant movie")
    dcg = sum(1.0 / math.log2(position + 2)
              for position, movie_id in enumerate(recommendations[:k])
              if movie_id in relevant_movies)
    ideal_length = min(len(relevant_movies), k)
    ideal_dcg = sum(1.0 / math.log2(position + 2)
                    for position in range(ideal_length))
    hits = sum(movie_id in relevant_movies for movie_id in recommendations[:k])
    return dcg / ideal_dcg, float(hits) / len(relevant_movies)


def evaluate_ranking(model, validation_rows, k=10, positive_threshold=4):
    """Macro-average on known users with at least one relevant known movie. """
    positives = defaultdict(set)
    counts = {"validation_ratings": len(validation_rows),
              "known_user_and_movie_ratings": 0,
              "ratings_with_unseen_user": 0,
              "ratings_with_unseen_movie": 0,
              "ratings_with_both_unseen": 0,
              "eligible_positive_ratings": 0}
    known_movies = set(model.counts)
    for user_id, movie_id, rating, timestamp in validation_rows:
        unseen_user = user_id not in model.known_users
        unseen_movie = movie_id not in known_movies
        if unseen_user:
            counts["ratings_with_unseen_user"] += 1
        if unseen_movie:
            counts["ratings_with_unseen_movie"] += 1
        if unseen_user and unseen_movie:
            counts["ratings_with_both_unseen"] += 1
        if unseen_user or unseen_movie:
            continue
        counts["known_user_and_movie_ratings"] += 1
        if rating >= positive_threshold and movie_id not in model.seen[user_id]:
            positives[user_id].add(movie_id)
            counts["eligible_positive_ratings"] += 1

    scores_ndcg, scores_recall = [], []
    for user_id, relevant in positives.items():
        recommended = model.recommend(user_id, k)
        ndcg, recall = ranking_metrics(recommended, relevant, k)
        scores_ndcg.append(ndcg)
        scores_recall.append(recall)
    n_users = len(scores_ndcg)
    counts["excluded_cold_start_union"] = (
        counts["ratings_with_unseen_user"] +
        counts["ratings_with_unseen_movie"] -
        counts["ratings_with_both_unseen"]
    )
    counts.update({
        "evaluated_users_with_positive": n_users,
        "relevance_definition": "validation rating >= {}".format(positive_threshold),
        "ndcg_at_{}".format(k): sum(scores_ndcg) / n_users if n_users else None,
        "recall_at_{}".format(k): sum(scores_recall) / n_users if n_users else None,
    })
    return counts


def evaluate_rating_mean(train_rows, validation_rows):
    """Movie-mean and global-mean RMSE on warm-start validation ratings."""
    if not train_rows:
        raise ValueError("Cannot evaluate an empty training dataset")
    totals = defaultdict(float)
    movie_counts = Counter()
    known_users = set()
    total_rating = 0.0
    for user_id, movie_id, rating, timestamp in train_rows:
        totals[movie_id] += rating
        movie_counts[movie_id] += 1
        known_users.add(user_id)
        total_rating += rating
    global_mean = total_rating / len(train_rows)
    errors_movie = errors_global = 0.0
    n = 0
    for user_id, movie_id, rating, timestamp in validation_rows:
        if user_id not in known_users or movie_id not in movie_counts:
            continue
        movie_mean = totals[movie_id] / movie_counts[movie_id]
        errors_movie += (rating - movie_mean) ** 2
        errors_global += (rating - global_mean) ** 2
        n += 1
    return {
        "evaluated_warm_start_ratings": n,
        "movie_mean_rmse": math.sqrt(errors_movie / n) if n else None,
        "global_mean_rmse": math.sqrt(errors_global / n) if n else None,
        "train_global_mean": round(global_mean, 6),
    }


def build_report(train_rows, validation_rows, k=10):
    model = PopularityBaseline().fit(train_rows)
    return {
        "experiment": "Day 3: popularity top-k + training movie-mean rating baseline",
        "fit_data": "data/splits/train.csv ONLY",
        "evaluation_data": "data/splits/validation.csv ONLY",
        "test_data_used": False,
        "candidate_movies": len(model.counts),
        "known_train_users": len(model.known_users),
        "ranking_method": "training movie rating count, descending; movie ID tie-break",
        "history_exclusion": "movies rated by this user in training only",
        "ranking_validation": evaluate_ranking(model, validation_rows, k=k),
        "rating_validation": evaluate_rating_mean(train_rows, validation_rows),
        "limitations": (
            "Offline held-out ratings are incomplete relevance labels, not dislikes. "
            "Results are for warm-start users and movies only. "
            "The global chronological split has substantial user cold-start."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "splits" / "train.csv")
    parser.add_argument("--validation", type=Path,
                        default=ROOT / "data" / "splits" / "validation.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports" / "day03_popularity.json")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")

    train_rows = load_ratings_csv(args.train)
    validation_rows = load_ratings_csv(args.validation)
    report = build_report(train_rows, validation_rows, k=args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("\nWrote {}".format(args.output))


if __name__ == "__main__":
    main()
