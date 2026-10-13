"""Audit MovieLens 1M rating validity, duplicates, activity, and sparsity."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def audit_ratings(path: Path) -> dict:
    """Read ratings.dat and return aggregate statistics (no raw user-level exports)."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing ratings file: {path}")

    user_counts: Counter[int] = Counter()
    movie_counts: Counter[int] = Counter()
    seen_pairs: set[tuple[int, int]] = set()
    rows_read = invalid_rows = duplicate_pairs = 0
    first_timestamp: int | None = None
    last_timestamp: int | None = None
    rating_counts: Counter[int] = Counter()

    with path.open("r", encoding="latin-1") as stream:
        for line in stream:
            rows_read += 1
            fields = line.strip().split("::")
            if len(fields) != 4:
                invalid_rows += 1
                continue
            try:
                user, movie, rating, timestamp = map(int, fields)
            except ValueError:
                invalid_rows += 1
                continue
            if user <= 0 or movie <= 0 or rating not in range(1, 6) or timestamp < 0:
                invalid_rows += 1
                continue

            pair = (user, movie)
            if pair in seen_pairs:
                duplicate_pairs += 1
                continue
            seen_pairs.add(pair)
            user_counts[user] += 1
            movie_counts[movie] += 1
            rating_counts[rating] += 1
            first_timestamp = timestamp if first_timestamp is None else min(first_timestamp, timestamp)
            last_timestamp = timestamp if last_timestamp is None else max(last_timestamp, timestamp)

    n_users, n_movies = len(user_counts), len(movie_counts)
    n_valid = len(seen_pairs)
    density = n_valid / (n_users * n_movies) if n_users and n_movies else 0.0

    def activity(counts: Counter[int]) -> dict:
        vals = list(counts.values())
        return {
            "min": min(vals) if vals else 0,
            "median": statistics.median(vals) if vals else 0,
            "max": max(vals) if vals else 0,
        }

    def utc(ts: int | None) -> str | None:
        return datetime.fromtimestamp(ts, timezone.utc).isoformat() if ts is not None else None

    return {
        "dataset": "MovieLens 1M",
        "source": "https://grouplens.org/datasets/movielens/1m/",
        "rows_read": rows_read,
        "valid_distinct_ratings": n_valid,
        "invalid_rows": invalid_rows,
        "duplicate_user_movie_pairs": duplicate_pairs,
        "distinct_users": n_users,
        "distinct_rated_movies": n_movies,
        "observed_matrix_density": round(density, 6),
        "observed_matrix_sparsity": round(1 - density, 6) if n_users and n_movies else None,
        "rating_distribution": {str(k): rating_counts[k] for k in range(1, 6)},
        "ratings_per_user": activity(user_counts),
        "ratings_per_movie": activity(movie_counts),
        "earliest_rating_utc": utc(first_timestamp),
        "latest_rating_utc": utc(last_timestamp),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=ROOT / "data" / "ml-1m" / "ratings.dat")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "day01_audit.json")
    args = parser.parse_args()

    report = audit_ratings(args.ratings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
