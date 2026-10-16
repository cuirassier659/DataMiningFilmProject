"""Create global chronological MovieLens splits without breaking timestamp ties.

Usage from the project root:
    python scripts/split_data.py

Split files are deliberately written under data/ (ignored by Git). The aggregate
report is written under reports/ and is safe to commit.
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_COLUMNS = ("user_id", "movie_id", "rating", "timestamp")


def load_ratings(path):
    """Read MovieLens 1M's ::-delimited ratings file as integer tuples."""
    if not path.is_file():
        raise FileNotFoundError("Missing ratings file: {}".format(path))
    rows = []
    with path.open("r", encoding="latin-1") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.rstrip("\r\n").split("::")
            if len(parts) != 4:
                raise ValueError("Invalid row {}: expected 4 fields".format(line_number))
            try:
                user, movie, rating, timestamp = map(int, parts)
            except ValueError:
                raise ValueError("Invalid row {}: non-integer field".format(line_number))
            if user < 1 or movie < 1 or rating not in (1, 2, 3, 4, 5) or timestamp < 0:
                raise ValueError("Invalid row {}: out-of-range value".format(line_number))
            rows.append((user, movie, rating, timestamp))
    return rows


def chronological_split(rows, train_fraction=0.70, validation_fraction=0.15):
    """Split by global timestamp, keeping each equal-timestamp group together.

    All entries in a timestamp group belong to exactly one split. We pick the
    closest feasible group boundaries to the desired fractions, so actual
    proportions can differ slightly from 70/15/15. This is intentional.
    """
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Train/validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Leave a positive fraction for the test split")

    ordered = sorted(rows, key=lambda r: (r[3], r[0], r[1]))
    n = len(ordered)
    boundaries = [i for i in range(1, n) if ordered[i - 1][3] != ordered[i][3]]
    if len(boundaries) < 2:
        raise ValueError("Need at least three distinct timestamps for three splits")

    # The first cutoff must leave room for a later, distinct cutoff.
    target_train = n * train_fraction
    train_end = min(boundaries[:-1], key=lambda i: (abs(i - target_train), i))
    target_pretest = n * (train_fraction + validation_fraction)
    validation_end = min(
        (i for i in boundaries if i > train_end),
        key=lambda i: (abs(i - target_pretest), i),
    )

    train = ordered[:train_end]
    validation = ordered[train_end:validation_end]
    test = ordered[validation_end:]
    assert train and validation and test
    assert train[-1][3] < validation[0][3]
    assert validation[-1][3] < test[0][3]
    return train, validation, test


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)


def iso_utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def split_statistics(rows, total):
    return {
        "ratings": len(rows),
        "fraction": round(len(rows) / total, 6),
        "users": len(set(r[0] for r in rows)),
        "movies": len(set(r[1] for r in rows)),
        "first_timestamp_utc": iso_utc(rows[0][3]),
        "last_timestamp_utc": iso_utc(rows[-1][3]),
    }


def heldout_coverage(history, heldout):
    """Count held-out ratings supported by users/movies seen in fitting data."""
    known_users = set(row[0] for row in history)
    known_movies = set(row[1] for row in history)
    return {
        "total_ratings": len(heldout),
        "known_user_and_movie_ratings": sum(
            row[0] in known_users and row[1] in known_movies for row in heldout
        ),
        "ratings_with_unseen_user": sum(row[0] not in known_users for row in heldout),
        "ratings_with_unseen_movie": sum(row[1] not in known_movies for row in heldout),
    }


def build_report(train, validation, test):
    n = len(train) + len(validation) + len(test)
    return {
        "method": "global timestamp ordering; identical timestamps never cross splits",
        "requested_fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "total_ratings": n,
        "train": split_statistics(train, n),
        "validation": split_statistics(validation, n),
        "test": split_statistics(test, n),
        "strict_timestamp_order": (
            train[-1][3] < validation[0][3] < test[0][3]
            and validation[-1][3] < test[0][3]
        ),
        "validation_coverage_from_train": heldout_coverage(train, validation),
        "test_coverage_from_train_plus_validation": heldout_coverage(train + validation, test),
        "evaluation_protocol": (
            "Tune on train->validation only; freeze choices; refit on train+validation; "
            "evaluate once on test. Exclude new-user/new-movie events from the "
            "main warm-start comparison, but report their counts."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=ROOT / "data" / "ml-1m" / "ratings.dat")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "day02_split_summary.json")
    args = parser.parse_args()

    train, validation, test = chronological_split(load_ratings(args.ratings))
    for name, rows in (("train", train), ("validation", validation), ("test", test)):
        write_csv(args.out_dir / (name + ".csv"), rows)

    report = build_report(train, validation, test)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("\nWrote split CSV files to {} and report to {}".format(args.out_dir, args.report))


if __name__ == "__main__":
    main()
