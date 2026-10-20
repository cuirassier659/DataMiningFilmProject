"""Graduate-level EDA for the MovieLens 1M recommender project.

Run from the repository root: python scripts/explore_data.py
Uses TRAIN and VALIDATION only. Never reads test.csv or test labels.
Output: reports/day04_eda.json plus optional reports/figures/*.png.
Compatible with Python 3.7+; no required third-party packages.
"""
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "splits" / "train.csv"
VAL_PATH = ROOT / "data" / "splits" / "validation.csv"
OUT_PATH = ROOT / "reports" / "day04_eda.json"
FIG_DIR = ROOT / "reports" / "figures"


def load_csv(path):
    if not path.is_file():
        raise FileNotFoundError("Missing {}. First restore MovieLens data and run python scripts/split_data.py".format(path))
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not {"user_id", "movie_id", "rating", "timestamp"}.issubset(reader.fieldnames or []):
            raise ValueError("Unexpected CSV columns in {}".format(path))
        return [(int(r["user_id"]), int(r["movie_id"]), int(r["rating"]), int(r["timestamp"])) for r in reader]


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    point = (len(values) - 1) * pct / 100.0
    lo = int(math.floor(point))
    hi = int(math.ceil(point))
    return round(values[lo] + (values[hi] - values[lo]) * (point - lo), 3)


def quantiles(values):
    return {"min": min(values) if values else None,
            "p25": percentile(values, 25), "median": percentile(values, 50),
            "p75": percentile(values, 75), "p90": percentile(values, 90),
            "p99": percentile(values, 99), "max": max(values) if values else None}


def rating_summary(rows):
    counts = Counter(r[2] for r in rows)
    total = len(rows)
    mean = sum(star * n for star, n in counts.items()) / total if total else None
    return {"ratings": total, "distribution": {str(k): counts[k] for k in range(1, 6)},
            "mean_rating": round(mean, 5) if mean is not None else None,
            "positive_4_or_5_share": round((counts[4] + counts[5]) / total, 5) if total else None}


def month(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m")


def concentration(counts, proportion):
    vals = sorted(counts.values(), reverse=True)
    top_n = max(1, math.ceil(len(vals) * proportion))
    return {"top_movies": top_n, "share_of_training_ratings": round(sum(vals[:top_n]) / sum(vals), 5)}


def date_iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def build_report(train, val):
    t_users, t_movies = set(), set()
    users = Counter(); movies = Counter()
    training_months = Counter()
    for u, m, r, ts in train:
        t_users.add(u); t_movies.add(m)
        users[u] += 1; movies[m] += 1
        training_months[month(ts)] += 1

    val_users = set(u for u, _, _, _ in val)
    coverage = Counter()
    warm = []; cold = []
    val_months = Counter()
    for u, m, r, ts in val:
        known_u, known_m = u in t_users, m in t_movies
        if known_u and known_m:
            coverage["warm"] += 1
            warm.append((u, m, r, ts))
        else:
            if not known_u and not known_m:
                coverage["new_user_and_movie"] += 1
            elif not known_u:
                coverage["new_user_only"] += 1
            else:
                coverage["new_movie_only"] += 1
            cold.append((u, m, r, ts))
        val_months[month(ts)] += 1

    warm_positive = [(u, m, r, ts) for u, m, r, ts in warm if r >= 4]
    eligible_users = set(u for u, _, _, _ in warm_positive)
    val_count = len(val)
    train_dates = sorted(r[3] for r in train)
    val_dates = sorted(r[3] for r in val)
    report = {
        "data_scope": "train.csv and validation.csv only; test.csv NOT read",
        "train": {
            "ratings": len(train), "users": len(t_users), "movies": len(t_movies),
            "density": round(len(train) / (len(t_users) * len(t_movies)), 6),
            "sparsity": round(1 - len(train) / (len(t_users) * len(t_movies)), 6),
            "rating_profile": rating_summary(train),
            "ratings_per_user_percentiles": quantiles(list(users.values())),
            "ratings_per_movie_percentiles": quantiles(list(movies.values())),
            "movies_with_at_most_5_ratings": sum(n <= 5 for n in movies.values()),
            "movies_with_at_most_20_ratings": sum(n <= 20 for n in movies.values()),
            "movie_rating_share_top_1pct": concentration(movies, 0.01),
            "movie_rating_share_top_10pct": concentration(movies, 0.10),
            "first_utc": date_iso(train_dates[0]), "last_utc": date_iso(train_dates[-1]),
            "ratings_by_month": dict(sorted(training_months.items())),
        },
        "validation": {
            "rating_profile_all": rating_summary(val),
            "rating_profile_warm_start_only": rating_summary(warm),
            "total_ratings": val_count,
            "total_users": len(val_users),
            "users_seen_in_train": len(val_users & t_users),
            "users_not_seen_in_train": len(val_users - t_users),
            "first_utc": date_iso(val_dates[0]), "last_utc": date_iso(val_dates[-1]),
            "ratings_by_month": dict(sorted(val_months.items())),
            "cold_start_rating_counts": dict(sorted(coverage.items())),
            "warm_rating_share": round(coverage["warm"] / val_count, 5),
            "cold_start_rating_share": round(len(cold) / val_count, 5),
            "warm_positive_ratings_4_or_5": len(warm_positive),
            "warm_users_with_positive": len(eligible_users),
            "train_history_percentiles_for_warm_positive_users": quantiles([users[u] for u in eligible_users]),
        },
        "methodological_notes": [
            "Interpret ranking scores only for users with a warm-start positive in validation.",
            "Global chronological splitting creates large new-user cold-start; report exclusions.",
            "Train/validation rating distribution differences may reflect changing user cohorts.",
            "An unobserved rating does not establish dislike or even exposure to the movie.",
            "Do not use test ratings for development, EDA-based choices, or model tuning.",
        ],
    }
    return report, users, movies, coverage, train, warm


def make_figures(users, movies, coverage, train, warm):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; JSON written but charts skipped")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    popularity = sorted(movies.values(), reverse=True)
    plt.figure(figsize=(6.4, 3.8))
    plt.plot(range(1, len(popularity) + 1), popularity)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Movie popularity rank (log scale)")
    plt.ylabel("Number of training ratings (log scale)")
    plt.title("Training movie popularity: long-tail exposure")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "movie_popularity_long_tail.png", dpi=180)
    plt.close()

    names = ["Warm-start", "New user only", "New movie only", "Both new"]
    vals = [coverage["warm"], coverage["new_user_only"], coverage["new_movie_only"], coverage["new_user_and_movie"]]
    plt.figure(figsize=(7.3, 3.8))
    bars = plt.bar(names, vals)
    plt.ylabel("Validation ratings")
    plt.title("Cold-start coverage at the validation cutoff")
    plt.tick_params(axis="x", labelrotation=12)
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width()/2, val, str(val), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "validation_cold_start.png", dpi=180)
    plt.close()

    train_counts = Counter(r[2] for r in train)
    warm_counts = Counter(r[2] for r in warm)
    total_train, total_warm = len(train), len(warm)
    x = [1, 2, 3, 4, 5]
    plt.figure(figsize=(6.4, 3.8))
    plt.bar([v - 0.18 for v in x], [train_counts[v]/total_train for v in x], width=0.36, label="Train")
    plt.bar([v + 0.18 for v in x], [warm_counts[v]/total_warm for v in x], width=0.36, label="Warm validation")
    plt.xticks(x)
    plt.xlabel("Rating (stars)")
    plt.ylabel("Share of ratings")
    plt.title("Rating distribution and warm-start cohort shift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "train_vs_warm_validation_rating.png", dpi=180)
    plt.close()

    print("Saved figures to {}".format(FIG_DIR))


def main():
    train = load_csv(TRAIN_PATH)
    validation = load_csv(VAL_PATH)
    report, users, movies, coverage, tr, warm = build_report(train, validation)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("EDA saved to {}".format(OUT_PATH))
    print("Train: {} ratings, {} users, {} movies".format(len(train), len(users), len(movies)))
    print("Movie top 10% share: {:.2%}".format(report["train"]["movie_rating_share_top_10pct"]["share_of_training_ratings"]))
    print("Validation warm: {} ({:.2%}); cold: {} ({:.2%})".format(
        coverage["warm"], report["validation"]["warm_rating_share"], len(validation) - coverage["warm"], report["validation"]["cold_start_rating_share"]))
    print("Warm positive users: {}".format(report["validation"]["warm_users_with_positive"]))
    make_figures(users, movies, coverage, tr, warm)


if __name__ == "__main__":
    main()
