"""Download the official MovieLens 1M data locally (never commit raw data)."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
FILES = ("ratings.dat", "movies.dat", "users.dat", "README")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace existing files")
    args = parser.parse_args()

    target = ROOT / "data" / "ml-1m"
    target.mkdir(parents=True, exist_ok=True)
    ratings = target / "ratings.dat"
    if ratings.exists() and not args.force:
        print(f"Dataset already present: {ratings}")
        return

    archive_path = ROOT / "data" / "ml-1m.zip"
    print(f"Downloading MovieLens 1M from {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "movie-recommender-research/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=90) as response, archive_path.open("wb") as dst:
            shutil.copyfileobj(response, dst)
        with zipfile.ZipFile(archive_path) as archive:
            for filename in FILES:
                member = f"ml-1m/{filename}"
                if member in archive.namelist():
                    with archive.open(member) as src, (target / filename).open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    print(f"Extracted {member}")
    finally:
        archive_path.unlink(missing_ok=True)

    if not ratings.exists():
        raise RuntimeError("ratings.dat not found in the downloaded archive")
    print("Download complete. Raw data is excluded from Git.")


if __name__ == "__main__":
    main()
