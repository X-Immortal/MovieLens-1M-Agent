#!/usr/bin/env python3
"""Hadoop Streaming mapper for MovieLens row-level quality facts."""
import argparse
import hashlib
import json
import os
import sys

MIN_TS = 946684800       # 2000-01-01 UTC
MAX_TS = 1100000000      # 2004-11-09 UTC, safely beyond the ml-1m period
# These are registered for Iteration 1 and must remain stable for later rounds.
T1_EPOCH = 1028504458
T2_EPOCH = 1037479524
VALID_AGES = {1, 18, 25, 35, 45, 50, 56}
VALID_GENRES = {
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
}


def load_ids(path, expected_fields):
    if not path:
        return set()
    result = set()
    try:
        with open(path, encoding="latin-1") as handle:
            for raw in handle:
                fields = raw.rstrip("\r\n").split("::")
                if len(fields) == expected_fields:
                    try:
                        result.add(int(fields[0]))
                    except ValueError:
                        pass
    except OSError:
        pass
    return result


def facts(table, raw, user_ids, movie_ids):
    fields = raw.split("::")
    expected = {"ratings": 4, "movies": 3, "users": 5}[table]
    complete = len(fields) == expected and all(field.strip() for field in fields)
    accurate = False
    consistent = False
    up_to_date = False
    key = None
    timestamp = None
    unknown_user = 0
    unknown_movie = 0

    if table == "ratings" and len(fields) == 4:
        try:
            uid, mid, rating, timestamp = map(int, fields[0:4])
            accurate = (
                complete and uid > 0 and mid > 0 and 1 <= rating <= 5
                and MIN_TS <= timestamp <= MAX_TS
            )
            up_to_date = complete and MIN_TS <= timestamp <= MAX_TS
            unknown_user = int(bool(user_ids) and uid not in user_ids)
            unknown_movie = int(bool(movie_ids) and mid not in movie_ids)
            consistent = accurate and not unknown_user and not unknown_movie
            key = f"{uid}|{mid}|{timestamp}"
        except ValueError:
            pass
    elif table == "movies" and len(fields) == 3:
        try:
            movie_id = int(fields[0])
            genres = [genre.strip() for genre in fields[2].split("|") if genre.strip()]
            accurate = complete and movie_id > 0 and bool(fields[1].strip()) and bool(genres)
            consistent = accurate and all(genre in VALID_GENRES for genre in genres)
            up_to_date = complete
            key = str(movie_id)
        except ValueError:
            pass
    elif table == "users" and len(fields) == 5:
        try:
            user_id, age, occupation = int(fields[0]), int(fields[2]), int(fields[3])
            accurate = complete and user_id > 0 and fields[1] in {"M", "F"} \
                and age in VALID_AGES and 0 <= occupation <= 20
            consistent = accurate and bool(fields[4].strip())
            up_to_date = complete
            key = str(user_id)
        except ValueError:
            pass

    valid = int(accurate and consistent)
    summary = {
        "total": 1,
        "complete": int(complete),
        "accurate": int(accurate),
        "consistent": int(consistent),
        "up_to_date": int(up_to_date),
        "valid": valid,
        "unknown_user": unknown_user,
        "unknown_movie": unknown_movie,
        "min_timestamp": timestamp if isinstance(timestamp, int) and timestamp >= MIN_TS else None,
        "max_timestamp": timestamp if isinstance(timestamp, int) and timestamp <= MAX_TS else None,
        "train": int(table == "ratings" and isinstance(timestamp, int) and MIN_TS <= timestamp <= T1_EPOCH),
        "validation": int(table == "ratings" and isinstance(timestamp, int) and T1_EPOCH < timestamp <= T2_EPOCH),
        "test": int(table == "ratings" and isinstance(timestamp, int) and timestamp > T2_EPOCH and timestamp <= MAX_TS),
    }
    return summary, key, hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True, choices=["ratings", "movies", "users"])
    parser.add_argument("--users-file")
    parser.add_argument("--movies-file")
    args = parser.parse_args()
    user_ids = load_ids(args.users_file, 5)
    movie_ids = load_ids(args.movies_file, 3)
    for raw_line in sys.stdin:
        raw = raw_line.rstrip("\r\n")
        summary, key, raw_hash = facts(args.table, raw, user_ids, movie_ids)
        print("S\t" + json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
        if key is not None:
            print("K:" + key + "\t1")
        print("R:" + raw_hash + "\t1")


if __name__ == "__main__":
    main()
