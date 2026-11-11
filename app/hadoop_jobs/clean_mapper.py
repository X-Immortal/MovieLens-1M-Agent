#!/usr/bin/env python3
"""Hadoop Streaming mapper for validation, normalization and deduplication."""
import argparse
import json
import sys

VALID_AGES = {1, 18, 25, 35, 45, 50, 56}
VALID_GENRES = {
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
}
MIN_TS = 946684800
MAX_TS = 1100000000


def load_ids(path, expected_fields):
    result = set()
    if not path:
        return result
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


def validate(table, raw, user_ids, movie_ids):
    fields = raw.split("::")
    expected = {"ratings": 4, "movies": 3, "users": 5}[table]
    if len(fields) != expected or any(not field.strip() for field in fields):
        return None, "field_count_or_missing_value"
    try:
        if table == "ratings":
            uid, mid, rating, timestamp = map(int, fields)
            if uid <= 0 or mid <= 0 or rating not in range(1, 6):
                return None, "rating_out_of_range_or_id_invalid"
            if timestamp < MIN_TS or timestamp > MAX_TS:
                return None, "timestamp_out_of_expected_range"
            if user_ids and uid not in user_ids:
                return None, "unknown_user"
            if movie_ids and mid not in movie_ids:
                return None, "unknown_movie"
            return f"{uid}::{mid}::{rating}::{timestamp}", f"{uid}|{mid}|{timestamp}"
        if table == "movies":
            movie_id = int(fields[0])
            genres = [genre.strip() for genre in fields[2].split("|") if genre.strip()]
            if movie_id <= 0 or not fields[1].strip() or not genres:
                return None, "movie_id_title_or_genres_invalid"
            if any(genre not in VALID_GENRES for genre in genres):
                return None, "unknown_genre"
            return f"{movie_id}::{fields[1].strip()}::{'|'.join(genres)}", str(movie_id)
        user_id, gender, age, occupation, zipcode = fields
        user_id, age, occupation = int(user_id), int(age), int(occupation)
        if user_id <= 0 or gender not in {"M", "F"} or age not in VALID_AGES:
            return None, "user_demographic_invalid"
        if occupation < 0 or occupation > 20 or not zipcode.strip():
            return None, "occupation_or_zip_invalid"
        return f"{user_id}::{gender}::{age}::{occupation}::{zipcode.strip()}", str(user_id)
    except ValueError:
        return None, "non_numeric_field"


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
        canonical, key_or_reason = validate(args.table, raw, user_ids, movie_ids)
        if canonical is None:
            payload = {"reason": key_or_reason, "raw": raw}
            print("REJECT:" + key_or_reason + "\t" + json.dumps(payload, ensure_ascii=False))
        else:
            print("DATA:" + key_or_reason + "\t" + json.dumps({
                "raw": raw, "canonical": canonical, "repaired": canonical != raw,
            }, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
