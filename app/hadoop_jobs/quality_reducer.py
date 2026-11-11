#!/usr/bin/env python3
"""Hadoop Streaming reducer that aggregates mapper quality facts."""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True)
    args = parser.parse_args()
    total = {k: 0 for k in (
        "total", "complete", "accurate", "consistent", "up_to_date", "valid",
        "unknown_user", "unknown_movie",
        "train", "validation", "test",
    )}
    unique_records = 0
    duplicate_records = 0
    unique_raw = 0
    duplicate_raw = 0
    min_timestamp = None
    max_timestamp = None
    current_key = None
    current_count = 0

    def flush_key():
        nonlocal unique_records, duplicate_records, unique_raw, duplicate_raw
        if current_key is None:
            return
        if current_key.startswith("K:"):
            unique_records += 1
            duplicate_records += max(0, current_count - 1)
        elif current_key.startswith("R:"):
            unique_raw += 1
            duplicate_raw += max(0, current_count - 1)

    for line in sys.stdin:
        key, _, value = line.rstrip("\r\n").partition("\t")
        if key.startswith("K:") or key.startswith("R:"):
            if key != current_key:
                flush_key()
                current_key = key
                current_count = 0
            current_count += 1
            continue
        if current_key is not None:
            flush_key()
            current_key = None
            current_count = 0
        if key == "S":
            fact = json.loads(value)
            for name in total:
                total[name] += int(fact.get(name, 0) or 0)
            ts_min = fact.get("min_timestamp")
            ts_max = fact.get("max_timestamp")
            if ts_min is not None:
                min_timestamp = ts_min if min_timestamp is None else min(min_timestamp, ts_min)
            if ts_max is not None:
                max_timestamp = ts_max if max_timestamp is None else max(max_timestamp, ts_max)
    flush_key()
    result = {
        "table": args.table,
        **total,
        "invalid": total["total"] - total["valid"],
        "unique_records": unique_records,
        "duplicate_records": duplicate_records,
        "unique_raw": unique_raw,
        "duplicate_raw": duplicate_raw,
        "min_timestamp": min_timestamp,
        "max_timestamp": max_timestamp,
    }
    print("summary\t" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
