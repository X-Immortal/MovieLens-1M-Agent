#!/usr/bin/env python3
"""Hadoop Streaming reducer retaining one canonical row per business key."""
import sys
import json

current_key = None
values = []


def flush():
    if current_key is None:
        return
    if current_key.startswith("DATA:"):
        if values:
            first = json.loads(values[0])
            print("DATA\t" + first["canonical"])
            if first.get("repaired"):
                print("REPAIRED\t1")
            for duplicate in values[1:]:
                duplicate_item = json.loads(duplicate)
                if duplicate_item["canonical"] != first["canonical"]:
                    print("CONFLICT\t" + json.dumps({
                        "reason": "conflicting_business_key",
                        "raw": duplicate_item["raw"],
                        "canonical": duplicate_item["canonical"],
                    }, ensure_ascii=False, separators=(",", ":")))
                else:
                    print("DUP\t" + duplicate_item["canonical"])
    elif current_key.startswith("REJECT:"):
        for rejected in values:
            print("REJECT\t" + rejected)


for line in sys.stdin:
    key, _, value = line.rstrip("\r\n").partition("\t")
    if key != current_key:
        flush()
        current_key = key
        values = []
    values.append(value)
flush()
