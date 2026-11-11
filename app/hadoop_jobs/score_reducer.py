#!/usr/bin/env python3
import json
import sys

groups = {"before": [], "after": []}
for line in sys.stdin:
    key, _, value = line.rstrip("\r\n").partition("\t")
    if key == "score":
        item = json.loads(value)
        groups[item["phase"]].append(item["item"])

def score(items):
    total = sum(x["total"] for x in items)
    ratings = next(x for x in items if x["table"] == "ratings")
    def ratio(name):
        return round(sum(x[name] for x in items) / total * 100, 2) if total else None
    return {
        "Accurate": ratio("accurate"),
        "Complete": ratio("complete"),
        "Unique": round(sum(x["unique_records"] for x in items) / total * 100, 2) if total else None,
        "Up-to-date": round(ratings["up_to_date"] / ratings["total"] * 100, 2) if ratings["total"] else None,
        "Consistent": ratio("consistent"),
    }

before = score(groups["before"])
after = score(groups["after"])
print("scores\t" + json.dumps({
    "before": before,
    "after": after,
    "delta": {name: round(after[name] - before[name], 2) for name in before},
}, separators=(",", ":")))
