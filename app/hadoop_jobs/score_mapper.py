#!/usr/bin/env python3
import json
import sys

for line in sys.stdin:
    phase, _, value = line.rstrip("\r\n").partition("\t")
    if phase in {"before", "after"}:
        item = json.loads(value)
        print("score\t" + json.dumps({"phase": phase, "item": item}, separators=(",", ":")))
