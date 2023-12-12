import sys
import json

with open(sys.argv[1]) as f:
    allowed = set(i.lower() for i in json.load(f).values())

with open(sys.argv[2]) as f:
    for i in f:
        if json.loads(i)["word"].lower() in allowed:
            print(i, end="")
