import sys
import json
import string
from collections import defaultdict


def rerender_chord(chord):
    out = ""
    had_hyphen = False
    had_mid = False
    for i in chord:
        if i == "-":
            out += i
            had_hyphen = True
        elif i in "*EU" and not had_hyphen:
            out += "-" + i
            had_hyphen = True
        elif i not in "AO*EU" and had_mid and not had_hyphen:
            out += "-" + i
            had_hyphen = True
        else:
            out += i
            if i in "AO":
                had_mid = True
    return out


def rerender_chords(s):
    return "/".join(
        rerender_chord(s.strip())
        for i in s.split("/") if i.strip()
    )


def valid(s):
    return set(s) <= set(string.ascii_uppercase + "*#-/ ")


with open(sys.argv[1]) as f:
    dictionary = json.load(f)

dict_rev = defaultdict(lambda: [])
for k, v in dictionary.items():
    if valid(k):
        dict_rev[v.lower()].append(rerender_chords(k))

min_weights = defaultdict(lambda: float("inf"))
included_translations = defaultdict(lambda: 0)
included_translations_max = defaultdict(lambda: 0)

with open(sys.argv[2]) as f:
    for i in f:
        item = json.loads(i)
        word = item["word"]
        if word in dict_rev:
            plover_candidates = dict_rev[item["word"]]

            first_weight = item["translations"][0]["weight"]
            min_weight = min(
                j["weight"] - first_weight
                for j in item["translations"]
                if j["translation"] in plover_candidates
            )

            included = sum(
                1
                for j in item["translations"]
                if j["translation"] in plover_candidates
            )

            min_weights[word] = \
                min(min_weights[word], min_weight)

            included_translations[word] += included
            included_translations_max[word] += \
                len(plover_candidates)
