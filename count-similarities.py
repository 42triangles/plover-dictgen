import json
import sys
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
        rerender_chord(i.strip())
        for i in s.split("/") if i.strip()
    )


def valid(s):
    return set(s) <= set(string.ascii_uppercase + "*#-/ ")


print("Loading scrutinized dictionary")
scrutinized = json.load(open(sys.argv[1]))

print("Preparing scrutinized dictionary")
scrutinized = {
    rerender_chords(k): v.lower()
    for k, v in scrutinized.items()
    if valid(k)
}

print("Loading reference dictionary")
reference = json.load(open(sys.argv[2]))

print("Preparing reference dictionary")
reference_chords = defaultdict(lambda: [])
for k, v in reference.items():
    if valid(k):
        reference_chords[v.lower()].append(rerender_chords(k))

print("Loading word counts")
word_counts = defaultdict(lambda: 0)
with open(sys.argv[3]) as f:
    for i in f:
        word, count = i.rsplit(' ', 1)
        word_counts[word.lower()] += int(count)

print("Finding shared words")
words = set(scrutinized.values())\
    .intersection(reference_chords.keys())

covered = 0
semicovered = 0
uncovered = 0

word_count_sum = 0
covered_counts = 0
semicovered_counts = 0
uncovered_counts = 0

coverage_max = 0
coverage = 0

coverage_max_counts = 0
coverage_counts = 0

print("Counting coverage")
for i in words:
    count = word_counts.get(i.lower(), 0)
    word_count_sum += count

    word_coverage = None
    for j in reference_chords[i]:
        coverage_max += 1
        coverage_max_counts += count
        if scrutinized.get(j) == i:
            coverage += 1
            coverage_counts += count

            if word_coverage is None:
                word_coverage = "full"
            elif word_coverage == "none":
                word_coverage = "partial"
        else:
            if word_coverage is None:
                word_coverage = "none"
            elif word_coverage == "full":
                word_coverage = "partial"

    if word_coverage == "none":
        uncovered += 1
        uncovered_counts += count
    elif word_coverage == "partial":
        semicovered += 1
        semicovered_counts += count
    else:
        covered += 1
        covered_counts += count

print("=" * 80)
sidespace = (80 - len("RESULTS")) // 2
print(f"{' ' * sidespace}RESULTS")
print("=" * 80)
print(f"Word count: {len(words)}")
print(f"Fully covered (scrutinized >= reference): {covered}")
print(f"  = {covered / len(words) * 100:.2f}%")
print(
    f"Partially covered (len(scrutinized & reference) > 0): "
    f"{semicovered}"
)
print(f"  = {semicovered / len(words) * 100:.2f}%")
print(f"Uncovered (not (scrutinized & reference)): {uncovered}")
print(f"  = {uncovered / len(words) * 100:.2f}%")
print()
print(f"Number of translations: {coverage_max}")
print(f"Number of covered translations : {coverage}")
print(f"  = {coverage / coverage_max * 100:.2f}%")
print()

sidespace = (80 - len("RELATIVE")) // 2
print(f"{' ' * sidespace}RELATIVE")
print()
print(f"Word count sum: {word_count_sum}")
print(
    f"Fully covered (scrutinized >= reference): "
    f"{covered_counts}"
)
print(f"  = {covered_counts / word_count_sum * 100:.2f}%")
print(
    f"Partially covered (len(scrutinized & reference) > 0): "
    f"{semicovered_counts}"
)
print(f"  = {semicovered_counts / word_count_sum * 100:.2f}%")
print(
    f"Uncovered (not (scrutinized & reference)): "
    f"{uncovered_counts}"
)
print(f"  = {uncovered_counts / word_count_sum * 100:.2f}%")
print()
print(f"Number of translations: {coverage_max_counts}")
print(f"Number of covered translations : {coverage_counts}")
print(f"  = {coverage_counts / coverage_max_counts * 100:.2f}%")
