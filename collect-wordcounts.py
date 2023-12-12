from collections import defaultdict
import sys
import string

wordcounts = defaultdict(lambda: 0)

for i in sys.argv[2:]:
    with open(i, 'r') as f:
        for line in f:
            for item in line.split():
                word = item.strip(string.punctuation).lower()
                if len(word) == 1 and word not in ["a", "i"]:
                    continue
                wordcounts[word] += 1

cmudict_words = set()
with open(sys.argv[1]) as f:
    for i in f:
        if i[0] in string.ascii_letters:
            cmudict_words.add(i.split("  ", 1)[0].lower())

for i in list(wordcounts.keys()):
    if i not in cmudict_words:
        del wordcounts[i]

words = list(wordcounts.keys())
words.sort(key=lambda i: wordcounts[i], reverse=True)

for i in words:
    print(f"{i} {wordcounts[i]}")
