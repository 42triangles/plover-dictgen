import csv
import sys


gdpr_dump = sys.argv[1]


def collect(file, keys):
    with open(f"{gdpr_dump}/{file}.csv", 'r') as f:
        for i in csv.DictReader(f):
            for k in keys:
                print(i[k])


collect("chat_history", ["message"])
collect("comments", ["body"])
collect("messages", ["body"])
collect("posts", ["title", "body"])
