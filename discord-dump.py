import csv
import sys
import os


gdpr_dump = sys.argv[1]


def collect(file, keys):
    with open(file, 'r') as f:
        for i in csv.DictReader(f):
            for k in keys:
                print(i[k])


for path, _, files in os.walk(gdpr_dump):
    for file in files:
        if file.endswith(".csv"):
            with open(f"{path}/{file}", 'r') as f:
                for i in csv.DictReader(f):
                    print(i["Contents"])
