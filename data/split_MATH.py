#!/usr/bin/env python3

"""
Generate data/MATH_csv/MATH_full_{test,train}.csv files, which contain a subset of the MATH
dataset, removing all questions without integer answers or containing diagrams.
"""

from collections import defaultdict
import json
import os
import pandas as pd
import random
import re


random.seed(99)

SRC_DIR = "../scrap/MATH"  # The extracted MATH dataset
OUT_DIR = "MATH_csv"
os.makedirs(OUT_DIR, exist_ok = True)

for dset in ("test", "train"):

    df = pd.DataFrame(columns = ['prob_name', 'problem', 'level', 'difficulty', 'subject', 'solution', 'answer'])
    lastlen = 0

    set_dir = SRC_DIR + "/" + dset
    subjects = os.listdir(set_dir)
    for subject in subjects:
        print(dset, subject)
        excluded = 0
        no_boxed = 0
        not_int = 0
        #frames = [pd.DataFrame() for i in range(0,5+1)]

        subj_dir = set_dir + "/" + subject

        if subject == 'intermediate_algebra':
            subject = 'int_algebra'
        elif subject == 'counting_and_probability':
            subject = 'counting_prob'

        files = 0
        added = 0
        for fname in os.listdir(subj_dir):
            files += 1
            with open(subj_dir + "/" + fname) as infile:
                data = json.load(infile)
            num = int(fname.split('.')[0])
            prob_name = subject + "-" + str(num)
            data['prob_name'] = prob_name
            try:
                problem = data['problem']
                if '[asy]' in problem or '.gif' in problem or '.png' in problem or 'figure' in problem:
                    excluded += 1
                    continue

                #result_output = re.findall(r'\\boxed\{(-?\d+[.]?\d*(?:e\d+)?)\}', result)
                result_output = re.findall(r'\\boxed{(.*)}', data['solution'])

                if len(result_output) == 0:
                    #print(result_output, "FROM", data['solution'])
                    no_boxed += 1
                    continue
                result_output = result_output[-1]
                try:
                    data['answer'] = int(result_output.strip())
                except:
                    #print('Not int', result_output)
                    not_int += 1
                    continue

                try:
                    level = int(data['level'].split()[1])  # was eg "Level 1"
                except:
                    print("Skip level=", data['level'])
                    continue
                del data['type']  # eg "Counting & Probability"
                data['subject'] = subject
                data['difficulty'] = level - 1
                assert prob_name not in df.index
                df.loc[len(df)] = data
                added += 1
            except RuntimeError:
                print(data, num)
        print("excluded", excluded, "no_boxed", no_boxed, "not_int", not_int, "added remaining", added, "total", len(df))


    df = df.sort_index()
    def tally(frame):
        print(frame.value_counts(['level', 'subject']).sort_index())
    tally(df)

    shuf = list(df.index)
    random.shuffle(shuf)

    if dset == "train":
        split = int(len(shuf) * 0.75)
        train = df.loc[shuf[:split]]
        print("train split")
        tally(train)
        train.to_csv(OUT_DIR + f"/MATH_new_train.csv")
        valid = df.loc[shuf[split:]]
        print("valid split")
        tally(valid)
        valid.to_csv(OUT_DIR + f"/MATH_new_valid.csv")

    else:
        df.to_csv(OUT_DIR + f"/MATH_new_{dset}.csv")

    # if dset == "train":
    #     for level in (3,4,5):
    #         subdf = df[(df.subject == 'algebra') & (df.level == level)]
    #         subdf.to_csv(OUT_DIR + f"/MATH_algebra{level}_{dest}.csv")
