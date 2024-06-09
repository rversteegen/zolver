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

    df = pd.DataFrame(columns = ['problem', 'level', 'subject', 'solution', 'answer'])

    set_dir = SRC_DIR + "/" + dset
    subjects = os.listdir(set_dir)
    for subject in subjects:
        print(dset, subject)
        excluded = 0
        no_boxed = 0
        not_int = 0
        #frames = [pd.DataFrame() for i in range(0,5+1)]

        subj_dir = set_dir + "/" + subject
        for fname in os.listdir(subj_dir):
            with open(subj_dir + "/" + fname) as infile:
                data = json.load(infile)
            num = int(fname.split('.')[0])
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

                level = int(data['level'].split()[1])  # was eg "Level 1"
                data['level'] = level
                del data['type']  # eg "Counting & Probability"
                data['subject'] = subject
                df.loc[num] = data
            except:   # A couple have Level "?"
                print(data, num)
        print("excluded", excluded, "no_boxed", no_boxed, "not_int", not_int, "remaining", len(df))


    df = df.sort_index()
    print(df.value_counts(['level', 'subject']).sort_index())
    #print(df)

    df.to_csv(OUT_DIR + f"/MATH_full_{dset}.csv")

    if dset == "train":
        for level in (3,4,5):
            subdf = df[(df.subject == 'algebra') & (df.level == level)]
            subdf.to_csv(OUTDIR + f"/MATH_algebra{level}_{dest}.csv")
