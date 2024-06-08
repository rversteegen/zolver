#!/usr/bin/env python3

"""
Generate ArtOfProblemSolving csvs by splitting up the raw parsed_ArtOfProblemSolving.csv into small csvs: train, valid, test, test24,
and also into each competition, while throwing away many questions without integer answers or containing diagrams or garbage.
Any questions in the miniF2F valid or test sets are put in valid or test splits.
However there's some overlap with MATH too.
"""

import os
import pandas as pd
import random
import re


random.seed(101)  # Selected so AIME and AMC12 test and valid datasets have nearly same difficulty

MINIF2F = "miniF2F"  # eg a symlink, https://github.com/openai/miniF2F.git
DATA_DIR = "ArtOfProblemSolving"

# From https://www.kaggle.com/datasets/alexryzhkov/amio-parsed-art-of-problem-solving-website
df = pd.read_csv(DATA_DIR + "/parsed_ArtOfProblemSolving.csv")

minif2f_test = os.listdir(MINIF2F + "/metamath/test")
minif2f_valid = os.listdir(MINIF2F + "/metamath/valid")


df = df.drop(columns = ['solution']).drop_duplicates()
df.insert(1, 'prob_name', "")
df.insert(1, 'competition', "")
df.insert(1, 'difficulty', 0)
df.insert(1, 'grouping', "")

# Parse link
for index in df.index:

    link = df.loc[index, 'link']
    #link = df['link'][index]

    # E.g. 'https://artofproblemsolving.com/wiki/index.php/2024_AMC_8_Problems/Problem_3'
    comp, probname = link.split('/')[-2:]
    assert comp[4] == '_'
    assert comp.endswith('_Problems')
    comp = comp.replace('_Problems', '')

    df.loc[index, 'prob_name'] = comp + '_' + probname

    # Cleaning
    #fullcomp = comp
    year = comp.split("_")[0]
    comp = comp[5:]  # Remove year
    comp = comp.replace('Fall_', '').replace('USO', 'USA')
    suffix = ""
    for suff in ('A', 'B', 'P', '_I', '_II'):
        if comp.endswith(suff):
            comp = comp[:-len(suff)]
            suffix = suff

    df.loc[index, 'competition'] = comp

    # Estimate difficulty based on https://artofproblemsolving.com/wiki/index.php/AoPS_Wiki:Competition_ratings
    # (note, wiki gives different rankings for each contest in three different places; using the table for each indivi as described #AIME
    rating = 0
    assert probname.startswith("Problem_")
    probnum = int(probname.split("_")[1])
    if comp == "AIME":
        if 1 <= probnum <= 5: rating = 3
        elif 6 <= probnum <= 9: rating = 4
        elif 10 <= probnum <= 12: rating = 5
        elif 13 <= probnum <= 15: rating = 6
    if comp == "AMC_12":
        if 1 <= probnum <= 10: rating = 2
        elif 11 <= probnum <= 20: rating = 3
        elif 21 <= probnum <= 25: rating = 4
    if comp == "AMC_10":
        if 1 <= probnum <= 5: rating = 1
        elif 6 <= probnum <= 20: rating = 2
        elif 21 <= probnum <= 25: rating = 3
    if comp == "AMC_8":
        rating = 1

    df.loc[index, 'difficulty'] = rating

    # Assign grouping

    check_fname = ""
    if comp == "AMC_12":
        # E.g. "2019_AMC_12A"
        #which = fullcomp.split("_")[2]
        # Files eg "amc12b-2021-p9.mm"
        check_fname = f"amc12{suffix.lower()}-{year}-p{probnum}.mm"
    if comp == "AIME":
        # Files eg "aime-1991-p9.mm", aimeII-2020-p6.mm
        check_fname = f"aime{suffix[1:]}-{year}-p{probnum}.mm"

    if check_fname in minif2f_test:
        df.loc[index, 'grouping'] = "test"
        print("Found", check_fname, " in test")
    if check_fname in minif2f_valid:
        df.loc[index, 'grouping'] = "valid"
        print("Found", check_fname, " in valid")
    if int(year) >= 2023:
        df.loc[index, 'grouping'] = "test24"


total_unfiltered = df.shape[0]

# Sort so that AMC 10 is before AMC 12, prefer AMC 10 in case of duplicates
df.sort_values(by = 'competition', inplace = True, ignore_index = True)

# E.g.
# $(\textbf{A})\: 75\qquad(\textbf{B}) ...
# $\mathrm{(A) \ } 3\pi\qquad \mathrm{(B) \ } ...
# $\textrm{(A)}\ 2 \qquad \textrm{(B)}\
# (A) 10 (B) 13 (C) 27 ...
# \[\textbf{(A)}\ 5\qquad\textbf{(B)}
# [mathjax]\textbf{(A) }50\qquad\textbf{(B)
# [katex]\text{(A)}\ 1 \qquad \text{(B)}
multi_re = re.compile(r'\n\$?\(A\)' '|' r'(\$\(?|\\\[|\[mathjax\]|\[katex\])' r' ?\\(text|textbf|textrm|mathrm) ?\{\(?A')

# Clean answers, and remove rows with non-integer answers or with figures or non-unique questions
seen_probs = set()
duplicates = 0
garbage = 0
with_figures = 0
selected = []
for index in df.index:
    select = True
    problem = df['problem'][index]

    if problem in seen_probs:
        selected.append(False)
        duplicates += 1
        continue
    seen_probs.add(problem)

    if 'problem_id' in df.loc[index, 'problem']:
        selected.append(False)
        garbage += 1
        continue

    # [asy] is a markup language for figures
    # Some problems mention a figure but are missing it
    if '[asy]' in problem or '.gif' in problem or '.png' in problem or 'figure' in problem:
        #print(index, df['problem_id'][index], df['link'][index])
        #print(problem)
        selected.append(False)
        with_figures += 1
        continue

    answer = df['answer'][index]
    if answer[-1] == '.':
        answer = answer[:-1]
    if len(answer) > 4 and answer[-4] == ',':  # Probably thousands separator
        answer = answer.replace(',', '')
    try:
        int(answer)
    except ValueError:
        #print("Skip answer", df['answer'][index])
        select = False

    # Remove multichoice answers from questions
    multi_match = multi_re.search(problem)
    if multi_match:
        problem = problem[:multi_match.start()]
    # Some formerly multichoice problems end in ":"
    problem = problem.strip()
    if problem[-1] == ':':
        problem = problem[:-1] + '?'
    df.loc[index, 'problem'] = problem

    selected.append(select)
df = df.loc[selected]

total = df.shape[0]
total_multichoice = df.count()['letter']  # letter not NaN
total_numerical = total - total_multichoice

print(f"{total} out of {total_unfiltered} problems kept, {duplicates} duplicates, {with_figures} have figures, {garbage} garbage")

print(df.value_counts('competition'))

print("XXXXXXXXXX", df[df['problem_id'] == '18bf638a0cb797d222d008b988c16b6d'])

# Randomly select 100 of each competition as a validation set

for comp in ('AHSME', 'AMC_8', 'AMC_10', 'AMC_12', 'AIME'):
    this_comp = df.loc[df['competition'] == comp]
    this_comp = this_comp.rename({'problem_id':'id'}, axis = 1)

    valid_idx = this_comp.index[this_comp['grouping'] == 'valid']
    test_idx = this_comp.index[this_comp['grouping'] == 'test']
    test24_idx = this_comp.index[this_comp['grouping'] == 'test24']
    other_idx = this_comp.index[this_comp['grouping'] == '']

    print(comp, "valid", len(valid_idx), "test", len(test_idx), "test24", len(test24_idx), "remain", len(other_idx))

    valid_idx = valid_idx.union( random.sample(list(other_idx), 50 - len(valid_idx)) )
    other_idx = other_idx.difference(valid_idx)

    test_idx = test_idx.union ( random.sample(list(other_idx), 50 - len(test_idx)) )
    other_idx = other_idx.difference(test_idx)

    print(comp, "*valid", len(valid_idx), "*test", len(test_idx), "test24", len(test24_idx), "remain", len(other_idx))

    print("VALID", this_comp.loc[valid_idx].value_counts('difficulty'))
    print("TEST", this_comp.loc[test_idx].value_counts('difficulty'))

    this_comp.loc[valid_idx].to_csv(DATA_DIR + f"/{comp}_valid.csv")
    this_comp.loc[other_idx].to_csv(DATA_DIR + f"/{comp}_train.csv")
    this_comp.loc[test_idx].to_csv(DATA_DIR + f"/{comp}_test.csv")
    this_comp.loc[test24_idx].to_csv(DATA_DIR + f"/{comp}_test24.csv")


    # print(comp, this_comp.shape)
    # print(random.choice(this_comp['problem'].array))
    # print()
