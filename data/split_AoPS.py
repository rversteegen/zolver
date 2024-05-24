#!/usr/bin/env python3

import pandas as pd
import random
import re


random.seed(99)

DATA_DIR = "ArtOfProblemSolving"
df = pd.read_csv(DATA_DIR + "/parsed_ArtOfProblemSolving.csv")

df = df.drop(columns = ['solution']).drop_duplicates()
df.insert(1, 'prob_name', "")
df.insert(1, 'competition', "")

# Parse link
for index in df.index:

    link = df.loc[index, 'link']
    #link = df['link'][index]

    # E.g. 'https://artofproblemsolving.com/wiki/index.php/2024_AMC_8_Problems/Problem_3'
    comp, probnum = link.split('/')[-2:]
    assert comp[4] == '_'
    assert comp.endswith('_Problems')
    comp = comp[5:].replace('_Problems', '')

    df.loc[index, 'prob_name'] = comp + '_' + probnum

    # Cleaning
    comp = comp.replace('Fall_', '').replace('USO', 'USA')
    for suff in ('A', 'B', 'P', '_I', '_II'):
        if comp.endswith(suff):
            comp = comp[:-len(suff)]

    df.loc[index, 'competition'] = comp

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

print(f"{total} out of {total_unfiltered} problems kept, {duplicates} duplicates, {with_figures} have figures")

print(df.value_counts('competition'))

# Randomly select 100 of each competition as a validation set

for comp in ('AHSME', 'AIME', 'AMC_8', 'AMC_10', 'AMC_12',):
    this_comp = df.loc[df['competition'] == comp]
    val_idx = random.sample(list(this_comp.index), 100)
    train_idx = this_comp.index.difference(val_idx)

    this_comp.loc[val_idx].to_csv(DATA_DIR + f"/{comp}_valid.csv")
    this_comp.loc[train_idx].to_csv(DATA_DIR + f"/{comp}_train.csv")



    # print(comp, this_comp.shape)
    # print(random.choice(this_comp['problem'].array))
    # print()
