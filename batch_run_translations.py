#!/usr/bin/env python3

"""
Parse and run translations saved in bulk.
"""

from collections import defaultdict
import random
import pandas as pd
import sys
import os
sys.path.append("ζolve")
from ζ import dsl, dsl_parse

INPUTDIR = "translations/new/"

extracts = ""

fails = []

def process_file(fname):
    global extracts
    global fails

    df = pd.read_csv(INPUTDIR + fname)

    stats = pd.DataFrame(
        {
         'total': len(df),
         'parsed': 0,
         'failed': 0,
         'ran': 0,
         'wrong': 0,
         'correct': 0,
         'unimp': 0,
         'excepts': 0,
         }, index = [fname])

    for idx in df.index:
        correct = False
        row = df.loc[idx]
        #rint(row)
        print(f"\n\n\n###### Problem:\n{row.problem}\n\n###### Translation:\n{row.translation}\n")
        try:
            workspace = dsl_parse.load_dsl(row.translation, verbose = False)
            print("--------------------------------------------******SUCCESS")
            workspace.print()
            stats.parsed += 1
            try:
                ans = workspace.solve()
                print("********************************************************************************DONE")
                stats.ran += 1
                res_note = f"True answer is {row.answer} got {ans}"
                print(res_note)
                if row.answer == ans:
                    stats.correct += 1
                    correct = True
                    extracts += "\n\n" + row.problem + "\n\n----->\n" + row.translation
                else:
                    stats.wrong += 1
            except NotImplementedError as e:
                print("NotImplementedError")
                stats.unimp += 1
                res_note = str(e)
            # except Exception as e:
            #     print("uncaught except", e)
            #     stats.excepts += 1

        except (SyntaxError, dsl.DSLError) as e:
            print("--------------------------------------------------FAILED")
            print(e)
            line_err = ""
            if e.lineno:
                line_err = f"On line {e.lineno}: " + row.translation.split("\n")[e.lineno]
                print(line_err)
            stats.failed += 1
            res_note = str(e) + " " + line_err
        # except Exception as e:
        #     print("uncaught except", e)
        #     stats.failed += 1
        #     stats.excepts += 1

        if not correct:
            fails += [(row.problem, row.translation, res_note)]


    return stats


if False:
    LEVEL = 5
    TAG = "MMMistral"
    #TAG = "OMMistral7b"
    VER = "v3"
    fname = f"{TAG}_{VER}_MATH_level{LEVEL}_translations.csv"
    stats = process_file(fname)
    print(stats)
    stats = stats.iloc[0]
    print(f"{fname}:\t solved {stats.correct}, wrong {stats.wrong}, {stats.ran} parsed+ran, {stats.parsed} parsed and {stats.failed} failed to parse of {stats.total} total; {stats.excepts} unexpected exceptions\n")

else:
    stats = pd.DataFrame()
    for fname in os.listdir(INPUTDIR):
        if fname.endswith('.csv'):
            stats = pd.concat([stats, process_file(fname)])
    print(stats)

    random.shuffle(fails)
    with open("translation_failures.txt", "w") as ofile:
        for prob, trans, note in fails:
            ofile.write("## PROBLEM\n" + prob + "\n\n## TRANS\n" + trans + "\n\nRESULT: " + note + "\n\n\n")


#print(extracts)
