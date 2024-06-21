#!/usr/bin/env python3

"""
Parse and run translations saved in bulk.
"""

from collections import defaultdict
import pandas as pd
import sys
import os
sys.path.append("ζolve")
from ζ import dsl_parse

INPUTDIR = "translations/new/"

extracts = ""

def process_file(fname):
    global extracts

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
                print("True answer is", row.answer, ans)
                if row.answer == ans:
                    stats.correct += 1
                    extracts += "\n\n" + row.problem + "\n\n----->\n" + row.translation
                else:
                    stats.wrong += 1
            except NotImplementedError:
                print("NotImplementedError")
                stats.unimp += 1
            except Exception as e:
                print("uncaught except", e)
                stats.excepts += 1

        except (SyntaxError, dsl_parse.DSLError) as e:
            print("--------------------------------------------------FAILED")
            print(e)
            stats.failed += 1
        except Exception as e:
            print("uncaught except", e)
            stats.failed += 1
            stats.excepts += 1


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

#print(extracts)
