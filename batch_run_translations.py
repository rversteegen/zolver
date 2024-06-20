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

#TAG = "MMMistral"
#TAG = "OMMistral7b"
#VER = "v3"
#for LEVEL in (3,4,5):
    #fname = f"{TAG}_{VER}_MATH_level{LEVEL}_translations.csv"

INPUTDIR = "translations/new/"

info = ""

for fname in os.listdir(INPUTDIR):
    if not fname.endswith('.csv'):
        continue

    df = pd.read_csv(INPUTDIR + fname)

    parsed = 0
    failed = 0
    ran = 0
    excepts = 0
    correct = 0
    wrong = 0

    for idx in df.index:
        row = df.loc[idx]
        #rint(row)
        print(f"\n\n\n###### Problem:\n{row.problem}\n\n###### Translation:\n{row.translation}\n")
        try:
            workspace = dsl_parse.load_dsl(row.translation, verbose = False)
            print("--------------------------------------------******SUCCESS")
            workspace.print()
            parsed += 1
            try:
                ans = workspace.solve()
                print("********************************************************************************DONE")
                ran += 1
                print("True answer is", row.answer, ans)
                if row.answer == ans:
                    correct += 1
                else:
                    wrong += 1
            except NotImplementedError:
                print("NotImplementedError")
            except Exception as e:
                print("uncaught except", e)
                excepts += 1

        except (SyntaxError, dsl_parse.DSLError) as e:
            print("--------------------------------------------------FAILED")
            print(e)
            failed += 1
        except Exception as e:
            print("uncaught except", e)
            failed += 1
            excepts += 1


    info += f"{fname}:\t solved {correct}, wrong {wrong}, {ran} parsed+ran, {parsed} parsed and {failed} failed to parse of {len(df)} total; {excepts} unexpected exceptions\n"

print(info)


