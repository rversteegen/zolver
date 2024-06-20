#!/usr/bin/env python3

"""
Parse and run translations saved in bulk.
"""

from collections import defaultdict
import pandas as pd
import sys
sys.path.append("ζolve")
from ζ import dsl_parse

info = ""

TAG = "MMMistral"  # "OMMistral7b"
#TAG = "OMMistral7b"
VER = "v3"
for LEVEL in (3,4,5):

    fname = f"{TAG}_{VER}_MATH_level{LEVEL}_translations.csv"
    df = pd.read_csv("translations/" + fname)

    parsed = 0
    failed = 0
    ran = 0

    for idx in df.index:
        row = df.loc[idx]
        #rint(row)
        print(f"\n\n\n###### Problem:\n{row.problem}\n\n###### Translation:\n{row.translation}\n")
        try:
            workspace = dsl_parse.load_dsl(row.translation, verbose = False)
            print("--------------------------------------------******SUCCESS")
            workspace.print()
            parsed += 1
            workspace.solve()
            print("DONE")
            ran += 1
            #exit()

        except (SyntaxError, dsl_parse.DSLError) as e:
            print("--------------------------------------------------FAILED")
            print(e)
            failed += 1

    info += f"{fname}:\t {ran} parsed+ran, {parsed} parsed and {failed} failed to parse\n"

print(info)


