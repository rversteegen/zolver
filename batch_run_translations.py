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
from ζ import dsl, dsl_parse, ζ3, solver

INPUTDIR = "translations/"

extracts = ""

selections = []

stats_template = {
            'prob_name' : '',
            'total': 1,  #len(df),
            'parsed': 0,
            'parsefailed': 0,
            'solvefailed': 0,
            'solve_ran': 0,
            'trivial': 0,
            'trivialcorrect': 0,
            'trivialwrong': 0,
            'wrong': 0,
            'correct': 0,
            'unimp': 0,
            'unsat': 0,
            'unknown': 0,
            'excepts': 0,
        }
by_prob = pd.DataFrame(stats_template, index = [])

def process_file(fname):
    global extracts
    global selections
    global by_prob

    df = pd.read_csv(INPUTDIR + fname)

    allstats = pd.DataFrame(stats_template, index = [fname])

    for idx in df.index:
        correct = False
        row = df.loc[idx]
        stats = pd.DataFrame(stats_template, index = [fname])
        stats.prob_name = row.prob_name

        print("********************************************************************************")
        print(f"\n\n\n###### Problem {row.prob_name}:\n{row.problem}\n\n###### Translation:\n{row.translation}\n")
        try:
            workspace = dsl_parse.load_dsl(row.translation, verbose = False)
            print("PARSE SUCCESS")
            workspace.print()
            stats.parsed += 1
            if workspace.goal is not None and workspace.goal.is_constant:
                stats.trivial += 1
            try:
                res = workspace.solve()
                stats.solve_ran += 1
                res_note = f"Result {res}, true answer is {row.answer} got {workspace.solution}"
                print("--------------RESULT")
                print(res_note)
                if res == solver.unknown:
                    stats.unknown += 1
                elif res == solver.unsat or res == solver.notunique:
                    stats.unsat += 1
                else:
                    assert res == solver.solved
                    if row.answer == workspace.solution:
                        stats.correct += 1
                        correct = True
                        if workspace.goal.is_constant:
                            stats.trivialcorrect += 1
                        extracts += "\n\n" + row.problem + "\n\n----->\n" + row.translation
                    else:
                        stats.wrong += 1
                        if workspace.goal.is_constant:
                            stats.trivialwrong += 1

            except NotImplementedError as e:
                print("---------------SOLVE FAILED: NotImplementedError")
                stats.unimp += 1
                #stats.solvefailed += 1
                res_note = repr(e)
            except (SyntaxError, dsl.DSLError, ζ3.MalformedError) as e:
                print("---------------SOLVE FAILED")
                stats.solvefailed += 1
                res_note = repr(e)

        except NotImplementedError as e:
            print("---------------PARSE FAILED: NotImplementedError")
            stats.unimp += 1
            #stats.solvefailed += 1
            res_note = repr(e)
        except (SyntaxError, dsl.DSLError, ζ3.MalformedError) as e:
            print("---------------PARSE FAILED")
            print(e)
            line_err = ""
            if e.lineno:
                line_err = f"On line {e.lineno}: " + row.translation.split("\n")[e.lineno]
                print(line_err)
            stats.parsefailed += 1
            res_note = str(e) + " " + line_err
        # except Exception as e:
        #     print("uncaught except", e)
        #     stats.excepts += 1

        if stats.parsefailed[0]:
            selections += [(row.problem, row.translation, res_note)]

        allstats = pd.concat([allstats,stats])
        by_prob = pd.concat([by_prob,stats])


    return allstats


if False:
    LEVEL = 5
    TAG = "MMMistral"
    #TAG = "OMMistral7b"
    VER = "v3"
    fname = f"{TAG}_{VER}_MATH_level{LEVEL}_translations.csv"
    stats = process_file(fname)
    print(stats)
    stats = stats.iloc[0]

else:
    stats = pd.DataFrame()
    for fname in os.listdir(INPUTDIR):
        if fname.endswith('.csv'): # and 'InternLM2Math7b_v4' in fname:
            stats = pd.concat([stats, process_file(fname)])
            # if len(stats)> 100:
            #     break
    print(stats)

    if len(selections):
        random.shuffle(selections)
        with open("translation_selections.txt", "w") as ofile:
            for prob, trans, note in selections:
                ofile.write("## PROBLEM\n" + prob + "\n\n## TRANS\n" + trans + "\n\nRESULT: " + note + "\n\n\n")
        print("Wrote translation_selections.txt")

#print(f"{fname}:\t solved {stats.correct}, wrong {stats.wrong}, {stats.solve_ran} parsed+ran, {stats.parsed} parsed and {stats.parsefailed} failed to parse of {stats.total} total; {stats.excepts} unexpected exceptions\n")




probs = stats.groupby(['prob_name']).sum(numeric_only = True)
print("PROB TALLYS")
print(probs.to_string())


tally = stats.sum(numeric_only = True)
print(f"""


Total: {tally.total}
-- Parsed: {tally.parsed}
---- /Trivial: {tally.trivial}  (trivial goal)
---- solve finished: {tally.solve_ran}
----   Correct: {tally.correct}
----     Trivial: {tally.trivialcorrect}
----   Wrong: {tally.wrong}
----     Trivial: {tally.trivialwrong}
----   Unsat: {tally.unsat}  (inc notunique)
----   Unknown: {tally.unknown}
---- solve failed:  {tally.solvefailed}
------ Notimp:  {tally.unimp}
-- Malformed: {tally.parsefailed}
-- Uncaught except: {tally.excepts}
""")

#print(extracts)
