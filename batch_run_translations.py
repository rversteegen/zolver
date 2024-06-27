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

INPUTDIR = "translations4/"

extracts = ""

selections = []

stats_template = {
    'fname': '',
    'temp': 0,
            'prob_name' : '',
            'difficulty': 0,
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

    allstats = pd.DataFrame(stats_template, index = [0])

    for idx in df.index:
        correct = False
        row = df.loc[idx]
        stats = pd.DataFrame(stats_template, index = [0])
        stats.fname = fname
        stats.temp = row.temperature
        stats.prob_name = row.prob_name
        stats.difficulty = int(row.difficulty)

        print("********************************************************************************")
        print(f"\n\n\n###### Problem {row.prob_name}:\n{row.problem}\n\n###### Translation:\n{row.translation}\n")
        try:
            workspace = dsl_parse.load_dsl(row.translation, verbose = False)
            print("PARSE SUCCESS")
            workspace.print()
            stats.parsed += 1
            trivial = dsl.is_a_constant(workspace.goal)
            if workspace.goal is not None and trivial:
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
                        if trivial:
                            stats.trivialcorrect += 1
                        extracts += "\n\n" + row.problem + "\n\n----->\n" + row.translation
                    else:
                        stats.wrong += 1
                        if trivial:
                            stats.trivialwrong += 1

            except NotImplementedError as e:
                print("---------------SOLVE FAILED: NotImplementedError")
                stats.unimp += 1
                #stats.solvefailed += 1
                res_note = repr(e)
            except (AssertionError, SyntaxError, dsl.DSLError, ζ3.MalformedError, ζ3.z3.Z3Exception) as e:
                print("---------------SOLVE FAILED")
                stats.solvefailed += 1
                res_note = repr(e)

        except NotImplementedError as e:
            print("---------------PARSE FAILED: NotImplementedError")
            stats.unimp += 1
            #stats.solvefailed += 1
            res_note = repr(e)
        except (AssertionError, SyntaxError, dsl.DSLError, ζ3.MalformedError) as e:
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
            selections += [(fname, row.prob_name, row.problem, row.translation, res_note)]

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
        if fname.endswith('.csv'):#  and 'v8' in fname:
            stats = pd.concat([stats, process_file(fname)])
            # if len(stats)> 100:
            #     break
    print(stats)

    if len(selections):
        #random.shuffle(selections)
        with open("translation_selections.txt", "w") as ofile:
            for fname, prob_name, prob, trans, note in selections:
                ofile.write(fname + " " + prob_name + "\n## PROBLEM\n" + prob + " " + "\n\n## TRANS\n" + trans + "\n\nRESULT: " + note + "\n\n\n")
        print("Wrote translation_selections.txt")

#print(f"{fname}:\t solved {stats.correct}, wrong {stats.wrong}, {stats.solve_ran} parsed+ran, {stats.parsed} parsed and {stats.parsefailed} failed to parse of {stats.total} total; {stats.excepts} unexpected exceptions\n")




print("######################################## TALLY BY PROBLEM")
probs = stats.groupby(['prob_name']).sum(numeric_only = True)
print(probs.to_string())


print("######################################## TALLY BY FILE")
files = stats.groupby(['fname',]).sum(numeric_only = True)
print(files.to_string())

print("######################################## TALLY BY FILE+TEMP")
files = stats.groupby(['fname','temp']).sum(numeric_only = True)
print(files.to_string())


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
