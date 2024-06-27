#!/usr/bin/env python3

import os
import ζ.dsl_parse
import ζ.util
import ζ.solver

testfiles =[]
for dirpath, dirnames, filenames in os.walk('tests'):
    for fname in filenames:
        if fname.endswith('.py'):
            testfiles.append(os.path.join(dirpath, fname))

testfiles = [
    #"examples/AIMO_example4.py"
    "tests/valid_dsl/count-734.py"
]


failures = ""

for testfile in testfiles:
    print("\n######################################## Testing", testfile)
    workspace = ζ.dsl_parse.load_dsl_file(testfile, verbose = True)#False)
    ret = workspace.solve()
    if ret == ζ.solver.solved:
        ans = workspace.solution
    else:
        ans = ret  # unknown, unsat, or notunique

    expected = workspace.locals['expected_answer']
    if expected == 'unknown':
        expected = ζ.solver.unknown
    elif expected == 'unsat':
        expected = ζ.solver.unsat
    elif expected == 'notunique':
        expected = ζ.solver.notunique

    if ans != expected:
        print("Expected", repr(expected), "got", ans)
        failures += testfile + "\n"
    else:
        print("PASS")


print("\nFailures:\n" + failures)
