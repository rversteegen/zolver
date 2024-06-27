#!/usr/bin/env python3

import os
import ζ.dsl_parse
import ζ.util
import ζ.solver

rootdir = "/mnt/common/proj/kaggle/AIMO/ζolve/"

testfiles =[]
for dirpath, dirnames, filenames in os.walk(rootdir + 'tests'):
    for fname in filenames:
        if fname.endswith('.py'):
            testfiles.append(os.path.join(dirpath, fname).replace(rootdir, ''))

# testfiles = [
#     #"examples/AIMO_example4.py"
#     # "tests/valid_dsl/count-734.py"
#     # "tests/valid_dsl/count-simple.py"
#     "tests/valid_dsl/forall.py",
#     "tests/valid_dsl/forall2.py",
#     "tests/valid_dsl/forall3.py",
#     "tests/valid_dsl/forall4.py",
# ]

skips = [
     "tests/valid_dsl/expo2.py",
]

failures = ""

for testfile in testfiles:
    if testfile in skips:
        continue
    print("\n######################################## Testing", testfile)
    workspace = ζ.dsl_parse.load_dsl_file(rootdir + testfile, verbose = True)#False)
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
    elif expected == '?':
        pass
    else:
        # WTF, whys is Integer(3) != 3.0 sometimes
        if expected.is_Integer:
            expected = int(expected)
        elif expected.is_Float:
            expected = float(expected)

    if ans != expected:
        print("Expected", expected, "got", ans)
        failures += testfile + "\n"
    else:
        print("Got expected", ans)
        print("PASS")


print("\nFailures:\n" + failures)
