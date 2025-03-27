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

testfiles = [               # 
"tests/valid_dsl/set_comp.py",
"tests/valid_dsl/constraints_on_inst_elements.py",
#     #"examples/AIMO_example4.py"
       "tests/valid_dsl/count-734.py",
# "tests/valid_dsl/count_interval.py",
# "tests/valid_dsl/count-simple.py",


# #    "tests/valid_dsl/setcomps.py",

#     # "tests/valid_dsl/set_min_with_constraints.py",
#     # "tests/valid_dsl/max_vacant_set.py",
#     "tests/valid_dsl/max_empty_set.py",

  "tests/valid_dsl/seqdecl.py",
  "tests/valid_dsl/sumints.py",
]

testfiles = [
 "tests/valid_dsl/seq_sum.py",
 "tests/valid_dsl/len_resolved_later.py",
]

# testfiles = [
# # "tests/valid_dsl/count-simple.py",
#  "tests/valid_dsl/count_range.py",
# ]
skips = [
"tests/valid_dsl/more_than_once.py",
    "tests/valid_dsl/setcomps.py",
     "tests/valid_dsl/expo2.py",
     "tests/valid_dsl/triangle.py",  # wrong 
   "tests/valid_dsl/funcs.py", # wrong
    "tests/valid_dsl/forall_func.py",   # fails to solve
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

