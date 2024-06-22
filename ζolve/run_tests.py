#!/usr/bin/env python3

import ζ.dsl_parse
import ζ.util


testfiles = [
    "examples/is_prime.py",
    "examples/min_multiarg_goal.py",
    "examples/max_pylist_goal.py",
    "examples/max_pyset_goal.py",
    "invalid_examples/multiple_solns.py",
    "invalid_examples/no_soln.py",
]


for testfile in testfiles:
    print("\n######################################## Testing", testfile)
    workspace = ζ.dsl_parse.load_dsl_file(testfile, verbose = False)
    ans = workspace.solve()
    expected = workspace.locals['expected_answer']
    if expected in (None, False) and ans is not expected:
        print("Expected ", expected, "got", ans)
    elif ans != expected:  # An int or float
        print("Expected", expected, "got", ans)
    else:
        print("PASS")
