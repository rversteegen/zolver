#!/usr/bin/env python3

import os
import ζ.dsl_parse
import ζ.util


testfiles = []
for dirpath, dirnames, filenames in os.walk('tests'):
    for fname in filenames:
        if fname.endswith('.py'):
            testfiles.append(os.path.join(dirpath, fname))


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
