#!/usr/bin/env python3

import ζ.dsl_parse
import ζ.util


testfiles = [
    "examples/is_prime.py"
]


for testfile in testfiles:
    print("Testing", testfile)
    workspace = ζ.dsl_parse.load_dsl_file(testfile, verbose = False)
    ans = workspace.solve()
    expected = workspace.locals['expected_answer']
    if ans != expected:
        print("Expected", expected, "got", ans)
