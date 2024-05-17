import math

def asserteq(res, expected):
    if res != expected:
        print("expected ", expected, " is != ", res)
        assert False

def assertclose(res, expected):
    if not math.isclose(res, expected):
        print("expected ", expected, " is != ", res)
        assert False
