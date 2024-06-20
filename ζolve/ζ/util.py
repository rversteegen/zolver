import math
from contextlib import contextmanager #, redirect_stdout
import signal
import resource


def asserteq(res, expected):
    if res != expected:
        print("expected ", expected, " is != ", res)
        assert False

def assertclose(res, expected):
    if not math.isclose(res, expected):
        print("expected ", expected, " is != ", res)
        assert False


# memory_limit and time_limit are from https://www.kaggle.com/code/eabdullin/mathgenie-interlm-20b-interactive-code-running
# who knows the original source.

class TimeoutException(Exception): pass

@contextmanager
def memory_limit(limit, type=resource.RLIMIT_AS):
    soft_limit, hard_limit = resource.getrlimit(type)
    resource.setrlimit(type, (limit, hard_limit)) # set soft limit
    try:
        yield
    finally:
        resource.setrlimit(type, (soft_limit, hard_limit)) # restore


# sometimes it may run code for very long time, we don't want that.
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException(f"Execution time exceeded {seconds} seconds")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
