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
# who knows the original source. I modified.

class TimeoutException(Exception): pass

# RLIMIT_DATA (data and heap) amount is roughly 60MB less than RLIMIT_AS (address space) but still
# seems to be 150MB more than the actual RSS. (RLIMIT_RSS does nothing in modern Linux).
# So need at least 200MB.
# Warning, creating a z3.Optimize() solver always fails with "WARNING: out of memory" if an rlimit
# is set, no matter how high it is!
@contextmanager
def memory_limit(limit, type=resource.RLIMIT_DATA):
    "limit in bytes"
    # Try to be more invariant to existing memory use. However, another thread could do anything
    # However maxrss does NOT correspond to RELIMIT_DATA!
    limit += 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("total limit", limit/2**20)

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
