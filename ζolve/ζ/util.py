import math
import time
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


class Timer:
    """
    Utility class for finding total time spent in multiple sections of code.
    Is a context manager. Use either like:
        timing = Timer()
        with timing:
            ...
    or
        with Timer() as timing:
            ...
        print 'Done in', timing
    """
    def __init__(self):
        self.time = 0.
    def start(self):
        self._start = time.time()
        return self
    def stop(self):
        self.time += time.time() - self._start
        del self._start
        return self
    def __enter__(self):
        self.start()
        return self
    def __exit__(self, *args):
        self.stop()
    def __str__(self):
        if hasattr(self, '_start'):
            #return '<Time.Time running>'
            interval = time.time() - self._start
        else:
            interval = self.time
        if interval < 0.01:
            return '%.3fms' % (1e3 * self.time)
        return '%.3fs' % self.time
