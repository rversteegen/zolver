import math
from math import inf
import numpy as np
from .util import *

def encode_ℝ(x):
    "Map [-∞, ∞] → [-1, 1]"
    if x >= 0:
        return 1 - 1 / (x + 1)
    else:
        return -1 - 1 / (x - 1)

def decode_ℝ(y):
    "Map [-1, 1] → [-∞, ∞]"
    if y >= 0:
        if y == 1:
            return inf
        return -1 - 1 / (y - 1)
    else:
        if y == -1:
            return -inf
        return 1 - 1 / (y + 1)

# Tests
for x in list(range(-5,5+1)) + [-inf, inf]:
    assertclose(x, decode_ℝ(encode_ℝ(x)))


def nstr(val, integer = False):
    "Convert a number to a string"
    if val == -inf:
        return "-∞"
    if val == inf:
        return "∞"
    if int(val) == val:  # integer:
        return str(int(val))
    return str(val)


def vector_dtype(dim: int, field_names: list[str]):
    fields = []
    for name in field_names:
        if type(name) is not str:
            raise TypeError('Field names must be strings')
        if not name.isidentifier():
            raise ValueError(f'Field names must be valid identifiers: {name!r}')
        fields.append((name, 'float32'))

    if dim:
        fields.append('extra', 'float32', dim - len(fields))

    return np.dtype(fields)


class VecX:
    "Vector representation of an order 0 (X) node."

    def __init__(self, x):
        self.v = np.ndarray([x], dtype = 'float32')

    def __str__(self):
        return str(self.v)

    def __repr__(self):
        return f"VecX({self})"


class VecSetX:
    "Vector representation of an order 1 (Set of X, ->X) constant (no variables)"

    # complete: fully defined by the builtin properties due to containing all possible values within
    #   the bounds & spacing.
    # lbound ubound llimit ulimit: are always correct
    # spacing: ℝ -> 0, ℤ -> 1, even/odd -> 2
    #   Undefined for size <= 1
    #   Should be 0 for union of real intervals,
    field_names = "complete size lbound median ubound spacing integer llimit ulimit".split()

    dtype = vector_dtype(0, field_names)

    def __init__(self, copy_of = None, **kwargs):
        if copy_of is None:
            #v = np.zeros(1, self.dtype)[0]
            # Create a scalar (looks like a tuple even when printed with repr()) full of 0's
            v = np.void(0, self.dtype)
        else:
            if isinstance(copy_of, type(self)):
                v = np.copy(copy_of.v)  # NOTE: returns an np.ndarray, not np.void
            elif isinstance(copy_of, (np.void, np.ndarray)) and copy_of.dtype == self.dtype:
                v = np.copy(copy_of)
            else:
                assert(False)
        for key, val in kwargs.items():
            v[key] = val
        self.v = v

    def __getattr__(self, name):
        return self.v.__getitem__(name)

    def __setattr__(self, name, value):
        if name == 'v':  # if name not in self.field_names:
            object.__setattr__(self, name, value)
        else:
            self.v.__setitem__(name, value)

    def isfinite(self) -> bool:
        #return self.integer and self.lbound != -inf and self.ubound != inf
        return self.size < inf

    def iscountable(self) -> bool:
        return self.isfinite() or self.spacing > 0

    def isint(self) -> bool:
        #return self.size == 0 or (self.spacing > 0 and self.spacing == int(self.spacing))
        return self.integer >= 1

    def __str__(self):
        v = self
        lbound = nstr(v.lbound)
        ubound = nstr(v.ubound)

        if v.size == 0:
            return "{}"
        elif v.size == 1:
            return f"{{{lbound}}}"
        elif v.size == 2:
            return f"{{{lbound}, {ubound}}}"
        elif v.size == 3:
            return f"{{{lbound}, {nstr(v.median)}, {ubound}}}"

        if v.complete == 1:
            if v.spacing == 0:
                if v.llimit == 0 and v.lbound == -inf and v.ulimit == 0 and v.ubound == inf:
                    return "ℝ"

                ret = "[" if v.llimit else "("
                ret += f"{lbound}, {ubound}"
                ret += "]" if v.ulimit else ")"
                return ret

            # Otherwise countable.

            if v.ulimit == 0 and v.ubound == inf:
                if v.llimit == 0 and v.lbound == -inf:
                    if v.spacing == 1:
                        return "ℤ"
                    else:
                        # Assuming spacing==inf invalid
                        return nstr(v.spacing) + "*ℤ"
                elif v.llimit == 1 and v.lbound == v.spacing:
                    if v.spacing == 1:
                        return "ℕ"
                    else:
                        return nstr(v.spacing) + "*ℕ"

            if v.llimit == 1:
                if v.lbound == -inf:
                    ret = f"{{-∞, -∞+{nstr(v.spacing)}, ..."
                else:
                    ret = f"{{{lbound}, {nstr(v.lbound + v.spacing)}, ..."
            else:
                if v.lbound == -inf:
                    ret = "{..."
                else:
                    # ??? A open interval of integers doesn't normally occur
                    ret = f"{{{lbound}<..."
                ret += f", {lbound}+{nstr(v.spacing)}, ..."
                #{3≤ℤ<9}

            if v.ulimit == 1:
                ret += f", {ubound}}}"  # Can include ∞
            elif v.ubound == inf:
                ret += "}"
            else:
                ret += f"<{ubound}}}"
            return ret
        else:
            ret = ""
            if v.lbound > -inf or v.ubound < inf:
                ret += f"[{lbound}, {ubound}]∩"
            # if v.primes == 1:
            #     ret += "Primes∩"
            elif v.integer == 1:
                ret += "ℤ∩"
            ret += f"{{<size={v.size} complete={v.complete} spacing={v.spacing} int={v.integer}>}}"
            return ret

    def __mul__(self, val: float):
        ret = VecSetX(self)
        ret.lbound *= val
        ret.ubound *= val
        ret.median *= val
        ret.spacing *= val
        # size unchanged
        if not math.isfinite(ret.spacing) or ret.spacing != int(ret.spacing):
            ret.integer = 0
        # Don't bother trying to recover ret.integer = 1 if scaled back to integers
        return ret

    def __add__(self, val: float):
        ret = VecSetX(self)
        ret.lbound += val
        ret.ubound += val
        ret.median += val
        # size, spacing unchanged
        if not math.isfinite(ret.lbound) or ret.lbound != int(ret.lbound):
            ret.integer = 0
        # Don't bother trying to recover ret.integer = 1 if shifted back to integers
        return ret

    def __eq__(self, rhs) -> bool:
        return self.equal(rhs) == 1

    def equal(self, rhs) -> float:
        "Returns 0 or 1 for definitely true, false, 0.5 for possible."
        for attr in ('lbound', 'ubound', 'llimit', 'ulimit'):
            if self.v[attr] != rhs.v[attr]:
                return 0
        if self.complete == 1 and rhs.complete == 1:
            return int(self.spacing == rhs.spacing)
        return 0.5
        #return diff(self.spacing, rhs.spacing) * diff(self.complete, rhs.complete)


def Z_interval(lbound, ubound):
    "An interval of integers, always excluding -inf or inf if they are the bounds"
    ret = VecSetX(lbound=lbound, ubound=ubound, integer=1, spacing=1, complete=1)
    if lbound != -inf:
        ret.llimit = 1
    if ubound != inf:
        ret.ulimit = 1
    if lbound == -inf and ubound == inf:
        ret.median = 0  # Dubious
    else:
        ret.median = (lbound + ubound) / 2
    ret.size = ubound - lbound + 1
    return ret

Z = ℤ = Z_interval(-inf, inf)
N = ℕ = Z_interval(1, inf)

#ℚ = ℤ / ℕ

asserteq(str(Z_interval(0, 1)), "{0, 1}")
asserteq(str(Z_interval(-1, 1)), "{-1, 0, 1}")
asserteq(str(Z_interval(-1, 2)), "{-1, 0, ..., 2}")
asserteq(str(Z_interval(2, inf)), "{2, 3, ...}")
asserteq(str(Z_interval(-inf, 1)), "{..., -∞+1, ..., 1}")
asserteq(str(Z_interval(1, inf)), "ℕ")
asserteq(str(Z_interval(-inf, inf)), "ℤ")
asserteq(Z_interval(-1, 2).size, 4)
asserteq(Z_interval(-inf, inf).integer, 1)
asserteq(ℤ, ℤ)
asserteq(ℤ.equal(ℕ), 0)
asserteq(ℤ.equal(ℕ), 0)
asserteq(ℤ + 2, ℤ)

def R_interval(lbound: float, ubound: float, llimit = True, ulimit = True):
    "A subinterval of [-∞, ∞]"
    ret = VecSetX(lbound=lbound, ubound=ubound, llimit=int(llimit), ulimit=int(ulimit), spacing=0, complete=1)
    if lbound == -inf and ubound == inf:
        ret.median = 0  # Dubious
    else:
        ret.median = (lbound + ubound) / 2
    if lbound == ubound:
        ret.size = 1   # Even for the interval [∞, ∞]?
        if math.isfinite(lbound):
            ret.integer = int(lbound == int(lbound))
    else:
        ret.size = inf
    return ret

asserteq(str(R_interval(0,1)), "[0, 1]")
asserteq(str(R_interval(0,1, True, False)), "[0, 1)")
asserteq(str(R_interval(0,1, False, True)), "(0, 1]")
asserteq(str(R_interval(-inf,1.5)), "[-∞, 1.5]")
asserteq(str(R_interval(-inf,inf, False, True)), "(-∞, ∞]")
asserteq(str(R_interval(-inf,inf, False, False)), "ℝ")
asserteq(str(R_interval(-inf,inf)), "[-∞, ∞]")
asserteq(str(R_interval(1.5,1.5)), "{1.5}")
asserteq(R_interval(1, 1).integer, 1)
asserteq(R_interval(1, 1).size, 1)
asserteq(R_interval(1.5, 1.5).integer, 0)
asserteq(R_interval(-inf, inf).integer, 0)

R = ℝ = R_interval(-inf, inf, False, False)

asserteq(ℝ.equal(ℝ), 1)
asserteq(ℝ, ℝ + 1)
asserteq(ℝ, ℝ * 2)
