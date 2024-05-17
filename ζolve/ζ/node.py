
from .vector import *

World = set()

class Node:
    def __init__(self, vec = None):
        self.args = []
        self.v = vec
        World.add(self)

class VarNode(Node):
    def __init__(self, name, vec):
        super().__init__(vec)
        self.name = name

def VariableX(name, integer = False, nat = False):
    "A real or integer valued variable."
    if nat:
        return VarNode(name, ℕ)
    elif integer:
        return VarNode(name, ℤ)
    else:
        return VarNode(name, ℝ)

def Variable(order = 0, **kwargs):
    if order == 0:
        return VariableX(**kwargs)
    else:
        assert False



