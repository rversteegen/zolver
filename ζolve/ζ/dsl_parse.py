#!/usr/bin/env python3

import re
import ast
from tokenize import NAME, OP
from typing import List

#import sys
#sys.path = ["/mnt/common/src/sympy"] + sys.path

import sympy
import sympy.parsing.ast_parser
from sympy.parsing import sympy_parser as spparser
from sympy.parsing.sympy_parser import TOKEN, DICT

import ζ.dsl
import ζ.solver
import ζ.util

global_namespace = {}
exec('from ζ.dsl import *', global_namespace)

SECONDS_LIMIT = 6
MB_LIMIT = 500  # Minimum needed is about 185


class DSLError(Exception):
    pass


################################################################################
### Python AST manipulation

AST_DEBUG = False

def makeCall(name, args = [], keywords = [], copyloc = None):
    "Create an ast.Call node"
    ret = ast.Call(ast.Name(name, ast.Load()), args, keywords)
    if copyloc:
        ret = ast.copy_location(ret, copyloc)
    return ast.fix_missing_locations(ret)


class ASTTransform(sympy.parsing.ast_parser.Transform):
    def visit_Name(self, node):
        "Modified version of Transform.visit_Name to remove auto-Symbol'ing"
        if node.id in self.local_dict:
            return node
        elif node.id in self.global_dict:
            name_obj = self.global_dict[node.id]

            if isinstance(name_obj, (sympy.Basic, type)) or callable(name_obj):
                return node
        elif node.id in ['True', 'False']:
            return node
        #return ast.fix_missing_locations(
        #    ast.Call(func=ast.Name('Symbol', ast.Load()),
        #             args=[ast.Constant(node.id)], keywords=[]))
        return node

    def visit_BoolOp(self, node):
        if AST_DEBUG:
            print("Bool:", ast.dump(node))
        op = node.op.__class__.__name__   # Or or And
        ret = makeCall(op, node.values, copyloc=node)
        return self.generic_visit(ret)  # Recurse

    def visit_UnaryOp(self, node):
        "Handle not -> Not"
        if AST_DEBUG:
            print("Unary:", ast.dump(node))
        if isinstance(node.op, ast.Not):
            ret = makeCall('Not', [node.operand], copyloc = node)
        else:
            ret = node
        return self.generic_visit(ret)  # Recurse

    def visit_Compare(self, node):
        """Convert 'a == b' to Eq(a, b), 'a in b' to Element(a, b) and split up
        multiple comparisons like a == b == 3 to And(Eq(a,b), Eq(b,3)) because
        sympy.Eq and other relational ops are binary only.
        """
        if AST_DEBUG:
            print("Compare:", ast.dump(node))
        comps = []
        args = [node.left] + node.comparators
        for idx, op in enumerate(node.ops):
            left, right = args[idx], args[idx+1]
            if isinstance(op, ast.Eq):
                newcomp = makeCall('Eq', [left, right])
            elif isinstance(op, ast.In):
                newcomp = makeCall('Element', [left, right])
                #assert len(args) == 2
            else:
                newcomp = ast.Compare(left, [op], [right])
            comps.append(ast.copy_location(newcomp, node))
        if len(comps) > 1:
            # Join together with And
            ret = makeCall('And', comps)
        else:
            ret = comps[0]
        return self.generic_visit(ret)  # Recurse

    def visit_AnnAssign(self, node):
        """Convert 'x : Y' to 'variable("x", Y)' so we can record the variable name. Y can be any expression."""
        # TODO: Consider allowing "f(x) : Int" -> "f : Function([x.var_type], Int)"
        assert isinstance(node.target, ast.Name)
        ret = makeCall('declarevar', [ast.Constant(node.target.id), node.annotation], copyloc =  node)
        # The annotation is toplevel, so have to put into a Expr
        ret = ast.Expr(ret)
        ret = ast.copy_location(ret, node)
        return ret  #ast.fix_missing_locations(ret)


# Modified copy of sympy.parsing.ast_parser.parse_expr
def parse_and_eval_dsl(string, local_dict, global_dict = global_namespace, mode = "eval"):
    """
    Parses the string "string" and convert python expressions to SymPy expressions
    and handle ζ-specific DSL syntax, and then executes it!

    mode: 'eval' for an expression, 'exec' for a line/statement.
    Evaluating a line modifies local_dict.

    It converts all numbers to Integers before feeding it to Python and
    <s>automatically creates Symbols</s>.
    """
    try:
        a = ast.parse(string.strip(), mode=mode)
        if AST_DEBUG:
            print("ast.parse result: ", ast.dump(a, indent=4, include_attributes = True))
    except SyntaxError as e:
        #raise sympy.SympifyError("Cannot parse %s." % repr(string))
        raise SyntaxError(f"{e}:  '{string}'")
    a = ASTTransform(local_dict, global_dict).visit(a)
    if AST_DEBUG:
        print("REWRITTEN:\n", ast.dump(a, indent=4, include_attributes = True))
    e = compile(a, "<string>", mode)
    try:
        with ζ.util.time_limit(SECONDS_LIMIT):
            with ζ.util.memory_limit(MB_LIMIT * 2**20):
                return eval(e, global_dict, local_dict)
    except Exception as e:
        raise DSLError(f"DSL script threw {type(e).__name__} '{str(e).strip()}'  on line  '{string}'")

#parse_expr("  x : Point() >= 4\nprint(__annotations__)", {'Point': lambda p=3: 42}, global_namespace, mode = "exec")
#parse_expr("x,y : int", {'Point': lambda p=3: 42}, global_namespace, mode = "exec")
#parse_expr("f(x) : Int", {'Point': lambda p=3: 42}, global_namespace, mode = "exec")
# exit()

################################################################################
### UNUSED
# Customised transformation for sympy.parse_expr()

# This is a modified copy of sympy_parser._transform_equality_sign, replacing = -> ==
def _transform_equality_sign(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    """Transforms the equality sign ``==`` to instances of Eq.

    This is a helper function for ``convert_equality_signs``.
    Works with expressions containing one equality sign and no
    nesting. Expressions like ``(1==2)==False`` will not work with this
    and should be used with ``convert_equality_signs``.

    Examples: 1==2    to Eq(1,2)
              1*2==x  to Eq(1*2, x)
    """
    result: List[TOKEN] = []
    if (OP, "==") in tokens:
        result.append((NAME, "Eq"))
        result.append((OP, "("))
        for token in tokens:
            if token == (OP, "=="):
                result.append((OP, ","))
                continue
            result.append(token)
        result.append((OP, ")"))
    else:
        result = tokens
    return result


# This is a modified copy of sympy_parser.convert_equals_signs, replacing = -> ==
def convert_equality_signs(tokens: List[TOKEN], local_dict: DICT,
                         global_dict: DICT) -> List[TOKEN]:
    """ Transforms all the equality signs ``==`` to instances of Eq.

    Parses the equality signs in the expression and replaces them with
    appropriate Eq instances. Also works with nested equality signs.

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import (parse_expr,
    ... standard_transformations, convert_equality_signs)
    >>> parse_expr("1*2==x", transformations=(
    ... standard_transformations + (convert_equality_signs,)))
    Eq(2, x)
    >>> parse_expr("(1*2==x)==False", transformations=(
    ... standard_transformations + (convert_equality_signs,)))
    Eq(Eq(2, x), False)

    """
    res1 = spparser._group_parentheses(convert_equality_signs)(tokens, local_dict, global_dict)
    res2 = spparser._apply_functions(res1, local_dict, global_dict)
    res3 = _transform_equality_sign(res2, local_dict, global_dict)
    result = spparser._flatten(res3)
    return result

spp = spparser
# standard_transformations but without auto_symbol
transformations = [spp.lambda_notation, spp.repeated_decimals, spp.auto_number, spp.factorial_notation]#, 
transformations = [convert_equality_signs]
del spp

################################################################################

def load_dsl(script, verbose = True):
    "`script` is the text of a ζ DSL script, of multiple lines. Parsed and executed."

    # Ugly but difficult to avoid
    workspace = ζ.solver.Workspace()
    ζ.dsl._ctx = workspace

    namespace = workspace.locals

    for line in script.split('\n'):

        # Remove comments
        line = line.split('#')[0]
        line = line.strip()
        if not line:
            continue

        # Special case, allow multiple variables in a type annotation
        m = re.match("(\w[ ,])+:(.*)$", line)
        if m:
            vars = m.group(1).split(',')
            annot = m.group(2)
            line = '\n'.join(v + " : " + annot for v in vars)

        if 'variable(' in line: # or 'seq(' in line:
            # Obsolete

            #line = re.sub('(\w+)\s*=\s*(\w+)\(', '\\1 = \\2("\\1", ', line)
            line = re.sub('(\w+)\s*=\s*(\w+)\(', '\\2("\\1", ', line)
            # Fix variable() => variable(x, )... doesn't matter.
            #line = line.replace(', )', ')')

            #var = eval(line, namespace)
            # Equivalently
            #sympy.sympify(line, namespace)
            if verbose: print("VARLINE: ", line)

            var = sympy.parse_expr(line, namespace, transformations = transformations)

            print(f"...adding : '{var.name}' = {var.func}({var.args})")
            namespace[var.name] = var
            continue

        line = re.sub('goal\s*=\s*(.*)$', 'goal(\\1)', line)

        if verbose: print(f"LINE: {repr(line)}")

        # Parse the func

        func = None
        match = re.match('\s*(\w+)\s*\((.*)\)$', line)
        if match:
            # Unused
            func = match.group(1)
            args = match.group(2)
            #print(f":func <{func}> args <{args}>")

        # Exponents. Could also do this with sympy.parse_expr or parse_and_eval_dsl
        line = line.replace('^', '**')


        # if line.startswith('constraint('):
        #     assert(line.endswith(')'))
        #     line = line.replace('constraint(', '')[:-1]

        # TODO, auto add/remove )'s at end

        #res = sympy.sympify(line, namespace)


        #res = sympy.parse_expr(line, namespace, transformations = transformations)

        res = parse_and_eval_dsl(line, namespace, global_namespace, "exec")

        if verbose: print(" eval->", res)
        #output.append(res)

    return workspace

def load_dsl_file(path):
    with open(path) as f:
        return load_dsl(f.read())


################################################################################


if __name__ == '__main__':


    intext="""
    b = variable(integer = True)
    constraint(not (b == 3 or b == 1))
    constraint(71 <= 10 * b + 3 * a <= 75)

    a = seq(length = 2005)
    constraint(a[1] == a[3] == 2005)
    #goal = a[b]
    goal = min(a)
    """

    intext = """
    b = variable(integer = True)
    r = variable(integer = True)
    g = variable(integer = True)
    c = variable(integer = True)
    constraint(b + r + g + c == 280)
    constraint(r == 2 * b)
    constraint(g == 3 * c)
    constraint(c == 4 * r)
    goal = c
    """

    intext = """
a : int
b : int
c : int
constraint(a + b + c == 3)
constraint(a**2 + b**2 + c**2 == 5)
constraint(a**3 + b**3 + c**3 == 9)
goal = a**4 + b**4 + c**4
    """

    intext = """
# We can either use a single sequence of two digits or a pair of variables (x, y). A sequence is easier for arithmetic
S : Seq(limit = 2)
i : Int
# The sequence must contain two distinct digits from the set {2, 4, 6, 7}
ForAll([i], i in S, (i in {2, 4, 6, 7} and (i[i-1] in S and i[i-1] < i)))
# We want to count the number of possible assignments to S
goal = count(S)
"""

    workspace = load_dsl(intext)
    workspace.print()

    if False:
        print(sympy.solve(workspace.facts, [workspace.goal]))
    else:
        workspace.solve()

