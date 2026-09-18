"""Calculatrice sûre : évaluation d'expressions arithmétiques via l'AST.

    calculator.execute("15 * 20")  → "300"
"""

from __future__ import annotations

import ast
import math
import operator

from .base import Tool

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    name: getattr(math, name)
    for name in ("sqrt", "sin", "cos", "tan", "log", "log2", "log10", "exp", "floor", "ceil", "fabs")
}
_FUNCS.update({"abs": abs, "round": round, "min": min, "max": max})
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exposant trop grand")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        if node.keywords:
            raise ValueError("arguments nommés non supportés")
        return _FUNCS[node.func.id](*(_eval(a) for a in node.args))
    raise ValueError(f"expression non autorisée : {ast.dump(node)[:40]}")


def safe_eval(expression: str) -> float:
    tree = ast.parse(expression.strip(), mode="eval")
    return _eval(tree)


class CalculatorTool(Tool):
    name = "calculator"
    description = "Évalue une expression arithmétique (ex: '15 * 20', 'sqrt(2) + pi')."

    def execute(self, input: str) -> str:
        expr = input.strip().replace("^", "**").replace(",", ".")
        if not expr:
            raise ValueError("expression vide")
        result = safe_eval(expr)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)
