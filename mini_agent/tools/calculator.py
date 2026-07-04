"""calculator tool: evaluate a basic arithmetic expression safely.

Uses an AST whitelist instead of ``eval`` so a hostile expression cannot execute
arbitrary code.
"""

from __future__ import annotations

import ast
import operator as op

from .registry import Tool, ToolContext

_BINOPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow,
}
_UNARYOPS = {ast.USub: op.neg, ast.UAdd: op.pos}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval(node.operand))
    raise ValueError(f"unsupported or unsafe expression element: {ast.dump(node)}")


def calculate(ctx: ToolContext, expression: str) -> dict:
    tree = ast.parse(expression, mode="eval")
    return {"expression": expression, "result": _eval(tree.body)}


CALCULATOR = Tool(
    name="calculator",
    description=(
        "Evaluate a basic arithmetic expression with + - * / // % ** and "
        "parentheses. Use this for any exact math instead of computing it yourself."
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression, e.g. '2 * (3 + 4) ** 2'",
            }
        },
        "required": ["expression"],
    },
    func=calculate,
)
