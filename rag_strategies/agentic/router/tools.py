import ast
import operator
import re

from core.models import ContextItem

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Num):  # pragma: no cover - py3.9
        return node.n
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Unsupported expression")


def calculator_context(query: str) -> ContextItem:
    expression = re.sub(r"[^0-9\.\+\-\*/\(\) ]", "", query.replace("x", "*").replace("×", "*").replace("÷", "/"))
    expression = expression.strip()
    try:
        value = _eval_node(ast.parse(expression, mode="eval"))
        content = f"Calculator result for `{expression}` is {value}."
    except Exception:
        content = "The calculator tool could not parse a safe arithmetic expression from the query."
    return ContextItem(content=content, source="tool", source_id="calculator", metadata={"tool": "calculator"})


def simulated_web_context(query: str) -> ContextItem:
    return ContextItem(
        content="Simulated web tool: this laboratory does not call the public internet. Use uploaded documents for grounded answers.",
        source="tool",
        source_id="simulated_web",
        score=0.0,
        metadata={"tool": "simulated_web", "query": query},
    )
