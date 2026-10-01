from dataclasses import dataclass
import re
from typing import Optional


@dataclass
class RouterDecision:
    route: str
    reason: str
    confidence: float
    tool: Optional[str] = None


_MATH = re.compile(r"(\d+\s*[\+\-\*/x×÷]\s*\d+)|(\bwhat is\b.+\d+)|(\bcalculate\b)|(\bcompute\b)", re.I)
_MEMORY = re.compile(r"\b(remember|i prefer|my preference|you know that i)\b", re.I)
_WEB = re.compile(r"\b(search the web|look up online|latest news|according to the internet)\b", re.I)
_DIRECT = re.compile(r"^\s*(hi|hello|hey|thanks|thank you)\b", re.I)
_DOC = re.compile(r"\b(document|chunk|paper|pdf|according to|in the text|retrieved|source)\b", re.I)


class HeuristicRouter:
    """Offline router. Returns a structured decision, never hidden chain-of-thought."""

    def decide(self, query: str) -> RouterDecision:
        text = query.strip()
        if _MATH.search(text):
            return RouterDecision("tool", "The question looks like arithmetic, so a calculator tool is enough.", 0.86, tool="calculator")
        if _WEB.search(text):
            return RouterDecision("tool", "The question asks for an external lookup. Only a simulated tool is available in this lab.", 0.72, tool="simulated_web")
        if _MEMORY.search(text):
            return RouterDecision("memory", "The question refers to stored preferences or prior user facts.", 0.78)
        if _DIRECT.search(text) and len(text.split()) < 8:
            return RouterDecision("direct", "This is a greeting or short social turn that does not need retrieval.", 0.9)
        if _DOC.search(text):
            return RouterDecision("rag", "The question asks about uploaded document content.", 0.91)
        return RouterDecision("rag", "Default laboratory route: retrieve from the indexed corpus.", 0.64)
