"""
tools.py
Custom tools required by the project:
- simple_calculator(expr): evaluate a math expression safely
- keyword_extractor(text): return top keywords (simple heuristic)
- duck_search(query): use DuckDuckGo search API to get quick web results
"""

from typing import List
import ast, operator, re, requests
from collections import Counter

# Basic Calculator 
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def safe_eval(node):
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.BinOp):
        left = safe_eval(node.left)
        right = safe_eval(node.right)
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[op_type](left, right)
        raise ValueError("Operator not allowed")
    if isinstance(node, ast.UnaryOp):
        operand = safe_eval(node.operand)
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[op_type](operand)
    raise ValueError("Unsupported expression")

def simple_calculator(expr: str) -> str:
    """Evaluate a simple math expression safely and return the result or error."""
    try:
        tree = ast.parse(expr, mode='eval')
        result = safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

# ------------------ Keyword Extractor ------------------
def keyword_extractor(text: str, top_k: int = 8) -> List[str]:
    """Simple keyword extractor: tokenize, remove short words, rank by frequency."""
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    counts = Counter(tokens)
    common = [w for w, _ in counts.most_common(top_k)]
    return common

# ------------------ DuckDuckGo Search ------------------
def duck_search(query: str, max_results: int = 5) -> List[str]:
    """
    Perform a quick DuckDuckGo search and return top result snippets.
    This uses the free DuckDuckGo Instant Answer API (no API key needed).
    """
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        results = []
        if data.get("AbstractText"):
            results.append(data["AbstractText"])
        for topic in data.get("RelatedTopics", []):
            if isinstance(topic, dict) and "Text" in topic:
                results.append(topic["Text"])
            if len(results) >= max_results:
                break

        if not results:
            results.append("No results found.")
        return results
    except Exception as e:
        return [f"Error fetching search results: {e}"]

# ------------------ Toolset Manager ------------------
class Toolset:
    """
    Simple container for managing callable tools.
    Allows registration and retrieval by name.
    """

    def __init__(self):
        self.tools = {
            "calculator": simple_calculator,
            "keywords": keyword_extractor,
            "duck_search": duck_search,
        }

    def add_tool(self, name: str, func):
        self.tools[name] = func

    def get_tool(self, name: str):
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())


# ------------------ Standalone Test ------------------
if __name__ == '__main__':
    print("Calculator test:", simple_calculator('2+2*3'))
    print("Keywords test:", keyword_extractor('This project implements RAG, a calculator tool and a keyword extractor tool.'))
    print("Duck search test:", duck_search('latest AI news'))
    t = Toolset()
    print("Available tools:", t.list_tools())
