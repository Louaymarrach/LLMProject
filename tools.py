"""
tools.py
Custom tools required by the project:
- simple_calculator(expr): evaluate a math expression safely
- keyword_extractor(text): return top keywords
- duck_search(query): use DuckDuckGo search API
- mood_support_tool(text): LLM-based emotional support (non-clinical)
"""

import os
from typing import List
from dotenv import load_dotenv
import ast, operator, re, requests
from collections import Counter

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ---------------------------------------------------------
#   SAFE CALCULATOR TOOL
# ---------------------------------------------------------

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
        raise ValueError("Operator not allowed.")

    if isinstance(node, ast.UnaryOp):
        operand = safe_eval(node.operand)
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[op_type](operand)

    raise ValueError("Unsupported expression.")

def simple_calculator(expr: str) -> str:
    """Evaluate a safe math expression."""
    try:
        tree = ast.parse(expr, mode='eval')
        result = safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------
#   KEYWORD EXTRACTION TOOL
# ---------------------------------------------------------

def keyword_extractor(text: str, top_k: int = 8) -> List[str]:
    """Extract simple keywords by frequency."""
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    counts = Counter(tokens)
    common = [w for w, _ in counts.most_common(top_k)]
    return common


# ---------------------------------------------------------
#   DUCKDUCKGO SEARCH TOOL
# ---------------------------------------------------------

def duck_search(query: str, max_results: int = 5) -> List[str]:
    """
    Perform DuckDuckGo Instant Answer API search.
    Returns snippets only.
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
            return ["No results found."]

        return results

    except Exception as e:
        return [f"Error fetching search results: {e}"]


# ---------------------------------------------------------
#   LLM-BASED MOOD SUPPORT TOOL
# ---------------------------------------------------------

MOOD_LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  
    temperature=0.4,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

MOOD_SYSTEM_PROMPT = """
You are a gentle, non-clinical emotional support assistant for university students.

Your job:
- Read the student's text and understand their emotions (positive, negative, or mixed).
- Reflect their emotions in simple, warm language.
- Give 2–5 practical and supportive suggestions.
- NO medical labels, NO diagnoses.
- If self-harm or suicide is mentioned, focus on safety and encourage reaching out to trusted people and professionals.
- Use a kind, warm tone — like a caring senior student.

Always output one coherent message in plain English.
"""

def mood_support_tool(text: str) -> str:
    """
    LLM-based emotional support tool.
    Returns only the supportive message (metadata added in main_agent.py).
    """
    text = text.strip()
    if not text:
        return (
            "You can share a few words about how you're feeling if you'd like. "
            "I'll respond with gentle emotional support."
        )

    try:
        messages = [
            SystemMessage(content=MOOD_SYSTEM_PROMPT),
            HumanMessage(content=text),
        ]
        resp = MOOD_LLM.invoke(messages)
        return getattr(resp, "content", str(resp))

    except Exception:
        # Safe fallback
        return (
            "Thank you for sharing how you feel. I'm having some trouble generating a detailed response "
            "right now, but what you're feeling matters. Talking with someone you trust or a counselor "
            "at your university could really help."
        )


# ---------------------------------------------------------
#   TOOLSET MANAGER
# ---------------------------------------------------------

class Toolset:
    """Container for managing tools."""

    def __init__(self):
        self.tools = {
            "calculator": simple_calculator,
            "keywords": keyword_extractor,
            "duck_search": duck_search,
            "mood_support": mood_support_tool,
        }

    def add_tool(self, name: str, func):
        self.tools[name] = func

    def get_tool(self, name: str):
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())


# ---------------------------------------------------------
#   STANDALONE TEST
# ---------------------------------------------------------

if __name__ == '__main__':
    print("Calculator test:", simple_calculator("2+2*3"))
    print("Keywords test:", keyword_extractor("This project uses RAG, a calculator and keyword extractor."))
    print("Duck search test:", duck_search("latest AI news"))
    t = Toolset()
    print("Available tools:", t.list_tools())
