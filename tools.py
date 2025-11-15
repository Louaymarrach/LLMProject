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
    
# ------------------ Mental Health Support: Mood Classifier ------------------

MOOD_KEYWORDS = {
    "positive": [
        "love", "loving", "like", "enjoy", "enjoying", "happy", "excited",
        "motivated", "grateful", "good", "great", "proud", "confident", "amazing"
    ],
    "anxious": [
        "anxious", "anxiety", "panicking", "panic", "worried",
        "worry", "nervous", "scared", "afraid", "tense"
    ],
    "stressed": [
        "stressed", "stress", "overwhelmed", "pressure", "burned out",
        "burnt out", "too much work", "can’t handle", "cant handle"
    ],
    "sad": [
        "sad", "down", "upset", "low", "crying", "cry", "tears",
        "hurt", "empty", "hopeless", "useless", "worthless"
    ],
    "angry": [
        "angry", "mad", "furious", "frustrated", "annoyed", "pissed"
    ],
    "tired": [
        "tired", "exhausted", "drained", "no energy", "fatigued",
        "sleepy", "can’t sleep", "cant sleep", "insomnia"
    ],
    "lonely": [
        "lonely", "alone", "isolated", "no one", "nobody", "left out"
    ]
}


def classify_mood(text: str) -> str:
    """
    Very simple, non-clinical mood classifier based on keyword matches.
    Returns one of:
        'positive', 'anxious', 'stressed', 'sad',
        'angry', 'tired', 'lonely', or 'mixed'.
    """
    text_lower = text.lower()
    scores = {}

    for mood, keywords in MOOD_KEYWORDS.items():
        count = 0
        for kw in keywords:
            if kw in text_lower:
                count += 1
        scores[mood] = count

    best_mood = max(scores, key=scores.get)
    if scores[best_mood] == 0:
        return "mixed"
    return best_mood


def mood_support_tool(text: str) -> str:
    """
    Take a student's free-text description and return a gentle, supportive message.
    This is NOT a medical or diagnostic tool – just general emotional support.
    Works for both positive and negative feelings.
    """
    mood = classify_mood(text)

    # ----- Intro depends on mood -----
    if mood == "positive":
        intro = (
            "That’s wonderful to hear! I’m really glad you’re feeling good about things right now.\n\n"
        )
    elif mood in {"anxious", "stressed", "sad", "angry", "tired", "lonely"}:
        intro = (
            "Thank you for sharing how you feel. It’s not always easy to put this into words, "
            "and it’s completely okay to feel this way.\n\n"
        )
    else:  # mixed / unclear
        intro = (
            "Thank you for sharing a bit of how you feel. It sounds like there’s a lot on your mind.\n\n"
        )

    # ----- Main suggestions by mood -----
    if mood == "positive":
        body = (
            "It sounds like you’re feeling **positive, motivated, or happy**.\n\n"
            "Here are a few ideas to help you keep that momentum:\n"
            "- Keep doing the activities and habits that energize you.\n"
            "- Stay consistent with the routines that are supporting your success.\n"
            "- Take a moment to appreciate the progress you’re making this semester.\n"
        )
    elif mood == "anxious":
        body = (
            "It sounds like you might be feeling **anxious or worried**.\n\n"
            "Here are a few gentle things you can try:\n"
            "- Take slow breaths: inhale for 4 seconds, hold for 4, exhale for 6.\n"
            "- Break your tasks into very small steps and focus on just one at a time.\n"
            "- Step away from your screen for a few minutes, stretch, or get some fresh air.\n"
        )
    elif mood == "stressed":
        body = (
            "It sounds like you might be feeling **stressed or overwhelmed**.\n\n"
            "You could try:\n"
            "- Writing down everything you need to do, then picking one small task to start with.\n"
            "- Using short focused study blocks (for example, 25 minutes work + 5 minutes break).\n"
            "- Giving yourself permission to rest without feeling guilty – rest also helps you succeed.\n"
        )
    elif mood == "sad":
        body = (
            "It sounds like you might be feeling **sad or low**.\n\n"
            "Some gentle ideas that might help a little:\n"
            "- Reach out to someone you trust and tell them how you’re feeling, even in a short message.\n"
            "- Do one small thing you usually enjoy (music, a walk, drawing, journaling).\n"
            "- Remind yourself that heavy feelings can be strong, but they are not permanent.\n"
        )
    elif mood == "angry":
        body = (
            "It sounds like you might be feeling **angry or frustrated**.\n\n"
            "You might try:\n"
            "- Taking a few minutes away from the situation, if possible, before reacting.\n"
            "- Writing down what’s bothering you without judging yourself.\n"
            "- Doing something physical but safe, like a brisk walk, to release some of the tension.\n"
        )
    elif mood == "tired":
        body = (
            "It sounds like you might be feeling **tired or drained**.\n\n"
            "Some small steps that might help:\n"
            "- Notice if you’ve been sleeping too little or studying for very long periods without breaks.\n"
            "- Drink some water and have a light snack if you haven’t eaten in a while.\n"
            "- Allow yourself short rest blocks and avoid endless scrolling, which can make you feel more tired.\n"
        )
    elif mood == "lonely":
        body = (
            "It sounds like you might be feeling **lonely or left out**.\n\n"
            "A few gentle ideas:\n"
            "- Message a friend or classmate, even just to say hello or ask about a course.\n"
            "- Join a small activity, club, or study group where you can see familiar faces regularly.\n"
            "- Remind yourself that feeling lonely doesn’t mean you are unimportant – you matter more than you think.\n"
        )
    else:  # mixed / unclear
        body = (
            "I’m not completely sure what you’re feeling, but it does sound like things are not very easy right now.\n\n"
            "Some general things that can sometimes help:\n"
            "- Take a short break away from screens and breathe slowly for a minute.\n"
            "- Write down how you feel in a few sentences – sometimes putting it on paper brings clarity.\n"
            "- Reach out to someone you trust and tell them you’re going through a tough moment.\n"
        )

    # ----- Closing depends on mood -----
    if mood == "positive":
        closing = (
            "\nKeep nurturing that positive energy – staying motivated and kind to yourself is one of the best ways "
            "to have a great semester. And if you ever go through a stressful or heavy moment later on, "
            "remember it’s okay to ask for support from friends, family, or professionals.\n"
        )
    else:
        closing = (
            "\nIf your feelings become very intense, last for a long time, or you ever feel unsafe, "
            "it’s really important to reach out to a mental health professional, a counselor, or a trusted person "
            "at your university or in your life. You don’t have to go through hard moments alone.\n"
        )

    return intro + body + closing

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
            "mood_support": mood_support_tool,
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
