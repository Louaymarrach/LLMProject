"""
tools.py

Provides tools for:
 - duck_search(query): quick web instant-answer via DuckDuckGo
 - ingest_documents(file_paths): ingest PDFs/TXTs into a FAISS vectorstore for RAG
 - summarize_document(file_path): short summary via the LLM
 - pdf_qa(question): RAG-style QA using stored vectorstore
 - mood_support(text): lightweight, safe non-clinical emotional support helper
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# HTTP requests
import requests

# PDF reading
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# Text splitting (langchain)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    # raise error later if missing
    RecursiveCharacterTextSplitter = None

# Vectorstore and Embeddings (try several import paths as different projects use different names)
FAISS = None
try:
    # modern langchain
    from langchain.vectorstores import FAISS
    FAISS = FAISS
except Exception:
    try:
        from langchain_community.vectorstores import FAISS as FAISS2
        FAISS = FAISS2
    except Exception:
        FAISS = None  # check later and raise clearer error

# Embeddings & chat
GoogleGenerativeAIEmbeddings = None
ChatGoogleGenerativeAI = None
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
except Exception:
    # if not available, leave None
    GoogleGenerativeAIEmbeddings = None
    ChatGoogleGenerativeAI = None

# -----------------------
# Configuration / env
# -----------------------
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "./vectorstore")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "gemini-2.5-flash")
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
DUCK_DDG_API = "https://api.duckduckgo.com/"

# -----------------------
# Toolset class
# -----------------------
class Toolset:
    """
    Simple registry for tools. Instantiate and register functions as named tools.
    """
    def __init__(self):
        self.tools = {}
        # Register builtin tools
        self.register("duck", duck_search)
        self.register("ingest", ingest_documents)
        self.register("summarize", summarize_document)
        self.register("pdf_qa", pdf_qa)
        # Register mood support tool and aliases so AgentInterface finds them
        self.register("mood_support", mood_support)
        self.register("mood", mood_support)
        self.register("emotion_support", mood_support)

    def register(self, name: str, fn):
        self.tools[name] = fn

    def get_tool(self, name: str):
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())

# -----------------------
# DuckDuckGo instant answer tool
# -----------------------
def duck_search(query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Use DuckDuckGo Instant Answer API to fetch the abstract and related topics.
    Returns a dict with 'query', 'abstract', and 'related' topics.
    """
    if not query or not isinstance(query, str):
        return {"error": "Query must be a non-empty string."}

    qparams = {
        "q": query,
        "format": "json",
        "no_redirect": "1",
        "no_html": "1",
        "skip_disambig": "1",
    }
    if params:
        qparams.update(params)
    try:
        resp = requests.get(DUCK_DDG_API, params=qparams, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        # extract useful fields
        result = {
            "query": query,
            "abstract": data.get("AbstractText") or data.get("Heading") or "",
            "abstract_url": data.get("AbstractURL") or "",
            "related_topics": [],
            "raw": data,
        }
        # related topics
        rtopics = data.get("RelatedTopics", [])
        for rt in rtopics[:8]:
            if isinstance(rt, dict):
                if rt.get("Text"):
                    result["related_topics"].append({"text": rt.get("Text"), "url": rt.get("FirstURL")})
                elif rt.get("Name") and rt.get("Topics"):
                    # grouped topics
                    for t in rt.get("Topics", [])[:4]:
                        result["related_topics"].append({"text": t.get("Text"), "url": t.get("FirstURL")})
        return result
    except Exception as e:
        return {"error": f"duck_search failed: {e}", "query": query}

# -----------------------
# Helpers for file extraction
# -----------------------
def _extract_text_from_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found")
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        if PyPDF2 is None:
            raise RuntimeError("PyPDF2 is required to read PDFs. Install with `pip install PyPDF2`.")
        text_parts = []
        with open(p, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for pg in reader.pages:
                try:
                    t = pg.extract_text() or ""
                except Exception:
                    t = ""
                text_parts.append(t)
        return "\n".join(text_parts).strip()
    else:
        # handle .txt and other text files
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return p.read_text(encoding="latin-1")

# -----------------------
# Embedding / LLM factories
# -----------------------
def _get_embeddings():
    if GoogleGenerativeAIEmbeddings is not None:
        return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    # Fallback hint
    raise RuntimeError("No embedding implementation found. Install or configure GoogleGenerativeAIEmbeddings or adjust _get_embeddings().")

def _get_llm_for_summarize():
    if ChatGoogleGenerativeAI is not None:
        return ChatGoogleGenerativeAI(model=SUMMARIZER_MODEL, temperature=0.1)
    raise RuntimeError("No Chat LLM found. Install langchain_google_genai or adjust _get_llm_for_summarize().")

# -----------------------
# Ingest documents -> FAISS vectorstore
# -----------------------
def ingest_documents(file_paths: List[str]) -> Dict[str, Any]:
    """
    Ingest files (list of local paths) into a FAISS vectorstore stored at VECTORSTORE_DIR.
    - Splits text into chunks using RecursiveCharacterTextSplitter
    - Embeds chunks with _get_embeddings()
    - Creates or updates FAISS index
    Returns a dict with status and info.
    """
    # Basic validation
    if not isinstance(file_paths, (list, tuple)) or not file_paths:
        return {"status": "error", "error": "Provide a non-empty list of file paths."}

    if RecursiveCharacterTextSplitter is None:
        return {"status": "error", "error": "RecursiveCharacterTextSplitter not available. Install langchain text_splitter."}

    if FAISS is None:
        return {"status": "error", "error": "FAISS vectorstore not available. Install faiss and langchain vectorstores."}

    embeddings = None
    try:
        embeddings = _get_embeddings()
    except Exception as e:
        return {"status": "error", "error": f"Embeddings init failed: {e}"}

    splitter = RecursiveCharacterTextSplitter(chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP)

    texts = []
    metadatas = []
    report = {"files": []}

    for fp in file_paths:
        try:
            txt = _extract_text_from_file(fp)
            if not txt or not txt.strip():
                report["files"].append({"file": fp, "status": "no_text"})
                continue
            chunks = splitter.split_text(txt)
            fname = os.path.basename(fp)
            for i, c in enumerate(chunks):
                texts.append(c)
                metadatas.append({"source": fname, "chunk": i})
            report["files"].append({"file": fp, "chunks": len(chunks)})
        except Exception as e:
            report["files"].append({"file": fp, "error": str(e)})

    if not texts:
        return {"status": "no_texts", "report": report}

    # create vectorstore dir if missing
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    try:
        # Try to load existing and add, else create new
        vs = None
        try:
            vs = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
        except Exception:
            vs = None

        if vs is None:
            vs = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        else:
            # Some FAISS wrappers support add_texts
            try:
                vs.add_texts(texts, metadatas=metadatas)
            except Exception:
                new_vs = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
                # Overwrite
                vs = new_vs

        vs.save_local(VECTORSTORE_DIR)
        return {"status": "ok", "report": report, "vectorstore_dir": VECTORSTORE_DIR}
    except Exception as e:
        return {"status": "error", "error": f"Failed to build/save vectorstore: {e}", "report": report}

# -----------------------
# Summarize a single document
# -----------------------
def summarize_document(file_path: str, max_chars: int = 1200) -> Dict[str, Any]:
    """
    Read the file, take a portion (to limit token usage), and ask the LLM to summarize
    into 5-8 bullet points + 1-line TL;DR.
    """
    try:
        text = _extract_text_from_file(file_path)
    except Exception as e:
        return {"status": "error", "error": f"File read error: {e}", "file": file_path}

    if not text:
        return {"status": "error", "error": "No text extracted", "file": file_path}

    # Shorten text to keep the prompt reasonable
    excerpt = text[: max_chars * 2]  # take some buffer
    if len(excerpt) > max_chars:
        # try cut at last sentence break
        cut = excerpt.rfind(".", 0, max_chars)
        if cut == -1:
            excerpt = excerpt[:max_chars]
        else:
            excerpt = excerpt[: cut + 1]

    prompt = (
        "You are a concise summarizer for university students.\n"
        "Summarize the following document in 5-8 bullet points (each 1-2 short sentences), then a 1-line TL;DR.\n"
        "Use simple, direct language and include the most important facts, definitions, or conclusions.\n\n"
        f"Document excerpt:\n{excerpt}\n\nSummary:\n"
    )

    try:
        llm = _get_llm_for_summarize()
    except Exception as e:
        return {"status": "error", "error": f"LLM init failed: {e}"}

    # Different Chat wrappers expose different call signatures. Try a few.
    summary_text = None
    try:
        # prefer message interface if available
        # build simple messages list if Chat object supports it
        if hasattr(llm, "invoke"):
            # try using invoke with a simple system/human pattern
            try:
                # some wrappers accept list-of-messages dicts or custom message objects
                resp = llm.invoke([{"role": "system", "content": "You summarize documents."}, {"role": "user", "content": prompt}])
                # resp could be a string, object, or with .content
                summary_text = getattr(resp, "content", None) or (resp[0].get("content") if isinstance(resp, (list, tuple)) and resp else None) or str(resp)
            except Exception:
                # fallback to direct string invoke
                resp = llm.invoke(prompt)
                summary_text = getattr(resp, "content", None) or str(resp)
        elif hasattr(llm, "generate"):
            # langchain chat models
            try:
                out = llm.generate([{"role": "user", "content": prompt}])
                # try to parse
                if hasattr(out, "generations"):
                    gens = out.generations[0][0]
                    summary_text = getattr(gens, "text", None) or str(gens)
                else:
                    summary_text = str(out)
            except Exception:
                # try call
                resp = llm(prompt)
                summary_text = getattr(resp, "content", None) or str(resp)
        else:
            # direct call
            resp = llm(prompt)
            summary_text = getattr(resp, "content", None) or str(resp)
    except Exception as e:
        return {"status": "error", "error": f"LLM call failed: {e}"}

    return {"status": "ok", "file": file_path, "summary": summary_text}

# -----------------------
# PDF / document QA tool using the saved vectorstore
# -----------------------
def pdf_qa(question: str, k: int = RAG_TOP_K) -> Dict[str, Any]:
    """
    Ask a question; retrieve top-k chunks from FAISS vectorstore and produce an answer
    (LLM used to synthesize the answer and include source snippets).
    """
    if not question or not isinstance(question, str):
        return {"status": "error", "error": "Question must be a non-empty string."}

    if FAISS is None:
        return {"status": "error", "error": "FAISS not available. Install faiss and langchain vectorstore."}

    # ensure vectorstore exists
    if not os.path.exists(VECTORSTORE_DIR):
        return {"status": "error", "error": "No vectorstore found. Ingest documents first."}

    try:
        embeddings = _get_embeddings()
    except Exception as e:
        return {"status": "error", "error": f"Embeddings init failed: {e}"}

    try:
        vs = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        return {"status": "error", "error": f"Failed to load vectorstore: {e}"}

    # Use as retriever if wrapping provides it
    retriever = None
    try:
        retriever = vs.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(question)
    except Exception:
        # fallback: do a similarity search query directly if available
        try:
            docs = vs.similarity_search(question, k=k)
        except Exception as e:
            return {"status": "error", "error": f"Failed to retrieve docs: {e}"}

    # Build context from top docs
    context_pieces = []
    sources = []
    for i, d in enumerate(docs):
        txt = getattr(d, "page_content", None) or getattr(d, "text", None) or str(d)
        meta = getattr(d, "metadata", None) or {}
        src = meta.get("source", f"doc_{i}")
        sources.append({"source": src, "metadata": meta})
        # Keep excerpts short
        excerpt = txt[:1000]
        context_pieces.append(f"Source: {src}\nExcerpt: {excerpt}")

    context_str = "\n\n".join(context_pieces)

    prompt = (
        "You are an expert assistant. Use the provided context excerpts from course documents to answer the user's question.\n"
        "Cite sources by their filenames shown in 'Source:'. If the answer is not found in the context, say so honestly.\n\n"
        f"Context:\n{context_str}\n\nQuestion:\n{question}\n\nAnswer concisely and include source citations.\n"
    )

    try:
        llm = _get_llm_for_summarize()
    except Exception as e:
        return {"status": "error", "error": f"LLM init failed: {e}"}

    answer_text = None
    try:
        if hasattr(llm, "invoke"):
            resp = llm.invoke([{"role": "system", "content": "You answer based on provided context."}, {"role": "user", "content": prompt}])
            answer_text = getattr(resp, "content", None) or str(resp)
        else:
            resp = llm(prompt)
            answer_text = getattr(resp, "content", None) or str(resp)
    except Exception as e:
        return {"status": "error", "error": f"LLM call failed: {e}"}

    return {
        "status": "ok",
        "question": question,
        "answer": answer_text,
        "sources": sources[:k],
        "context_excerpt": context_pieces
    }

# -----------------------
# Mood Support tool
# -----------------------
def mood_support(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Lightweight mood support helper.

    Returns:
      (message: str, metadata: dict)

    Behavior:
      - Detects crisis/self-harm language and returns immediate, concise emergency guidance + metadata flag.
      - Otherwise uses keyword heuristics to classify mood (positive/neutral/negative) and returns an empathetic,
        practical short response with coping suggestions.
      - Metadata includes detected_mood, confidence, and optional flags.
    """
    # Basic input validation
    if not text or not isinstance(text, str) or not text.strip():
        return (
            "I didn't quite catch that. Could you tell me in a sentence how you're feeling right now?",
            {"error": "empty_input", "detected_mood": "unknown", "confidence": 0.0},
        )

    txt = text.strip()
    lower = txt.lower()

    # ---- Crisis / self-harm detection ----
    crisis_indicators = [
        "kill myself", "i want to die", "i want to end my life", "suicide",
        "i'm going to kill myself", "i will kill myself", "end my life", "i can't go on",
        "cant go on", "can't go on", "i might hurt myself", "hurt myself", "kill me"
    ]
    for phrase in crisis_indicators:
        if phrase in lower:
            # Immediate crisis advice - be concise and action oriented.
            message = (
                "I'm really sorry you're feeling this way. If you are in immediate danger or think you might act on these thoughts, "
                "please call your local emergency number right now.\n\n"
                "If you can, please consider telling someone nearby that you need urgent help.\n\n"
                "If you'd like, tell me your country and I can provide local crisis helpline numbers and resources. "
                "You don't have to go through this alone."
            )
            meta = {
                "detected_mood": "crisis",
                "severity": "high",
                "flag": "suicide_or_self_harm",
                "confidence": 0.99,
            }
            return (message, meta)

    # ---- Keyword-based sentiment heuristics (lightweight) ----
    positive_words = {"good", "great", "happy", "well", "fine", "awesome", "ok", "okay", "better", "relieved", "productive"}
    negative_words = {"sad", "depressed", "anxious", "anxiety", "stressed", "stress", "tired", "terrible", "bad", "upset", "angry", "alone", "lonely", "overwhelmed", "hopeless"}
    worried_words = {"worried", "concerned", "scared", "afraid", "nervous", "panic", "panicking"}

    # tokenization 
    words = set([w.strip(".,!?;:()\"'").lower() for w in lower.split() if w.strip()])

    pos_hits = len(words & positive_words)
    neg_hits = len(words & negative_words)
    worry_hits = len(words & worried_words)

    # Heuristic score: positive increases, negative/worry decreases
    score = (pos_hits * 1.0) - (neg_hits * 1.2) - (worry_hits * 1.1)

    if score >= 1:
        mood = "positive"
        confidence = min(0.9, 0.4 + 0.2 * pos_hits)
    elif score <= -1:
        mood = "negative"
        confidence = min(0.95, 0.4 + 0.2 * max(neg_hits, worry_hits))
    else:
        mood = "neutral"
        confidence = 0.55

    # ---- Compose supportive message ----
    if mood == "positive":
        message = (
            "I'm glad to hear that. It's great that things feel better right now — small routines and self-care really help keep that going. "
            "If you'd like, I can suggest a few short activities to keep your momentum (simple habits, micro-goals, or quick relaxation exercises)."
        )
    elif mood == "neutral":
        message = (
            "Thanks for sharing — it sounds like things might be okay but a bit unclear or mixed. "
            "If you'd like something to try right now, I can suggest a short breathing exercise, a tiny task you can complete, or a grounding technique. "
            "Which would you prefer?"
        )
    else:  # negative
        suggestions: List[str] = []
        if worry_hits:
            suggestions.append("Try a 4-4-4 breathing exercise for one minute (inhale 4s — hold 4s — exhale 4s).")
        if "tired" in words or "sleep" in lower:
            suggestions.append("If you're tired, a short restful break or a 20-minute nap might help.")
        if "alone" in words or "lonely" in words:
            suggestions.append("Consider sending a quick message to a friend or family member, even just 'not feeling great today' — connecting helps.")
        if not suggestions:
            suggestions = [
                "Take a short walk or step outside for a few minutes",
                "Write one small thing you can do right now, and do it (small wins help)",
                "Try grounding: name 5 things you see, 4 you can touch, 3 you hear."
            ]

        message = (
            "I'm really sorry you're going through a tough time — that sounds hard. Here are a few small things that might help right now:\n\n"
            + "\n".join(f"- {s}" for s in suggestions)
            + "\n\nIf you'd like, I can guide you through one of these step-by-step, or help find other support."
        )

    metadata = {"detected_mood": mood, "confidence": round(float(confidence), 2), "severity": "low"}

    return (message, metadata)