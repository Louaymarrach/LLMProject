"""
main_agent.py - LLM Agent Interface using Gemini with RAG and Tools

Replace your existing main_agent.py with this file. It expects a local tools.py
module that exposes a Toolset class (registered tools).
"""

import os
from typing import Any, Dict, List, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "./vectorstore")

# Local Toolset (expects tools.py in same folder)
from tools import Toolset

# LLM / embeddings imports (keep consistent with your environment)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ConversationBufferMemory fallback shim (keeps compatibility)
try:
    from langchain_core.memory import ConversationBufferMemory
    # print(" Imported ConversationBufferMemory from langchain_core")
except Exception:
    try:
        from langchain.memory import ConversationBufferMemory
        # print(" Imported ConversationBufferMemory from langchain")
    except Exception:
        # print(" Using local shim for ConversationBufferMemory")
        class ConversationBufferMemory:
            def __init__(self, memory_key="chat_history", return_messages=True):
                self.memory_key = memory_key
                self.return_messages = return_messages
                self._buffer = []

            def save_context(self, inputs, outputs):
                human_text = inputs.get("input", "")
                ai_text = outputs.get("output", "")
                if human_text:
                    self._buffer.append({"role": "human", "content": human_text})
                if ai_text:
                    self._buffer.append({"role": "ai", "content": ai_text})

            def load_memory_variables(self, inputs=None):
                if self.return_messages:
                    return {self.memory_key: list(self._buffer)}
                return {self.memory_key: "\n".join(
                    f"{m['role'].capitalize()}: {m['content']}" for m in self._buffer
                )}

            def clear(self):
                self._buffer = []


# LangChain retrieval/QA pieces (optional; used if available)
try:
    from langchain_community.vectorstores import FAISS
except Exception:
    try:
        from langchain.vectorstores import FAISS
    except Exception:
        FAISS = None

try:
    from langchain_classic.chains import RetrievalQA
    from langchain_core.prompts import PromptTemplate
except Exception:
    RetrievalQA = None
    PromptTemplate = None


class AgentInterface:
    """
    LLM Agent using Gemini (ChatGoogleGenerativeAI) with optional RAG (FAISS) and custom tools.

    Provides:
     - run_query(query): main entrypoint (tries RAG -> duck -> direct LLM)
     - ingest_tool(file_paths): wrapper to tools.ingest (and reloads retriever)
     - summarize_tool(file_path): wrapper to tools.summarize
     - pdf_qa_tool(question): wrapper to tools.pdf_qa (works directly with vectorstore)
     - other small wrappers for existing tools (duck, keywords, calculator, mood_support)
    """

    def __init__(self, model_name: Optional[str] = None):
        # LLM init (Gemini variant used in your project)
        self.llm = ChatGoogleGenerativeAI(model=(model_name or "gemini-2.5-flash"), temperature=0.1)

        # Memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Tools registry
        self.tools = Toolset()

        # Retriver / vectorstore placeholders
        self.vectorstore = None
        self.retriever = None

        # Try to load an existing FAISS vectorstore so RAG can be used immediately
        if FAISS is not None and os.path.exists(VECTORSTORE_DIR):
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
                self.vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
                # many FAISS wrappers provide as_retriever
                try:
                    self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
                except Exception:
                    self.retriever = None
                print("Vectorstore loaded from", VECTORSTORE_DIR)
            except Exception as e:
                print("Could not load vectorstore:", e)

    # ---------------- Core query flow ----------------
    def run_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        metadata: Dict[str, Any] = {}
        query = (query or "").strip()
        if not query:
            return "Please enter a non-empty question.", {"error": "empty_query"}

        # ----------------------- 1) ATTEMPT RAG -----------------------
        if self.retriever and RetrievalQA is not None and PromptTemplate is not None:
            try:
                # Check if RAG has any relevant documents BEFORE running QA
                try:
                    docs = self.retriever.get_relevant_documents(query)
                except:
                    docs = []

                if not docs:
                    metadata["rag_info"] = "no_docs_retrieved"
                    raise Exception("Skipping RAG because no relevant documents were found")

                # Build prompt template
                rag_prompt = PromptTemplate(
                    input_variables=["context", "question"],
                    template=(
                        "You are an assistant that answers ONLY using the context.\n"
                        "If answer is not found in context, reply with NO_CONTEXT.\n\n"
                        "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
                    ),
                )

                qa = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.retriever,
                    chain_type_kwargs={"prompt": rag_prompt},
                )

                rag_answer = qa.run(query).strip()

                # If RAG produced a valid contextual answer → return
                if rag_answer and rag_answer != "NO_CONTEXT":
                    metadata["method"] = "RAG"
                    metadata["num_docs"] = len(docs)
                    self.memory.save_context({"input": query}, {"output": rag_answer})
                    return rag_answer, metadata

                # Otherwise allow fall-through to Duck tool
                metadata["rag_info"] = "no_answer_in_docs"

            except Exception as e:
                metadata["rag_error"] = str(e)

        # ------------------------ 2) DUCK SEARCH ------------------------
        duck_tool = self._get_tool_flexible("duck", "duck_search")
        if duck_tool:
            try:
                duck_out = duck_tool(query)

                # Convert duck output to readable text
                duck_snippets = []
                if isinstance(duck_out, dict):
                    abstract = duck_out.get("abstract", "") or duck_out.get("AbstractText", "")
                    if abstract:
                        duck_snippets.append(abstract)
                    for rt in duck_out.get("related_topics", [])[:3]:
                        t = rt.get("text", "") if isinstance(rt, dict) else str(rt)
                        if t:
                            duck_snippets.append(t)

                elif isinstance(duck_out, list):
                    duck_snippets = duck_out[:3]

                elif isinstance(duck_out, str):
                    duck_snippets = [duck_out]

                duck_snippets = [s for s in duck_snippets if s.strip()]
                if duck_snippets:
                    metadata["method"] = "duck_search"
                    metadata["web_results"] = duck_snippets
                    answer = (
                        "No answer found in course materials, but here is what I found on the web:\n\n"
                        + "\n\n".join(f"- {s}" for s in duck_snippets)
                    )
                    self.memory.save_context({"input": query}, {"output": answer})
                    return answer, metadata

            except Exception as e:
                metadata["duck_error"] = str(e)

        # ------------------------ 3) DIRECT LLM ------------------------
        try:
            prompt = f"You are a helpful assistant.\nUser: {query}"
            resp = self.llm.invoke(prompt)
            text = getattr(resp, "content", str(resp))
        except Exception as e:
            text = f"LLM error: {e}"
            metadata["llm_error"] = str(e)

        metadata["method"] = "direct_llm"
        self.memory.save_context({"input": query}, {"output": text})
        return text, metadata



    # ---------------- Helper / tool wrappers ----------------
    def _get_tool_flexible(self, *names) -> Optional[Any]:
        """
        Try to fetch a tool by any of the provided names (useful when tools.py might
        register different names across versions, e.g. 'duck' vs 'duck_search')
        """
        for n in names:
            t = self.tools.get_tool(n)
            if t:
                return t
        return None

    def keyword_extractor_tool(self, text: str) -> List[str]:
        fn = self._get_tool_flexible("keywords", "keyword_extractor", "keyword_tool")
        if not fn:
            return []
        try:
            return fn(text)
        except Exception:
            return []

    def calculator_tool(self, expr: str) -> str:
        fn = self._get_tool_flexible("calculator")
        if not fn:
            return "Calculator tool not available."
        try:
            return fn(expr)
        except Exception as e:
            return f"Calculator error: {e}"

    def duck_search_tool(self, query: str) -> Any:
        fn = self._get_tool_flexible("duck_search", "duck")
        if not fn:
            return {"error": "Duck tool not available."}
        return fn(query)

    def mood_support_tool(self, text: str) -> Tuple[str, Dict[str, Any]]:
        fn = self._get_tool_flexible("mood_support")
        metadata: Dict[str, Any] = {"method": "mood_support"}
        if not fn:
            return "Mood support tool not available.", {"error": "tool_not_available"}
        try:
            res = fn(text)
            # If tool returns tuple (text, meta), handle it
            if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
                return res[0], {**metadata, **res[1]}
            return res, metadata
        except Exception as e:
            return "Mood support tool failed.", {"error": str(e)}

    # ---------------- RAG ingestion + summarization wrappers ----------------
    def ingest_tool(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Call the registered 'ingest' or 'ingest_documents' tool. After ingestion,
        attempt to reload the local FAISS vectorstore so RAG is immediately available.
        """
        ingest_fn = self._get_tool_flexible("ingest", "ingest_documents")
        if ingest_fn is None:
            return {"status": "error", "error": "ingest tool not available."}
        try:
            result = ingest_fn(file_paths)
        except Exception as e:
            return {"status": "error", "error": f"ingest call failed: {e}"}

        # If ingestion reported OK, try to reload retriever
        try:
            if isinstance(result, dict) and (result.get("status") == "ok" or result.get("report")):
                if FAISS is not None:
                    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
                    try:
                        self.vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
                        try:
                            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
                        except Exception:
                            self.retriever = None
                        result["retriever_reloaded"] = True
                    except Exception as e:
                        result["retriever_reloaded_error"] = str(e)
        except Exception:
            pass

        return result

    def summarize_tool(self, file_path: str, max_chars: int = 1200) -> Dict[str, Any]:
        summarize_fn = self._get_tool_flexible("summarize", "summarize_document")
        if not summarize_fn:
            return {"status": "error", "error": "summarize tool not available."}
        try:
            return summarize_fn(file_path, max_chars=max_chars)
        except TypeError:
            # maybe function only accepts (file_path)
            try:
                return summarize_fn(file_path)
            except Exception as e:
                return {"status": "error", "error": f"summarize failed: {e}"}
        except Exception as e:
            return {"status": "error", "error": f"summarize failed: {e}"}

    def pdf_qa_tool(self, question: str, k: int = 3) -> Dict[str, Any]:
        pdfqa_fn = self._get_tool_flexible("pdf_qa", "doc_qa", "rag_qa")
        if not pdfqa_fn:
            return {"status": "error", "error": "pdf_qa tool not available."}
        try:
            return pdfqa_fn(question, k=k)
        except Exception as e:
            return {"status": "error", "error": f"pdf_qa failure: {e}"}


# CLI Debugging
if __name__ == "__main__":
    ai = AgentInterface()
    print("Agent ready. Type questions (exit/q to quit).")
    while True:
        q = input("Ask: ")
        if q.strip().lower() in {"exit", "quit", "q"}:
            break
        r, meta = ai.run_query(q)
        print("\n--- Response ---")
        print(r)
        if meta:
            print("\n--- Metadata ---")
            print(meta)
