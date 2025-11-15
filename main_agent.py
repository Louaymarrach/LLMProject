"""
main_agent.py - LLM Agent Interface using Gemini with RAG and Tools
"""

import os
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv
from tools import Toolset

load_dotenv()
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "./vectorstore")

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# ---------------- ConversationBufferMemory Fallback ----------------
try:
    from langchain_core.memory import ConversationBufferMemory
    print(" Imported ConversationBufferMemory from langchain_core")
except ImportError:
    try:
        from langchain.memory import ConversationBufferMemory
        print(" Imported ConversationBufferMemory from langchain")
    except ImportError:
        print(" Using local shim for ConversationBufferMemory")

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

from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

# ---------------- Agent Interface ----------------
class AgentInterface:
    """LLM Agent using Gemini with optional RAG (FAISS) and custom tools."""

    def __init__(self, model_name: str = None):


        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.retriever = None
        self.vectorstore = None
        self.tools = Toolset()

        # Load FAISS vectorstore if available
        if os.path.exists(VECTORSTORE_DIR):
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
                self.vectorstore = FAISS.load_local(
                    VECTORSTORE_DIR,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
                print("Vectorstore loaded")
            except Exception as e:
                print(f" Could not load vectorstore: {e}")

    # ---------------- Query Handler ----------------
    def run_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Run a query through RAG if available, otherwise use direct LLM."""
        metadata: Dict[str, Any] = {}

        # Use Retrieval-Augmented Generation if retriever exists
        if self.retriever:
            try:
                qa = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.retriever
                )
                answer = qa.run(query)
                metadata["method"] = "RAG"
                return answer, metadata
            except Exception as e:
                metadata["rag_error"] = str(e)

        # Fallback: direct LLM query
        system_prompt = "You are a helpful assistant specialized in the University domain."
        messages = [HumanMessage(content=f"{system_prompt}\nUser: {query}")]

        try:
            resp = self.llm(messages)
            text = getattr(resp, "content", str(resp))
        except Exception as e:
            text = f"LLM Query Failed: {e}"
            metadata["llm_error"] = str(e)

        # Save to memory
        try:
            self.memory.save_context({"input": query}, {"output": text})
        except Exception:
            pass

        return text, metadata

    # ---------------- Tool Access ----------------
    def keyword_extractor_tool(self, text: str) -> List[str]:
        return self.tools.get_tool("keywords")(text)

    def calculator_tool(self, expr: str) -> str:
        return self.tools.get_tool("calculator")(expr)

    def duck_search_tool(self, query: str) -> List[str]:
        return self.tools.get_tool("duck_search")(query)
    
    def mood_support_tool(self, text: str) -> str:
        tool = self.tools.get_tool("mood_support")
        if tool is None:
            return "The mood support tool is not available right now."
        return tool(text)

# ---------------- CLI Debug ----------------
if __name__ == "__main__":
    ai = AgentInterface()
    while True:
        q = input("Ask: ")
        if q.strip().lower() in {"exit", "quit"}:
            break
        r, meta = ai.run_query(q)
        print("\n--- Response ---")
        print(r)
        if meta:
            print("\n--- Metadata ---")
            print(meta)
