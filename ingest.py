"""
 python ingest.py --docs_folder ./data
"""
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError(" No API key found. Please set GEMINI_API_KEY in your .env file.")
os.environ["GOOGLE_API_KEY"] = api_key  

try:
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_community.vectorstores import FAISS
except Exception as e:
    raise RuntimeError(
        "Install required packages"
    ) from e



def ingest(docs_folder: str, output_dir: str = './vectorstore'):
    """Load, split, embed, and store documents for retrieval."""
    docs = []
    for root, _, files in os.walk(docs_folder):
        for fname in files:
            path = os.path.join(root, fname)
            if fname.lower().endswith('.txt'):
                loader = TextLoader(path, encoding='utf-8')
            elif fname.lower().endswith('.pdf'):
                loader = PyPDFLoader(path)
            else:
                print(f"Skipping unsupported file: {path}")
                continue
            docs.extend(loader.load())

    if not docs:
        print("No documents found to ingest.")
        return

    print(f" Loaded {len(docs)} documents. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f" Created {len(chunks)} chunks. Generating embeddings with Gemini...")

    # Use Gemini embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(output_dir, exist_ok=True)
    vectorstore.save_local(output_dir)
    print(f" Ingested {len(chunks)} chunks into {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs_folder', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./vectorstore')
    args = parser.parse_args()
    ingest(args.docs_folder, args.output_dir)
