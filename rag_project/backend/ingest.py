import os
import sys
from pathlib import Path

try:
    from backend.config import PDF_DIR, VECTOR_DIR, CHUNK_SIZE, CHUNK_OVERLAP, PROVIDER, OLLAMA_EMBED_MODEL
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from backend.config import PDF_DIR, VECTOR_DIR, CHUNK_SIZE, CHUNK_OVERLAP, PROVIDER, OLLAMA_EMBED_MODEL
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

def get_embeddings():
    """Returns embeddings model based on provider"""
    if PROVIDER == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
    
    raise ValueError(f"Unknown provider: {PROVIDER}")

def ingest_all_pdfs():
    """Load PDFs, chunk them, create embeddings, save to FAISS"""
    
    # Find all PDFs
    pdf_paths = [
        os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR)
        if f.lower().endswith(".pdf")
    ]
    
    if not pdf_paths:
        raise RuntimeError(f"No PDFs found in {PDF_DIR}")
    
    print(f"📄 Found {len(pdf_paths)} PDF(s)")
    
    # Load all PDFs
    docs = []
    for pdf_path in pdf_paths:
        print(f"Loading: {os.path.basename(pdf_path)}")
        docs.extend(PyPDFLoader(pdf_path).load())
    
    print(f"📖 Loaded {len(docs)} page(s)")
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"✂️  Created {len(chunks)} chunk(s)")
    
    # Create embeddings and vector store
    print("🧠 Creating embeddings (this may take a minute)...")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save to disk
    os.makedirs(VECTOR_DIR, exist_ok=True)
    vectorstore.save_local(VECTOR_DIR)
    print(f"✅ Vector store saved to: {VECTOR_DIR}")
    
    return {
        "pdfs": len(pdf_paths),
        "pages": len(docs),
        "chunks": len(chunks),
        "vector_dir": VECTOR_DIR
    }

if __name__ == "__main__":
    result = ingest_all_pdfs()
    print("\n🎉 Ingestion complete!")
    print(result)
