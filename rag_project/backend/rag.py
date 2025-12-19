from backend.config import VECTOR_DIR, TOP_K, PROVIDER, OLLAMA_EMBED_MODEL
from backend.llm import get_llm
from langchain_community.vectorstores import FAISS

def get_embeddings():
    """Returns embeddings model based on provider"""
    if PROVIDER == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
    
    raise ValueError(f"Unknown provider: {PROVIDER}")

def get_rag_answer(query: str) -> str:
    """
    Retrieves relevant chunks from vector DB and generates answer using LLM
    """
    # Load vector store
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        VECTOR_DIR, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    
    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(query)
    
    # Combine retrieved chunks into context
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Build prompt with context
    prompt = (
    "You are an intelligent assistant answering questions using the provided document context.\n\n"

    "Guidelines:\n"
    "- Use the context as the primary source of information\n"
    "- You may paraphrase and rephrase to sound natural\n"
    "- If the answer is partially present, infer carefully without adding new facts\n"
    "- If the user asks for a list, steps, or explanation, format the answer accordingly\n"
    "- Do NOT copy text verbatim unless necessary\n"
    "- If the information is truly missing, clearly say so\n\n"

    f"Context:\n{context}\n\n"
    f"Question:\n{query}\n\n"
    "Answer:"
)

    
    # Generate answer using LLM
    llm = get_llm()
    response = llm.invoke(prompt)
    
    return response.content
