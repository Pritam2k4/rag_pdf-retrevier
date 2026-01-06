"""
Sunjos RAG Engine - Core retrieval and generation pipeline
Handles document processing, embedding, and Q&A with source citations

Uses:
- Hugging Face Sentence Transformers (all-MiniLM-L6-v2) for embeddings - FREE, offline
- FAISS for vector store - FREE, local
- Groq (Llama 3.3) for LLM - FREE tier
"""

import os
import json
import hashlib
import pickle
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class FAISSVectorStore:
    """
    FAISS-based vector store using Hugging Face Sentence Transformers
    - Semantic search (understands meaning, not just keywords)
    - Works completely offline
    - No API costs
    """
    
    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = persist_path
        self.documents: List[Document] = []
        
        # Use all-MiniLM-L6-v2 (same as reference repo)
        # Small, fast, and great for semantic search
        print("Loading embedding model (first time may take a moment)...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384  # Dimension of all-MiniLM-L6-v2
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product (cosine similarity)
        
        # Try to load existing data
        if persist_path and os.path.exists(persist_path):
            self._load()
    
    def add_documents(self, documents: List[Document]):
        """Add documents with their embeddings to the index"""
        if not documents:
            return
        
        # Get text content from documents
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings using Sentence Transformers
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
        
        # Add to FAISS index
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Store documents
        self.documents.extend(documents)
        
        # Persist
        if self.persist_path:
            self._save()
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Find most similar documents to query using semantic search"""
        if not self.documents or self.index.ntotal == 0:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        
        # Search FAISS index
        k = min(k, self.index.ntotal)  # Don't ask for more than we have
        scores, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        
        # Return documents with positive similarity
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and scores[0][i] > 0:
                results.append(self.documents[idx])
        
        return results
    
    def get_all_documents(self) -> List[Document]:
        """Get all documents"""
        return self.documents
    
    def delete_by_doc_id(self, doc_id: str):
        """Delete all chunks for a document ID"""
        # Find indices to keep
        indices_to_keep = []
        docs_to_keep = []
        
        for i, doc in enumerate(self.documents):
            if doc.metadata.get("doc_id") != doc_id:
                indices_to_keep.append(i)
                docs_to_keep.append(doc)
        
        if len(docs_to_keep) == len(self.documents):
            return  # Nothing to delete
        
        # Rebuild index with remaining documents
        self.documents = []
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        if docs_to_keep:
            self.add_documents(docs_to_keep)
        elif self.persist_path:
            self._save()
    
    def clear(self):
        """Clear all documents"""
        self.documents = []
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        if self.persist_path:
            self._save()
    
    def _save(self):
        """Persist to disk"""
        Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save documents
        doc_data = [(doc.page_content, doc.metadata) for doc in self.documents]
        with open(self.persist_path, 'wb') as f:
            pickle.dump(doc_data, f)
        
        # Save FAISS index
        index_path = self.persist_path.replace('.pkl', '.faiss')
        faiss.write_index(self.index, index_path)
    
    def _load(self):
        """Load from disk"""
        try:
            # Load documents
            with open(self.persist_path, 'rb') as f:
                doc_data = pickle.load(f)
            self.documents = [Document(page_content=d[0], metadata=d[1]) for d in doc_data]
            
            # Load FAISS index
            index_path = self.persist_path.replace('.pkl', '.faiss')
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
            else:
                # Rebuild index from documents
                if self.documents:
                    texts = [doc.page_content for doc in self.documents]
                    embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
                    self.index.add(np.array(embeddings).astype('float32'))
        except Exception as e:
            print(f"Warning: Could not load vector store: {e}")
            self.documents = []
            self.index = faiss.IndexFlatIP(self.embedding_dim)


class RAGEngine:
    """
    Sunjos RAG (Retrieval Augmented Generation) Engine
    
    Uses:
    - Hugging Face Sentence Transformers for semantic embeddings (FREE, offline)
    - FAISS for fast vector search (FREE, local)
    - Groq for LLM inference (FREE tier)
    """

    def __init__(self, persist_directory: str = "./data"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize FAISS vector store with HuggingFace embeddings
        self.vector_store = FAISSVectorStore(
            persist_path=os.path.join(persist_directory, "vectors.pkl")
        )
        
        # Use Groq LLM (FREE and super fast!)
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1
        )
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Document metadata store
        self.meta_path = os.path.join(persist_directory, "documents_meta.json")
        self.documents_meta: Dict[str, dict] = self._load_meta()
        
        # Conversation history
        self.conversation_history: List[Dict] = []

    def _load_meta(self) -> Dict[str, dict]:
        """Load document metadata from disk"""
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_meta(self):
        """Save document metadata to disk"""
        with open(self.meta_path, 'w') as f:
            json.dump(self.documents_meta, f, indent=2)

    def _generate_doc_id(self, filename: str, content: str) -> str:
        """Generate unique document ID"""
        hash_input = f"{filename}:{content[:1000]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def _load_document(self, file_path: str) -> List[Document]:
        """Load document based on file extension"""
        extension = os.path.splitext(file_path)[1].lower()
        
        if extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif extension == ".docx":
            loader = Docx2txtLoader(file_path)
        elif extension in [".txt", ".md"]:
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {extension}")
        
        return loader.load()

    def add_document(self, file_path: str, original_filename: str) -> Dict:
        """Add a document to the knowledge base"""
        # Load document
        documents = self._load_document(file_path)
        
        if not documents:
            raise ValueError("No content could be extracted from the document")
        
        # Generate document ID
        full_content = "\n".join([doc.page_content for doc in documents])
        doc_id = self._generate_doc_id(original_filename, full_content)
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Add metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "doc_id": doc_id,
                "filename": original_filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "timestamp": datetime.now().isoformat()
            })
        
        # Add to FAISS vector store (embeddings generated automatically)
        self.vector_store.add_documents(chunks)
        
        # Store document metadata
        doc_meta = {
            "id": doc_id,
            "filename": original_filename,
            "chunks": len(chunks),
            "characters": len(full_content),
            "added_at": datetime.now().isoformat()
        }
        self.documents_meta[doc_id] = doc_meta
        self._save_meta()
        
        return doc_meta

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the knowledge base"""
        if doc_id not in self.documents_meta:
            return False
        
        # Delete from vector store
        self.vector_store.delete_by_doc_id(doc_id)
        
        # Remove from metadata
        del self.documents_meta[doc_id]
        self._save_meta()
        return True

    def get_documents(self) -> List[Dict]:
        """Get list of all documents"""
        return list(self.documents_meta.values())

    def query(self, question: str, k: int = 4) -> Dict:
        """Query the knowledge base and get an answer with sources"""
        
        # Check if user is asking for summarization or generic document queries
        question_lower = question.lower()
        is_summarization_request = any(word in question_lower for word in [
            'summarize', 'summary', 'summarise', 'overview', 'about',
            'what is', 'tell me', 'explain', 'describe',
            'my pdf', 'my document', 'the document', 'the pdf', 'uploaded'
        ])
        
        # Get all available documents
        all_docs = self.vector_store.get_all_documents()
        
        # If user has documents and asks generic/summarization question
        if is_summarization_request and all_docs:
            # Use more chunks for summarization
            relevant_docs = all_docs[:min(8, len(all_docs))]
        else:
            # Normal semantic search
            relevant_docs = self.vector_store.similarity_search(question, k=k)
            
            # Fallback: if no matches but user has docs, use first few chunks
            if not relevant_docs and all_docs:
                relevant_docs = all_docs[:min(k, len(all_docs))]
        
        # If still no documents, respond conversationally
        if not relevant_docs:
            chat_prompt = ChatPromptTemplate.from_template("""You are Sunjos, a warm and friendly AI companion.

Be natural and human-like. Keep responses short (1-3 sentences).
DON'T mention "documents" or "upload" unless specifically asked.

User says: {question}

Your response:""")
            
            chain = chat_prompt | self.llm | StrOutputParser()
            answer = chain.invoke({"question": question})
            
            self.conversation_history.append({
                "question": question,
                "answer": answer,
                "sources": [],
                "timestamp": datetime.now().isoformat()
            })
            
            return {"answer": answer, "sources": [], "has_context": False}
        
        # Build context from retrieved documents
        context_parts = []
        sources = []
        seen_sources = set()
        
        for i, doc in enumerate(relevant_docs):
            context_parts.append(f"[{i+1}] {doc.page_content}")
            
            source_key = f"{doc.metadata.get('filename', 'Unknown')}_{doc.metadata.get('chunk_index', 0)}"
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append({
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "chunk": doc.metadata.get("chunk_index", 0) + 1,
                    "total_chunks": doc.metadata.get("total_chunks", 1),
                    "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
        
        context = "\n\n".join(context_parts)
        
        # Create RAG prompt
        prompt = ChatPromptTemplate.from_template("""You are Sunjos, a helpful assistant that answers questions based on the provided context.

Be accurate, clear, and friendly. Reference the source material when appropriate.
If the context doesn't fully answer the question, acknowledge this and provide what you can.

Context:
{context}

Question: {question}

Answer:""")
        
        # Generate answer using Groq
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        
        # Add to history
        self.conversation_history.append({
            "question": question,
            "answer": answer,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        })
        
        return {"answer": answer, "sources": sources, "has_context": True}

    def get_conversation_history(self, limit: int = 50) -> List[Dict]:
        """Get recent conversation history"""
        return self.conversation_history[-limit:]

    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []


# Singleton instance
_engine_instance: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """Get or create the RAG engine singleton"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RAGEngine()
    return _engine_instance
