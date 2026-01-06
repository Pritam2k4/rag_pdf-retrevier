"""
Sunjos RAG Engine - Core retrieval and generation pipeline
Handles document processing, embedding, and Q&A with source citations

Uses Groq (FREE) for LLM and simple TF-IDF for embeddings (no API cost)
"""

import os
import json
import hashlib
import pickle
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


class SimpleVectorStore:
    """
    Simple vector store using TF-IDF embeddings (FREE - no API needed)
    Works offline and has no rate limits!
    """
    
    def __init__(self, persist_path: Optional[str] = None):
        self.documents: List[Document] = []
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.persist_path = persist_path
        
        if persist_path and os.path.exists(persist_path):
            self._load()
    
    def add_documents(self, documents: List[Document]):
        """Add documents and rebuild TF-IDF index"""
        self.documents.extend(documents)
        
        # Rebuild TF-IDF matrix with all documents
        if self.documents:
            texts = [doc.page_content for doc in self.documents]
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        if self.persist_path:
            self._save()
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Find most similar documents to query using TF-IDF"""
        if not self.documents or self.tfidf_matrix is None:
            return []
        
        # Transform query using the same vectorizer
        query_vec = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        return [self.documents[i] for i in top_k_indices if similarities[i] > 0.01]
    
    def get(self, where: Dict = None) -> Dict:
        """Get documents matching filter"""
        if where is None:
            return {"ids": list(range(len(self.documents)))}
        
        matching_ids = []
        for i, doc in enumerate(self.documents):
            match = True
            for key, value in where.items():
                if doc.metadata.get(key) != value:
                    match = False
                    break
            if match:
                matching_ids.append(i)
        
        return {"ids": matching_ids}
    
    def delete(self, ids: List[int]):
        """Delete documents by indices and rebuild index"""
        for i in sorted(ids, reverse=True):
            if 0 <= i < len(self.documents):
                del self.documents[i]
        
        # Rebuild TF-IDF matrix
        if self.documents:
            texts = [doc.page_content for doc in self.documents]
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        else:
            self.tfidf_matrix = None
        
        if self.persist_path:
            self._save()
    
    def _save(self):
        """Persist to disk"""
        Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "documents": [(doc.page_content, doc.metadata) for doc in self.documents],
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix
        }
        with open(self.persist_path, 'wb') as f:
            pickle.dump(data, f)
    
    def _load(self):
        """Load from disk"""
        try:
            with open(self.persist_path, 'rb') as f:
                data = pickle.load(f)
            self.documents = [Document(page_content=d[0], metadata=d[1]) for d in data["documents"]]
            self.vectorizer = data.get("vectorizer", TfidfVectorizer(max_features=5000, stop_words='english'))
            self.tfidf_matrix = data.get("tfidf_matrix")
        except Exception as e:
            print(f"Warning: Could not load vector store: {e}")
            self.documents = []
            self.tfidf_matrix = None


class RAGEngine:
    """
    Sunjos RAG (Retrieval Augmented Generation) Engine
    Processes documents and answers questions with source citations
    
    Uses:
    - Groq (FREE) for fast LLM inference
    - TF-IDF (FREE, offline) for document retrieval
    """

    def __init__(self, persist_directory: str = "./data"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Use Groq (FREE and super fast!)
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",  # Best free model
            temperature=0.1
        )
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize vector store (TF-IDF based - FREE!)
        self.vector_store = SimpleVectorStore(
            persist_path=os.path.join(persist_directory, "vectors.pkl")
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
        """Generate unique document ID based on filename and content hash"""
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
        """
        Add a document to the knowledge base
        Returns document metadata
        """
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
        
        # Add to vector store (TF-IDF - no API call needed!)
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
        
        # Get all chunk indices for this document
        results = self.vector_store.get(where={"doc_id": doc_id})
        if results and results.get("ids"):
            self.vector_store.delete(ids=results["ids"])
        
        # Remove from metadata
        del self.documents_meta[doc_id]
        self._save_meta()
        return True

    def get_documents(self) -> List[Dict]:
        """Get list of all documents in the knowledge base"""
        return list(self.documents_meta.values())

    def query(self, question: str, k: int = 4) -> Dict:
        """
        Query the knowledge base and get an answer with sources
        
        Args:
            question: The user's question
            k: Number of relevant chunks to retrieve
            
        Returns:
            Dict with answer, sources, and confidence
        """
        # Retrieve relevant documents using TF-IDF (FREE!)
        relevant_docs = self.vector_store.similarity_search(question, k=k)
        
        # If no documents or no relevant content, respond as friendly assistant
        if not relevant_docs:
            # Use LLM to respond naturally and conversationally
            chat_prompt = ChatPromptTemplate.from_template("""You are Sunjos, a warm and friendly AI companion with a sweet personality.

IMPORTANT GUIDELINES:
- Be natural and human-like, NOT robotic or formulaic
- Vary your responses - never use the same phrases repeatedly
- DON'T mention "documents" or "upload" unless the user specifically asks about your capabilities
- Just chat naturally like a caring friend would
- Use 1-2 emojis max, not in every sentence
- Keep responses short (1-3 sentences usually)
- Show genuine interest in the person
- Be playful and warm, but not overly enthusiastic

Examples of GOOD responses:
- "Hey there! 💕 How's your day going?"
- "Haha, nice to hear from you! What's on your mind?"
- "Oh that's interesting! Tell me more~"

Examples of BAD responses (too robotic):
- "Hello! ✿ I'm Sunjos, your friendly AI assistant! ♡ I'm here to help you with documents!"
- "I'd be happy to assist you! Please upload your documents!"

User says: {question}

Your natural response:""")
            
            chain = chat_prompt | self.llm | StrOutputParser()
            answer = chain.invoke({"question": question})
            
            # Add to conversation history
            self.conversation_history.append({
                "question": question,
                "answer": answer,
                "sources": [],
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "answer": answer,
                "sources": [],
                "has_context": False
            }
        
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
        
        # Create prompt with context
        prompt = ChatPromptTemplate.from_template("""You are Sunjos, a helpful and friendly assistant that answers questions based on the provided context. 
Your answers should be:
- Accurate and based only on the given context
- Clear and well-structured
- Friendly and helpful in tone
- Include specific references to the source material when appropriate

If the context doesn't contain enough information to answer the question fully, acknowledge this and provide what information is available.

Context:
{context}

Question: {question}

Answer:""")
        
        # Generate answer using Groq (FREE and fast!)
        chain = prompt | self.llm | StrOutputParser()
        
        answer = chain.invoke({"context": context, "question": question})
        
        # Add to conversation history
        self.conversation_history.append({
            "question": question,
            "answer": answer,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "answer": answer,
            "sources": sources,
            "has_context": True
        }

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
