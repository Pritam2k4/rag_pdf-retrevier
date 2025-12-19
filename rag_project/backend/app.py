from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.rag import get_rag_answer

app = FastAPI(title="RAG Chatbot API")

# Allow frontend (running on port 8080) to call this API
origins = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

@app.get("/")
def root():
    return {"message": "RAG Chatbot API is running!"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Main chat endpoint - receives question, returns RAG answer"""
    answer = get_rag_answer(request.query)
    return {"answer": answer}
