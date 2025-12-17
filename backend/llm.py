from backend.config import PROVIDER, OLLAMA_CHAT_MODEL

def get_llm():
    """Returns the appropriate LLM based on PROVIDER setting"""
    
    if PROVIDER == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model=OLLAMA_CHAT_MODEL,
            temperature=0.2
        )
    
    # Future: Add OpenAI support here if needed
    raise ValueError(f"Unknown provider: {PROVIDER}")
