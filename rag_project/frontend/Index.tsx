import { useState, useRef, useEffect } from "react";
import ChatMessage from "./src/components/ChatMessage";
import ChatInput from "./src/components/ChatInput";



import { Sparkles } from "lucide-react";

/**
 * Message interface with optional sources
 */
interface Message {
  id: string;
  content: string;
  isUser: boolean;
  sources?: string[];
}

/**
 * Index Page - Enhanced ChatGPT-inspired Chat Interface
 * With streaming, markdown support, and session persistence
 */
const Index = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [isBotTyping, setIsBotTyping] = useState(false);

  const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isBotTyping]);

  // Load chat history from localStorage on mount
  useEffect(() => {
    const savedMessages = localStorage.getItem('chat_history');
    if (savedMessages) {
      try {
        setMessages(JSON.parse(savedMessages));
      } catch (error) {
        console.error('Failed to load chat history:', error);
      }
    }
  }, []);

  // Save chat history to localStorage whenever messages change
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem('chat_history', JSON.stringify(messages));
    }
  }, [messages]);

  const generateId = () => {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
  };

  const handleSendMessage = async (content: string) => {
    // Add user message
    const userMessage: Message = {
      id: generateId(),
      content,
      isUser: true,
    };
    
    setMessages((prev) => [...prev, userMessage]);
    setIsBotTyping(true);

    // Create placeholder bot message for streaming
    const botMessageId = generateId();
    const botMessage: Message = {
      id: botMessageId,
      content: "",
      isUser: false,
    };
    setMessages((prev) => [...prev, botMessage]);

    try {
      // Call backend API with streaming support
      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: content }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      // Check if response supports streaming
      const contentType = response.headers.get("content-type");
      
      if (contentType?.includes("text/event-stream")) {
        // Handle Server-Sent Events streaming
        await handleSSEStream(response, botMessageId);
      } else if (response.body) {
        // Handle ReadableStream streaming
        await handleReadableStream(response, botMessageId);
      } else {
        // Fallback to regular JSON response
        const data = await response.json();
        updateBotMessage(botMessageId, data.answer, data.sources);
      }
    } catch (error: any) {
      // Update bot message with error
      updateBotMessage(
        botMessageId,
        `❌ Error: ${error.message}. Make sure the backend is running on port 8000.`,
        []
      );
    } finally {
      setIsBotTyping(false);
    }
  };

  /**
   * Handle streaming via ReadableStream (Fetch API)
   */
  const handleReadableStream = async (response: Response, messageId: string) => {
    const reader = response.body?.getReader();
    const decoder = new TextDecoder();
    let accumulatedText = "";

    if (!reader) return;

    try {
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        accumulatedText += chunk;

        // Update message incrementally
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === messageId
              ? { ...msg, content: accumulatedText }
              : msg
          )
        );
      }
    } catch (error) {
      console.error('Stream reading error:', error);
    } finally {
      reader.releaseLock();
    }
  };

  /**
   * Handle Server-Sent Events streaming
   */
  const handleSSEStream = async (response: Response, messageId: string) => {
    const reader = response.body?.getReader();
    const decoder = new TextDecoder();
    let accumulatedText = "";

    if (!reader) return;

    try {
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            
            if (data === '[DONE]') {
              return;
            }
            
            accumulatedText += data;
            
            // Update message incrementally with typing effect
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === messageId
                  ? { ...msg, content: accumulatedText }
                  : msg
              )
            );
          }
        }
      }
    } catch (error) {
      console.error('SSE stream error:', error);
    } finally {
      reader.releaseLock();
    }
  };

  /**
   * Update bot message with final content and sources
   */
  const updateBotMessage = (messageId: string, content: string, sources?: string[]) => {
    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === messageId
          ? { ...msg, content, sources }
          : msg
      )
    );
  };

  /**
   * Clear chat history
   */
  const handleClearChat = () => {
    setMessages([]);
    localStorage.removeItem('chat_history');
  };

  /**
   * Regenerate last bot response
   */
  const handleRegenerate = async () => {
    if (messages.length < 2) return;

    // Find the last user message
    const lastUserMessage = [...messages]
      .reverse()
      .find((msg) => msg.isUser);

    if (lastUserMessage) {
      // Remove last bot response
      setMessages((prev) => prev.slice(0, -1));
      // Resend last user message
      await handleSendMessage(lastUserMessage.content);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-background">
      {/* Header with controls */}
      {messages.length > 0 && (
        <header className="border-b border-border px-4 py-3 flex justify-between items-center">
          <h2 className="text-sm font-medium text-muted-foreground">
            Chat Session
          </h2>
          <button
            onClick={handleClearChat}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            Clear Chat
          </button>
        </header>
      )}

      {/* Messages Area */}
      <main className="flex-1 overflow-y-auto">
        {/* Empty state */}
        {messages.length === 0 && !isBotTyping && (
          <div className="flex flex-col items-center justify-center h-full px-4">
            <div className="h-16 w-16 rounded-full bg-card border border-border flex items-center justify-center mb-6">
              <Sparkles className="h-8 w-8 text-primary" />
            </div>
            <h1 className="text-2xl font-semibold text-foreground mb-2">
              How can I help you today?
            </h1>
            <p className="text-muted-foreground text-center max-w-md">
              Upload a PDF and ask questions about its content. I'll use RAG to provide accurate answers.
            </p>
          </div>
        )}

        {/* Messages list */}
        <div className="pb-4">
          {messages.map((message) => (
            <ChatMessage
              key={message.id}
              content={message.content}
              isUser={message.isUser}
              sources={message.sources}
              onRegenerate={
                !message.isUser && 
                message.id === messages[messages.length - 1]?.id
                  ? handleRegenerate
                  : undefined
              }
            />
          ))}
          
          {isBotTyping && (
            <ChatMessage
              content=""
              isUser={false}
              isTyping={true}
            />
          )}
        </div>
        
        <div ref={messagesEndRef} />
      </main>

      {/* Input */}
      <ChatInput 
        onSendMessage={handleSendMessage} 
        disabled={isBotTyping}
      />
    </div>
  );
};

export default Index;


