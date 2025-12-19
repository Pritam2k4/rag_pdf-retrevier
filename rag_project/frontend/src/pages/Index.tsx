import { useState, useRef, useEffect } from "react";
import ChatMessage from "@/components/ChatMessage";
import ChatInput from "@/components/ChatInput";
import { Sparkles } from "lucide-react";

/**
 * Message interface
 */
interface Message {
  id: string;
  content: string;
  isUser: boolean;
}

/**
 * Index Page - ChatGPT-inspired Chat Interface
 * Dark gray theme with clean, minimal design
 */
const Index = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [isBotTyping, setIsBotTyping] = useState(false);

  const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isBotTyping]);

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

    try {
      // Call backend API
      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: content }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      
      // Add bot response
      const botMessage: Message = {
        id: generateId(),
        content: data.answer,
        isUser: false,
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error: any) {
      // Add error message
      const errorMessage: Message = {
        id: generateId(),
        content: `Error: ${error.message}. Make sure backend is running on port 8000.`,
        isUser: false,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsBotTyping(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-background">
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
              Start a conversation by typing a message below.
            </p>
          </div>
        )}

        {/* Messages list */}
        <div>
          {messages.map((message) => (
            <ChatMessage
              key={message.id}
              content={message.content}
              isUser={message.isUser}
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
        disabled={false}
      />
    </div>
  );
};

export default Index;

