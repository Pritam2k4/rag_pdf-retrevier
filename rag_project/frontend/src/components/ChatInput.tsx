import { useState, KeyboardEvent, useRef, useEffect } from "react";
import { ArrowUp } from "lucide-react";
import { cn } from "@/lib/utils";

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
}

/**
 * ChatInput Component
 * ChatGPT-style input with rounded container and send button
 */
const ChatInput = ({ onSendMessage, disabled = false }: ChatInputProps) => {
  const [message, setMessage] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea based on content
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, [message]);

  const handleSend = () => {
    const trimmedMessage = message.trim();
    if (trimmedMessage && !disabled) {
      onSendMessage(trimmedMessage);
      setMessage("");
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="border-t border-border bg-background px-4 py-4">
      <div className="max-w-3xl mx-auto">
        <div className="relative flex items-end rounded-2xl border border-border bg-card shadow-card">
          {/* Textarea */}
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message ChatBot..."
            disabled={disabled}
            rows={1}
            className="flex-1 resize-none bg-transparent px-4 py-3.5 pr-12 text-foreground placeholder:text-muted-foreground focus:outline-none disabled:opacity-50 max-h-[200px] overflow-y-auto"
          />
          
          {/* Send button */}
          <button
            onClick={handleSend}
            disabled={!message.trim() || disabled}
            className={cn(
              "absolute right-2 bottom-2 h-8 w-8 rounded-lg flex items-center justify-center transition-colors",
              message.trim()
                ? "bg-foreground text-background hover:bg-foreground/90"
                : "bg-muted text-muted-foreground cursor-not-allowed"
            )}
          >
            <ArrowUp className="h-5 w-5" />
          </button>
        </div>
        
        <p className="text-xs text-muted-foreground text-center mt-2">
          Press Enter to send, Shift + Enter for new line
        </p>
      </div>
    </div>
  );
};

export default ChatInput;
