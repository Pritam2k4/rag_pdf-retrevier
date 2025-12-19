import { cn } from "@/lib/utils";
import { Bot, User } from "lucide-react";

interface ChatMessageProps {
  content: string;
  isUser: boolean;
  isTyping?: boolean;
}

/**
 * ChatMessage Component
 * ChatGPT-style message layout without bubbles
 * Messages span full width with icon avatars
 */
const ChatMessage = ({ content, isUser, isTyping = false }: ChatMessageProps) => {
  return (
    <div
      className={cn(
        "animate-fade-in py-6 px-4",
        !isUser && "bg-secondary/50"
      )}
    >
      <div className="max-w-3xl mx-auto flex gap-4">
        {/* Avatar */}
        <div
          className={cn(
            "h-8 w-8 rounded-sm flex items-center justify-center shrink-0",
            isUser ? "bg-primary" : "bg-muted"
          )}
        >
          {isUser ? (
            <User className="h-5 w-5 text-primary-foreground" />
          ) : (
            <Bot className="h-5 w-5 text-foreground" />
          )}
        </div>

        {/* Message Content */}
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-muted-foreground mb-1">
            {isUser ? "You" : "Assistant"}
          </p>
          <div
            className={cn(
              "text-foreground leading-relaxed",
              isTyping && "text-muted-foreground"
            )}
          >
            {isTyping ? (
              <span className="inline-flex items-center gap-1">
                <span className="typing-dots">Thinking</span>
              </span>
            ) : (
              <p className="whitespace-pre-wrap">{content}</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;
