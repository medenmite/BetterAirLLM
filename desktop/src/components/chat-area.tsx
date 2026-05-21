"use client";

import React, { useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";
import { ChevronDown } from "lucide-react";
import { PromptBox } from "@/components/ui/prompt-box";

// ─────────────────────────── Types ───────────────────────────
export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
}

// ─────────────────────────── Message Bubble ──────────────────
function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      className={`flex gap-4 ${isUser ? "justify-end" : ""}`}
    >
      {!isUser && (
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-[var(--color-gray-100)] dark:border-none overflow-hidden">
          <img
            src="/favicon.ico"
            className="h-9 w-9 rounded-full"
            alt=""
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
        </div>
      )}

      <div className={`${isUser ? "max-w-[70%]" : "flex-1 min-w-0"}`}>
        {/* Model name label for assistant */}
        {!isUser && (
          <div className="text-sm font-medium text-[var(--color-gray-200)] mb-1 font-primary">
            BetterAirLLM
          </div>
        )}

        {isUser ? (
          <div className="rounded-3xl bg-[var(--color-gray-850)] px-5 py-2.5 text-white">
            <p className="text-[15px] leading-relaxed whitespace-pre-wrap">{message.content}</p>
          </div>
        ) : (
          <div className="markdown-prose whitespace-pre-line">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}
      </div>
    </motion.div>
  );
}

// ─────────────────────────── Typing Indicator ────────────────
function TypingIndicator() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      className="flex gap-4"
    >
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full overflow-hidden">
        <img src="/favicon.ico" className="h-9 w-9 rounded-full" alt="" />
      </div>
      <div>
        <div className="text-sm font-medium text-[var(--color-gray-200)] mb-1 font-primary">
          BetterAirLLM
        </div>
        <div className="flex items-center gap-1.5 pt-1">
          <span className="typing-dot" />
          <span className="typing-dot" />
          <span className="typing-dot" />
        </div>
      </div>
    </motion.div>
  );
}

// ─────────────────────────── Suggestions ─────────────────────
function Suggestions({ onSelect }: { onSelect: (text: string) => void }) {
  const suggestions = [
    { title: "Explain transformer attention", subtitle: "Architecture fundamentals" },
    { title: "Optimize inference speed", subtitle: "Performance tuning" },
    { title: "Compare quantization methods", subtitle: "GPTQ vs AWQ vs GGUF" },
    { title: "Debug CUDA out of memory", subtitle: "Troubleshooting" },
  ];

  return (
    <div className="mt-2">
      <div className="mb-1 flex gap-1 text-xs font-medium items-center text-[var(--color-gray-600)]">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-3.5 w-3.5">
          <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
        </svg>
        Suggested
      </div>
      <div className="max-h-40 overflow-auto">
        {suggestions.map((s, idx) => (
          <button
            key={idx}
            className="waterfall flex flex-col flex-1 shrink-0 w-full justify-between px-3 py-2 rounded-xl bg-transparent hover:bg-white/5 transition group"
            style={{ animationDelay: `${idx * 60}ms` }}
            onClick={() => onSelect(s.title)}
          >
            <div className="flex flex-col text-left">
              <div className="font-medium text-[var(--color-gray-300)] group-hover:text-[var(--color-gray-200)] transition line-clamp-1 text-sm">
                {s.title}
              </div>
              <div className="text-xs text-[var(--color-gray-600)] font-normal line-clamp-1">
                {s.subtitle}
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────── Home State ──────────────────────
function HomeState({ onSendMessage }: { onSendMessage: (msg: string) => void }) {
  return (
    <div className="m-auto w-full max-w-6xl px-2 md:px-20 translate-y-6 py-24 text-center">
      {/* Model Icon + Name */}
      <div className="w-full text-3xl text-[var(--color-gray-100)] text-center flex items-center gap-4 font-primary">
        <div className="w-full flex flex-col justify-center items-center">
          <div className="flex flex-row justify-center gap-2.5 md:gap-3 w-fit px-5 max-w-xl">
            <div className="flex shrink-0 justify-center">
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.3 }}
              >
                <img
                  src="/favicon.ico"
                  className="h-10 w-10 rounded-full"
                  alt=""
                />
              </motion.div>
            </div>
            <motion.div
              className="text-3xl line-clamp-1 flex items-center"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3, delay: 0.05 }}
            >
              BetterAirLLM
            </motion.div>
          </div>

          <motion.div
            className="mt-1 mb-2"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          >
            <div className="mt-0.5 px-2 text-sm font-normal text-[var(--color-gray-500)] line-clamp-2 max-w-xl">
              High-performance local AI inference. Run any model on your hardware.
            </div>
          </motion.div>

          {/* MessageInput */}
          <motion.div
            className="text-base font-normal md:max-w-3xl w-full py-3"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.2 }}
          >
            <PromptBox
              placeholder="How can I help you today?"
              onSendMessage={onSendMessage}
            />
          </motion.div>
        </div>
      </div>

      {/* Suggestions */}
      <motion.div
        className="mx-auto max-w-2xl font-primary mt-2"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3, delay: 0.4 }}
      >
        <div className="mx-5">
          <Suggestions onSelect={(text) => onSendMessage(text)} />
        </div>
      </motion.div>
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════
// ████ Main Chat Area Export ████████████████████████████████████
// ═════════════════════════════════════════════════════════════════
export default function ChatArea({
  messages,
  isTyping,
  onSendMessage,
}: {
  messages: Message[];
  isTyping: boolean;
  onSendMessage: (msg: string) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  const hasMessages = messages.length > 0;

  return (
    <main className="flex flex-col h-full bg-[var(--color-gray-900)] relative">
      {!hasMessages ? (
        <div className="flex-1 overflow-y-auto scrollbar-hidden">
          <HomeState onSendMessage={onSendMessage} />
        </div>
      ) : (
        <>
          {/* Scrollable messages */}
          <div
            ref={scrollRef}
            className="flex-1 overflow-y-auto scrollbar-hidden"
          >
            <div className="max-w-3xl mx-auto px-4 md:px-6 py-6 space-y-6">
              {messages.map((msg) => (
                <MessageBubble key={msg.id} message={msg} />
              ))}

              <AnimatePresence>
                {isTyping && <TypingIndicator />}
              </AnimatePresence>
            </div>
          </div>

          {/* Bottom-anchored PromptBox */}
          <div className="bg-[var(--color-gray-900)]">
            <div className="max-w-3xl mx-auto px-4 md:px-6 pb-4 pt-2">
              <PromptBox onSendMessage={onSendMessage} />
              <div className="text-center mt-2">
                <span className="text-[var(--color-gray-600)] text-xs font-primary">
                  BetterAirLLM Desktop · Running locally
                </span>
              </div>
            </div>
          </div>
        </>
      )}
    </main>
  );
}
