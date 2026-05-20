"use client";

import React, { useState, useCallback } from "react";
import Sidebar from "@/components/sidebar";
import ChatArea, { type Message } from "@/components/chat-area";

// ─────────────────────── Mock AI responses ───────────────────
const mockResponse = `Great question! Let me break this down.

## Layer-wise Inference

AirLLM processes models one transformer layer at a time:

1. **Load Layer** — Memory-map the layer's weights from disk
2. **Forward Pass** — Run the input through the layer on GPU
3. **Store Activations** — Keep the output in a buffer
4. **Unload Layer** — Release GPU memory
5. **Repeat** — Move to the next layer

\`\`\`python
from airllm import AirLLMEngine

engine = AirLLMEngine("mistral-7b-v0.3.gguf")
response = engine.generate("Hello, world!", max_tokens=256)
print(response)
\`\`\`

### Performance Comparison

| Method | VRAM Required | Speed |
|--------|---------------|-------|
| Full Load | 14 GB | ~40 tok/s |
| AirLLM | 2 GB | ~8 tok/s |
| GGML Q4 | 4 GB | ~25 tok/s |

> **Note:** Results vary depending on hardware and quantization level.

This approach reduces VRAM requirements from **14 GB → 2 GB** for a 7B model, making it possible to run on consumer GPUs like the RTX 3060.`;

// ═════════════════════════════════════════════════════════════════
// Main Page
// ═════════════════════════════════════════════════════════════════
export default function HomePage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [isTyping, setIsTyping] = useState(false);

  const handleSendMessage = useCallback((content: string) => {
    const userMsg: Message = {
      id: `msg-${Date.now()}-user`,
      role: "user",
      content,
    };
    setMessages((prev) => [...prev, userMsg]);
    setIsTyping(true);

    setTimeout(() => {
      const aiMsg: Message = {
        id: `msg-${Date.now()}-ai`,
        role: "assistant",
        content: mockResponse,
      };
      setMessages((prev) => [...prev, aiMsg]);
      setIsTyping(false);
    }, 1200 + Math.random() * 800);
  }, []);

  const handleNewChat = useCallback(() => {
    setMessages([]);
    setActiveChatId(null);
  }, []);

  const handleChatSelect = useCallback((id: string) => {
    setActiveChatId(id);
    setMessages([
      {
        id: "hist-1",
        role: "user",
        content: "Can you explain how layer-wise inference works in AirLLM?",
      },
      {
        id: "hist-2",
        role: "assistant",
        content: `## Layer-wise Inference in AirLLM

Layer-wise inference is the core innovation of AirLLM. Instead of loading the entire model into GPU memory, we process one transformer layer at a time:

1. **Load Layer** — Memory-map the layer's weights from disk
2. **Forward Pass** — Run the input through the layer on GPU
3. **Store Activations** — Keep the output in a buffer
4. **Unload Layer** — Release GPU memory
5. **Repeat** — Move to the next layer

\`\`\`python
for layer in model.layers:
    weights = mmap_load(layer.path)
    activations = layer.forward(activations, weights)
    del weights  # Free GPU memory immediately
\`\`\`

This approach reduces VRAM requirements from **14 GB → 2 GB** for a 7B parameter model, making it possible to run on consumer GPUs like the RTX 3060.

### Trade-offs

| Aspect | Traditional | Layer-wise |
|--------|------------|------------|
| VRAM | High (14GB+) | Low (2GB) |
| Speed | ~40 tok/s | ~8 tok/s |
| Flexibility | Limited by VRAM | Any model size |

The speed reduction is acceptable for many use cases, especially when running locally without internet dependency.`,
      },
    ]);
  }, []);

  return (
    <div className="app relative">
      <div className="text-[var(--color-gray-100)] bg-[var(--color-gray-900)] h-screen max-h-screen overflow-auto flex flex-row justify-end">
        {/* Sidebar — Open WebUI uses var(--sidebar-width), typically 260px */}
        <div
          className="h-screen max-h-screen min-h-screen select-none shrink-0 fixed top-0 left-0 overflow-x-hidden z-50"
          style={{ width: "260px" }}
        >
          <Sidebar
            activeChatId={activeChatId}
            onNewChat={handleNewChat}
            onChatSelect={handleChatSelect}
          />
          {/* Resize handle */}
          <div className="absolute right-0 top-0 bottom-0 w-px bg-[var(--color-gray-50)]/0 dark:bg-[var(--color-gray-850)]/30 hover:bg-[var(--color-gray-200)] dark:hover:bg-[var(--color-gray-800)] transition z-20 cursor-col-resize" />
        </div>

        {/* Main Chat Area — fills remaining space */}
        <div className="flex-1 h-full" style={{ marginLeft: "260px" }}>
          <ChatArea
            messages={messages}
            isTyping={isTyping}
            onSendMessage={handleSendMessage}
          />
        </div>
      </div>
    </div>
  );
}
