"use client";

import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  SquarePen,
  Search,
  ChevronDown,
  ChevronRight,
  MoreHorizontal,
  Pencil,
  Trash2,
  ArrowRightLeft,
  Settings,
  X,
  PanelLeft,
  StickyNote,
  LayoutGrid,
} from "lucide-react";
import ModelStatusPanel from "@/components/model-status-panel";

// ─────────────────────────── Types ─────────────────────────────
interface ChatItem {
  id: string;
  title: string;
  timeRange?: string;
}

interface FolderGroup {
  id: string;
  name: string;
  chats: ChatItem[];
}

// ─────────────────────────── Mock Data ─────────────────────────
const pinnedChats: ChatItem[] = [
  { id: "pin1", title: "BetterAirLLM Architecture Overview" },
  { id: "pin2", title: "Memory-mapped Tensor Loading" },
];

const todayChats: ChatItem[] = [
  { id: "t1", title: "Optimize CUDA kernel for attention", timeRange: "Today" },
  { id: "t2", title: "Compare LoRA vs QLoRA training", timeRange: "Today" },
  { id: "t3", title: "Debug tokenizer encoding issues", timeRange: "Today" },
];

const yesterdayChats: ChatItem[] = [
  { id: "y1", title: "Fine-tune Mistral 7B on custom data", timeRange: "Yesterday" },
  { id: "y2", title: "Research FlashAttention v3 paper", timeRange: "Yesterday" },
];

const olderChats: ChatItem[] = [
  { id: "o1", title: "GPTQ vs AWQ quantization benchmarks", timeRange: "Previous 7 days" },
  { id: "o2", title: "Export model to GGUF format", timeRange: "Previous 7 days" },
  { id: "o3", title: "Layer-wise loading architecture design", timeRange: "Previous 7 days" },
];

const folders: FolderGroup[] = [
  {
    id: "f1",
    name: "Model Quantization",
    chats: [
      { id: "f1c1", title: "4-bit quantization strategy" },
      { id: "f1c2", title: "Benchmark results Q4_K_M vs Q5_K_M" },
    ],
  },
  {
    id: "f2",
    name: "Inference Pipeline",
    chats: [
      { id: "f2c1", title: "KV-cache optimization notes" },
      { id: "f2c2", title: "Batch inference implementation" },
    ],
  },
];

// ─────────────────────────── Chat Item Row ─────────────────────
function ChatRow({
  chat,
  isActive,
  onClick,
  indent = false,
}: {
  chat: ChatItem;
  isActive: boolean;
  onClick: () => void;
  indent?: boolean;
}) {
  const [isHovered, setIsHovered] = useState(false);
  const [showMenu, setShowMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setShowMenu(false);
      }
    }
    if (showMenu) document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [showMenu]);

  return (
    <div
      id="sidebar-chat-item"
      className={`group relative flex items-center gap-2.5 rounded-xl text-sm cursor-pointer transition ${
        indent ? "ml-3 pl-1 border-l border-[var(--color-gray-900)]" : ""
      } ${
        isActive
          ? "bg-[var(--color-gray-900)] text-white"
          : "text-[var(--color-gray-200)] hover:bg-[var(--color-gray-900)]"
      }`}
      style={{
        minHeight: "32px",
        paddingInline: "11px",
        paddingBlock: "6px",
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => {
        setIsHovered(false);
        if (!showMenu) setShowMenu(false);
      }}
      onClick={onClick}
    >
      <div
        dir="auto"
        className="flex-1"
        style={{ height: "20px", lineHeight: "20px" }}
      >
        <span className="line-clamp-1">{chat.title}</span>
      </div>

      {/* Gradient fade on long text */}
      {!isHovered && !showMenu && (
        <div
          className="absolute right-0 top-0 bottom-0 w-12 pointer-events-none rounded-r-xl"
          style={{
            background: isActive
              ? "linear-gradient(to right, transparent, var(--color-gray-900))"
              : undefined,
          }}
        />
      )}

      {/* 3-dot menu */}
      <AnimatePresence>
        {(isHovered || showMenu) && (
          <motion.button
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.1 }}
            onClick={(e) => {
              e.stopPropagation();
              setShowMenu(!showMenu);
            }}
            className="shrink-0 flex h-6 w-6 items-center justify-center rounded-lg hover:bg-[var(--color-gray-800)] transition z-10"
          >
            <MoreHorizontal className="h-3.5 w-3.5" />
          </motion.button>
        )}
      </AnimatePresence>

      {/* Dropdown Menu */}
      <AnimatePresence>
        {showMenu && (
          <motion.div
            ref={menuRef}
            initial={{ opacity: 0, y: -4, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.95 }}
            transition={{ duration: 0.12 }}
            className="absolute right-0 top-full z-50 mt-1 w-48 rounded-xl bg-[var(--color-gray-850)] border border-[var(--color-gray-800)] py-1 shadow-xl"
          >
            <button
              onClick={(e) => { e.stopPropagation(); setShowMenu(false); }}
              className="flex w-full items-center gap-2 px-3 py-2 text-sm text-[var(--color-gray-300)] hover:bg-[var(--color-gray-800)] transition"
            >
              <ArrowRightLeft className="h-3.5 w-3.5" />
              Move to folder
            </button>
            <button
              onClick={(e) => { e.stopPropagation(); setShowMenu(false); }}
              className="flex w-full items-center gap-2 px-3 py-2 text-sm text-[var(--color-gray-300)] hover:bg-[var(--color-gray-800)] transition"
            >
              <Pencil className="h-3.5 w-3.5" />
              Rename
            </button>
            <div className="my-1 h-px bg-[var(--color-gray-800)]" />
            <button
              onClick={(e) => { e.stopPropagation(); setShowMenu(false); }}
              className="flex w-full items-center gap-2 px-3 py-2 text-sm text-red-400 hover:bg-red-500/10 transition"
            >
              <Trash2 className="h-3.5 w-3.5" />
              Delete
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─────────────────────────── Collapsible Folder ───────────────
function FolderSection({
  name,
  children,
  defaultOpen = false,
  onAdd,
}: {
  name: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
  onAdd?: () => void;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="mt-0.5 px-2">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="group flex w-full items-center gap-1 rounded-lg px-1.5 py-1 text-xs font-medium text-[var(--color-gray-500)] hover:text-[var(--color-gray-300)] transition"
      >
        {isOpen ? (
          <ChevronDown className="h-3 w-3 shrink-0" />
        ) : (
          <ChevronRight className="h-3 w-3 shrink-0" />
        )}
        <span className="flex-1 text-left font-primary">{name}</span>
        {onAdd && (
          <span
            onClick={(e) => { e.stopPropagation(); onAdd(); }}
            className="invisible group-hover:visible p-0.5 hover:bg-[var(--color-gray-800)] rounded-md transition"
          >
            <SquarePen className="h-3 w-3" />
          </span>
        )}
      </button>

      <AnimatePresence initial={false}>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="overflow-hidden"
          >
            {children}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─────────────────────────── Settings Modal ────────────────────
function SettingsModal({
  isOpen,
  onClose,
}: {
  isOpen: boolean;
  onClose: () => void;
}) {
  const tabs = ["General", "Models", "Connections", "Interface", "Audio", "About"];
  const [activeTab, setActiveTab] = useState(0);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm"
            onClick={onClose}
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.2 }}
            className="fixed left-1/2 top-1/2 z-50 -translate-x-1/2 -translate-y-1/2 w-[90vw] max-w-[720px] bg-[var(--color-gray-900)] border border-[var(--color-gray-850)] rounded-2xl shadow-2xl overflow-hidden"
          >
            <div className="flex items-center justify-between px-5 py-4 border-b border-[var(--color-gray-850)]">
              <h2 className="text-lg font-semibold text-white">Settings</h2>
              <button
                onClick={onClose}
                className="flex h-8 w-8 items-center justify-center rounded-full hover:bg-[var(--color-gray-850)] transition"
              >
                <X className="h-4 w-4 text-[var(--color-gray-400)]" />
              </button>
            </div>
            <div className="flex border-b border-[var(--color-gray-850)] overflow-x-auto">
              {tabs.map((tab, i) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(i)}
                  className={`px-4 py-3 text-sm font-medium transition whitespace-nowrap relative ${
                    activeTab === i
                      ? "text-white"
                      : "text-[var(--color-gray-500)] hover:text-[var(--color-gray-300)]"
                  }`}
                >
                  {tab}
                  {activeTab === i && (
                    <motion.div
                      layoutId="settingsTab"
                      className="absolute bottom-0 left-0 right-0 h-0.5 bg-white rounded-full"
                    />
                  )}
                </button>
              ))}
            </div>
            <div className="p-6 min-h-[350px] max-h-[70vh] overflow-y-auto">
              {tabs[activeTab] === "Models" || tabs[activeTab] === "Connections" ? (
                <ModelStatusPanel />
              ) : (
                <div className="flex flex-col items-center justify-center h-[280px] text-[var(--color-gray-600)]">
                  <Settings className="h-12 w-12 mb-3 opacity-20" />
                  <p className="text-sm">{tabs[activeTab]}</p>
                  <p className="text-xs mt-1 opacity-50">Coming soon</p>
                </div>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

// ═════════════════════════════════════════════════════════════════
// ████ Main Sidebar Export ██████████████████████████████████████
// ═════════════════════════════════════════════════════════════════
export default function Sidebar({
  activeChatId,
  onNewChat,
  onChatSelect,
}: {
  activeChatId: string | null;
  onNewChat: () => void;
  onChatSelect: (id: string) => void;
}) {
  const [showSettings, setShowSettings] = useState(false);
  const [scrollTop, setScrollTop] = useState(0);

  return (
    <>
      <div className="flex flex-col h-screen max-h-screen select-none bg-[var(--color-gray-950)]/70 text-[var(--color-gray-200)] text-sm">
        {/* ── Sticky Header ─────────────────────────────────── */}
        <div className="px-[0.5625rem] pt-2 pb-1.5 flex justify-between items-center space-x-1 text-[var(--color-gray-600)] sticky top-0 z-10">
          {/* Logo / Brand */}
          <button
            type="button"
            className="flex items-center rounded-xl h-[34px] justify-center hover:bg-[var(--color-gray-100)]/5 transition px-1.5"
            onClick={onNewChat}
          >
            <img
              src="/favicon.ico"
              className="size-6 rounded-full"
              alt=""
            />
          </button>

          <button
            type="button"
            className="flex flex-1 px-0.5"
            onClick={onNewChat}
          >
            <div className="self-center font-medium text-white font-primary">
              BetterAirLLM
            </div>
          </button>

          <button
            className="flex rounded-xl h-[34px] w-[34px] justify-center items-center hover:bg-[var(--color-gray-100)]/5 transition"
            title="Toggle Sidebar"
          >
            <PanelLeft className="h-[18px] w-[18px]" />
          </button>

          {/* Gradient indicator for scroll */}
          <div
            className={`${scrollTop > 0 ? "visible" : "invisible"} pointer-events-none absolute inset-0 -z-10 -mb-6`}
            style={{
              background: "linear-gradient(to bottom, var(--color-gray-950) 50%, transparent)"
            }}
          />
        </div>

        {/* ── Scrollable Content ─────────────────────────────── */}
        <div
          className="relative flex flex-col flex-1 overflow-y-auto scrollbar-hidden pt-3 pb-3"
          onScroll={(e) => setScrollTop((e.target as HTMLElement).scrollTop)}
        >
          {/* New Chat Button */}
          <div className="pb-1.5">
            <div className="px-[0.4375rem] flex justify-center text-[var(--color-gray-800)] dark:text-[var(--color-gray-200)]">
              <button
                type="button"
                className="group grow flex items-center space-x-3 rounded-2xl px-2.5 py-2 hover:bg-[var(--color-gray-900)] transition outline-none"
                onClick={onNewChat}
              >
                <div className="self-center">
                  <SquarePen className="h-[18px] w-[18px]" strokeWidth={2} />
                </div>
                <div className="flex flex-1 self-center translate-y-[0.5px]">
                  <div className="self-center text-sm font-primary">New Chat</div>
                </div>
              </button>
            </div>

            {/* Search Button */}
            <div className="px-[0.4375rem] flex justify-center text-[var(--color-gray-200)]">
              <button
                className="group grow flex items-center space-x-3 rounded-2xl px-2.5 py-2 hover:bg-[var(--color-gray-900)] transition outline-none"
              >
                <div className="self-center">
                  <Search className="h-[18px] w-[18px]" strokeWidth={2} />
                </div>
                <div className="flex flex-1 self-center translate-y-[0.5px]">
                  <div className="self-center text-sm font-primary">Search</div>
                </div>
              </button>
            </div>

            {/* Pinned Menu Items */}
            <div className="px-[0.4375rem] flex justify-center text-[var(--color-gray-200)]">
              <a
                href="/notes"
                className="grow flex items-center space-x-3 rounded-2xl px-2.5 py-2 hover:bg-[var(--color-gray-900)] transition"
                onClick={(e) => e.preventDefault()}
              >
                <div className="self-center">
                  <StickyNote className="h-[18px] w-[18px]" strokeWidth={2} />
                </div>
                <div className="flex self-center translate-y-[0.5px]">
                  <div className="self-center text-sm font-primary">Notes</div>
                </div>
              </a>
            </div>

            <div className="px-[0.4375rem] flex justify-center text-[var(--color-gray-200)]">
              <a
                href="/workspace"
                className="grow flex items-center space-x-3 rounded-2xl px-2.5 py-2 hover:bg-[var(--color-gray-900)] transition"
                onClick={(e) => e.preventDefault()}
              >
                <div className="self-center">
                  <LayoutGrid className="h-[18px] w-[18px]" strokeWidth={2} />
                </div>
                <div className="flex self-center translate-y-[0.5px]">
                  <div className="self-center text-sm font-primary">Workspace</div>
                </div>
              </a>
            </div>
          </div>

          {/* Folders Section */}
          <FolderSection name="Folders" defaultOpen>
            {folders.map((folder) => (
              <FolderSection key={folder.id} name={folder.name} defaultOpen={false}>
                <div className="ml-3 pl-1 mt-[1px] flex flex-col border-l border-[var(--color-gray-900)]">
                  {folder.chats.map((chat) => (
                    <ChatRow
                      key={chat.id}
                      chat={chat}
                      isActive={chat.id === activeChatId}
                      onClick={() => onChatSelect(chat.id)}
                    />
                  ))}
                </div>
              </FolderSection>
            ))}
          </FolderSection>

          {/* Chats Section */}
          <FolderSection name="Chats" defaultOpen>
            {/* Pinned */}
            {pinnedChats.length > 0 && (
              <div className="mb-1">
                <FolderSection name="Pinned" defaultOpen>
                  <div className="ml-3 pl-1 mt-[1px] flex flex-col border-l border-[var(--color-gray-900)]">
                    {pinnedChats.map((chat) => (
                      <ChatRow
                        key={chat.id}
                        chat={chat}
                        isActive={chat.id === activeChatId}
                        onClick={() => onChatSelect(chat.id)}
                      />
                    ))}
                  </div>
                </FolderSection>
              </div>
            )}

            {/* Time-grouped chats */}
            <div className="flex-1 flex flex-col pt-1.5">
              {[todayChats, yesterdayChats, olderChats].map((group, gi) => (
                <React.Fragment key={gi}>
                  {group.length > 0 && (
                    <>
                      <div className={`w-full pl-2.5 text-xs text-[var(--color-gray-500)] font-medium ${gi === 0 ? "" : "pt-5"} pb-1.5`}>
                        {group[0].timeRange}
                      </div>
                      {group.map((chat) => (
                        <ChatRow
                          key={chat.id}
                          chat={chat}
                          isActive={chat.id === activeChatId}
                          onClick={() => onChatSelect(chat.id)}
                        />
                      ))}
                    </>
                  )}
                </React.Fragment>
              ))}
            </div>
          </FolderSection>
        </div>

        {/* ── Bottom User Area ──────────────────────────────── */}
        <div className="px-1.5 pt-1.5 pb-2 sticky bottom-0 z-10">
          {/* Gradient fade */}
          <div
            className="pointer-events-none absolute inset-0 -z-10 -mt-6"
            style={{
              background: "linear-gradient(to top, var(--color-gray-950) 50%, transparent)"
            }}
          />

          <div className="flex flex-col font-primary">
            <button
              onClick={() => setShowSettings(true)}
              className="flex items-center rounded-2xl py-2 px-1.5 w-full hover:bg-[var(--color-gray-900)]/50 transition"
            >
              <div className="self-center mr-3 relative">
                <div className="flex h-7 w-7 items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-violet-600 text-white text-xs font-semibold">
                  U
                </div>
                <div className="absolute -bottom-0.5 -right-0.5">
                  <span className="relative flex h-2.5 w-2.5">
                    <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-green-500 border-2 border-[var(--color-gray-950)]" />
                  </span>
                </div>
              </div>
              <div className="self-center font-medium text-sm">User</div>
            </button>
          </div>
        </div>
      </div>

      <SettingsModal isOpen={showSettings} onClose={() => setShowSettings(false)} />
    </>
  );
}
