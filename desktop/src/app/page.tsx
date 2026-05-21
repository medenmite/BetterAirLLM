"use client";

import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  Cpu,
  Database,
  Loader2,
  Play,
  Trash2,
  Check,
  AlertCircle,
  MessageSquare,
  HardDrive,
  RefreshCw,
  Layout,
  Send,
  X,
  Server,
  CloudLightning
} from "lucide-react";
import ReactMarkdown from "react-markdown";


export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
}

type ModelItem = {
  id: string;
  repo_id: string;
  source: string;
  backend: string;
  format: string;
  description: string;
  support?: {
    family?: string;
    prompt_format?: string;
    status?: string;
    tested_level?: string;
    quantization?: string | null;
    memory?: {
      active_parameters?: string;
      vram_loaded?: string;
    };
  };
  metadata?: {
    context_length?: number;
    quantization_level?: string;
    memory?: {
      active_parameters?: string;
      vram_loaded?: string;
    };
  };
};

type RuntimeResponse = {
  loaded_model?: string | null;
  model_manager?: {
    loaded_model_ids: string[];
    loaded_models: Array<{
      id: string;
      backend: string;
      repo_id: string;
      loaded_at: number;
      last_used_at: number;
      load_kwargs?: Record<string, unknown>;
    }>;
    loads: number;
    reuses: number;
    evictions: number;
    last_error?: string | null;
  };
  hardware?: {
    device: string;
    cuda_available: boolean;
    mps_available?: boolean;
    torch_version?: string;
    os?: {
      system: string;
      release: string;
      version: string;
      machine: string;
      python_version: string;
    };
    cuda?: {
      device_index: number;
      name: string;
      total_vram_mb: number;
      allocated_mb: number;
      reserved_mb: number;
      max_allocated_mb: number;
    };
    mps?: {
      name: string;
      allocated_mb?: number;
      driver_allocated_mb?: number;
    };
    workspace_disk?: {
      path: string;
      total_mb: number;
      free_mb: number;
      used_mb: number;
    };
    ram?: {
      total_mb: number;
      available_mb: number;
      used_mb: number;
      percent: number;
    };
    cpu?: {
      percent: number;
      cores_physical?: number;
      cores_logical?: number;
      cores?: number;
    };
  };
};

const API_BASE =
  process.env.NEXT_PUBLIC_BETTERAIRLLM_API_BASE_URL?.replace(/\/$/, "") ||
  process.env.NEXT_PUBLIC_AIRLLM_API_BASE_URL?.replace(/\/$/, "") ||
  "http://localhost:8000";


export default function Page() {
  const [activeTab, setActiveTab] = useState<"dashboard" | "chat">("dashboard");
  const [models, setModels] = useState<ModelItem[]>([]);
  const [runtime, setRuntime] = useState<RuntimeResponse | null>(null);
  const [connected, setConnected] = useState(false);
  const [polling, setPolling] = useState(true);


  const [loadingModels, setLoadingModels] = useState<Record<string, boolean>>({});
  const [unloadingModels, setUnloadingModels] = useState<Record<string, boolean>>({});
  const [actionMessage, setActionMessage] = useState<{
    text: string;
    type: "success" | "error" | "info";
  } | null>(null);


  const [selectedModel, setSelectedModel] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  const scrollRef = useRef<HTMLDivElement>(null);
  const toastTimeoutRef = useRef<number | null>(null);


  const showToast = useCallback((text: string, type: "success" | "error" | "info" = "info") => {
    if (toastTimeoutRef.current) window.clearTimeout(toastTimeoutRef.current);
    setActionMessage({ text, type });
    toastTimeoutRef.current = window.setTimeout(() => {
      setActionMessage(null);
    }, 4000);
  }, []);


  const refreshStats = useCallback(async (quiet = false) => {
    try {

      const runtimeRes = await fetch(`${API_BASE}/v1/runtime`);
      if (runtimeRes.ok) {
        const data = (await runtimeRes.json()) as RuntimeResponse;
        setRuntime(data);
        setConnected(true);
      } else {
        setConnected(false);
      }


      const modelsRes = await fetch(`${API_BASE}/v1/models`);
      if (modelsRes.ok) {
        const data = await modelsRes.json();
        const modelList = (data.data || []) as ModelItem[];
        setModels(modelList);


        if (modelList.length > 0 && !selectedModel) {
          setSelectedModel(modelList[0].id);
        }
      }
    } catch (err) {
      setConnected(false);
      if (!quiet) {
        console.warn("BetterAirLLM API server is offline.", err);
      }
    }
  }, [selectedModel]);


  useEffect(() => {
    const initialRefresh = window.setTimeout(() => {
      void refreshStats(true);
    }, 0);
    let interval: number | null = null;
    if (polling) {
      interval = window.setInterval(() => {
        void refreshStats(true);
      }, 3000);
    }
    return () => {
      window.clearTimeout(initialRefresh);
      if (interval) window.clearInterval(interval);
      if (toastTimeoutRef.current) window.clearTimeout(toastTimeoutRef.current);
    };
  }, [polling, refreshStats]);


  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isTyping]);


  const handleLoadModel = async (modelId: string) => {
    setLoadingModels((prev) => ({ ...prev, [modelId]: true }));
    showToast(`Initiating load of model '${modelId}'...`, "info");
    try {
      const res = await fetch(`${API_BASE}/v1/models/${encodeURIComponent(modelId)}/load`, {
        method: "POST",
      });
      const data = await res.json();
      if (res.ok) {
        showToast(`Loaded model successfully: ${modelId}`, "success");
        await refreshStats(true);
      } else {
        showToast(data.detail || "Load failed", "error");
      }
    } catch {
      showToast("Could not communicate with local server", "error");
    } finally {
      setLoadingModels((prev) => ({ ...prev, [modelId]: false }));
    }
  };


  const handleUnloadModel = async (modelId: string) => {
    setUnloadingModels((prev) => ({ ...prev, [modelId]: true }));
    showToast(`Unloading model '${modelId}'...`, "info");
    try {
      const res = await fetch(`${API_BASE}/v1/models/${encodeURIComponent(modelId)}/unload`, {
        method: "POST",
      });
      const data = await res.json();
      if (res.ok) {
        showToast(`Unloaded model: ${modelId}`, "success");
        await refreshStats(true);
      } else {
        showToast(data.detail || "Unload failed", "error");
      }
    } catch {
      showToast("Could not communicate with local server", "error");
    } finally {
      setUnloadingModels((prev) => ({ ...prev, [modelId]: false }));
    }
  };


  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;
    if (!connected) {
      showToast("API server is offline. Start the BetterAirLLM server first.", "error");
      return;
    }

    const promptText = inputValue.trim();
    setInputValue("");

    const userMsg: Message = {
      id: `msg-${Date.now()}-user`,
      role: "user",
      content: promptText,
    };

    const assistantMsg: Message = {
      id: `msg-${Date.now()}-assistant`,
      role: "assistant",
      content: "",
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsTyping(true);

    try {
      const response = await fetch(`${API_BASE}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: selectedModel,
          messages: [...messages, userMsg].map(({ role, content }) => ({ role, content })),
          stream: true,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `Error code: ${response.status}`);
      }

      if (!response.body) {
        throw new Error("Streaming body not readable");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let streamBuffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        streamBuffer += decoder.decode(value, { stream: true });
        const lines = streamBuffer.split("\n");
        streamBuffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          if (trimmed === "data: [DONE]") continue;

          if (trimmed.startsWith("data: ")) {
            try {
              const payload = JSON.parse(trimmed.substring(6));
              const chunk = payload.choices?.[0]?.delta?.content || "";
              if (chunk) {
                setMessages((prev) => {
                  const updated = [...prev];
                  const lastIndex = updated.length - 1;
                  if (lastIndex >= 0 && updated[lastIndex].role === "assistant") {
                    updated[lastIndex] = {
                      ...updated[lastIndex],
                      content: updated[lastIndex].content + chunk,
                    };
                  }
                  return updated;
                });
              }
            } catch {
              console.warn("Failed to parse stream token:", trimmed);
            }
          }
        }
      }
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : "Failed to get reply";
      showToast(errorMessage, "error");
      setMessages((prev) => {
        const updated = [...prev];
        const lastIndex = updated.length - 1;
        if (lastIndex >= 0 && updated[lastIndex].role === "assistant") {
          updated[lastIndex] = {
            ...updated[lastIndex],
            content: `**Error:** ${errorMessage || "Could not complete text generation request."}`,
          };
        }
        return updated;
      });
    } finally {
      setIsTyping(false);
      void refreshStats(true);
    }
  };


  const hfModels = models.filter((m) => m.backend !== "ollama");
  const ollamaModels = models.filter((m) => m.backend === "ollama");
  const loadedModels = runtime?.model_manager?.loaded_models || [];
  const loadedModelIds = runtime?.model_manager?.loaded_model_ids || [];


  const cudaInfo = runtime?.hardware?.cuda;
  const mpsInfo = runtime?.hardware?.mps;
  const ramInfo = runtime?.hardware?.ram;
  const cpuInfo = runtime?.hardware?.cpu;
  const diskInfo = runtime?.hardware?.workspace_disk;
  const osInfo = runtime?.hardware?.os;
  const cpuCores = cpuInfo?.cores_logical ?? cpuInfo?.cores ?? 0;


  const hasGpu = !!cudaInfo || !!mpsInfo;
  const gpuName = cudaInfo?.name ?? mpsInfo?.name ?? "CPU Only (No GPU)";
  const gpuAllocatedMb = cudaInfo?.allocated_mb ?? mpsInfo?.allocated_mb ?? 0;
  const gpuTotalMb = cudaInfo?.total_vram_mb ?? (mpsInfo ? (ramInfo?.total_mb ?? 0) : 0);
  const gpuPercent = gpuTotalMb > 0 ? Math.min(100, Math.round((gpuAllocatedMb / gpuTotalMb) * 100)) : 0;
  const gpuLabel = cudaInfo ? "CUDA VRAM" : mpsInfo ? "MPS Unified Memory" : "GPU Memory";


  const osBadge = osInfo
    ? osInfo.system === "Darwin"
      ? `macOS ${osInfo.release}`
      : osInfo.system === "Windows"
        ? `Windows ${osInfo.release}`
        : `${osInfo.system} ${osInfo.release}`
    : null;

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#0d0d0d] font-sans text-neutral-200">

      <aside className="w-64 shrink-0 border-r border-neutral-800 bg-[#121212]/90 flex flex-col justify-between p-4 relative z-10">
        <div>
          <div className="flex items-center gap-3 mb-6 px-1">
            <div className="relative">
              <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-indigo-500 via-purple-600 to-pink-500 flex items-center justify-center shadow-lg shadow-purple-950/40">
                <CloudLightning className="h-5 w-5 text-white" />
              </div>
              <span className={`absolute -bottom-0.5 -right-0.5 h-3 w-3 rounded-full border-2 border-[#121212] ${connected ? "bg-emerald-500" : "bg-rose-500"}`} />
            </div>
            <div>
              <h1 className="font-semibold text-white tracking-wide text-sm font-primary leading-tight">BetterAirLLM</h1>
              <p className="text-[10px] text-neutral-500 tracking-wider">LOCAL CONTROL CENTER</p>
            </div>
          </div>

          <div className="space-y-1.5">
            <button
              onClick={() => setActiveTab("dashboard")}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition duration-200 ${
                activeTab === "dashboard"
                  ? "bg-neutral-800 text-white shadow-md border border-neutral-700/50"
                  : "text-neutral-400 hover:bg-neutral-900 hover:text-neutral-200"
              }`}
            >
              <Layout className="h-4.5 w-4.5" />
              <span>Metrics & Models</span>
            </button>
            <button
              onClick={() => setActiveTab("chat")}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition duration-200 ${
                activeTab === "chat"
                  ? "bg-neutral-800 text-white shadow-md border border-neutral-700/50"
                  : "text-neutral-400 hover:bg-neutral-900 hover:text-neutral-200"
              }`}
            >
              <MessageSquare className="h-4.5 w-4.5" />
              <span>Chat Playground</span>
            </button>
          </div>
        </div>

        <div className="rounded-xl bg-neutral-900/50 border border-neutral-800/80 p-3 space-y-2 text-xs">
          <div className="flex items-center justify-between">
            <span className="text-neutral-500">API Server</span>
            <button
              onClick={() => void refreshStats()}
              className="p-1 rounded-lg hover:bg-neutral-800 text-neutral-400 transition"
              title="Force Refresh Data"
            >
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
          </div>
          <div className="flex items-center gap-2">
            <span className={`h-2 w-2 rounded-full ${connected ? "bg-emerald-500" : "bg-rose-500"}`} />
            <span className="font-mono text-neutral-300 truncate">{API_BASE.replace(/^https?:\/\//, "")}</span>
          </div>
          <div className="pt-1 text-[10px] text-neutral-500 leading-normal">
            {connected ? (
              <span className="text-emerald-400">Connected and ready. Telemetry polling active.</span>
            ) : (
              <span className="text-rose-400 font-medium">Offline. Verify python server is running.</span>
            )}
          </div>
        </div>
      </aside>

      <main className="flex-1 flex flex-col bg-[#090909] overflow-hidden">

        {actionMessage && (
          <div className="fixed top-4 right-4 z-50 animate-in fade-in slide-in-from-top-4 duration-300">
            <div className={`flex items-center gap-3 px-4 py-3 rounded-xl border shadow-xl backdrop-blur-md ${
              actionMessage.type === "success"
                ? "bg-emerald-950/80 border-emerald-500/30 text-emerald-200"
                : actionMessage.type === "error"
                  ? "bg-rose-950/80 border-rose-500/30 text-rose-200"
                  : "bg-neutral-900/90 border-neutral-700/50 text-neutral-200"
            }`}>
              {actionMessage.type === "success" && <Check className="h-4.5 w-4.5 text-emerald-400 shrink-0" />}
              {actionMessage.type === "error" && <AlertCircle className="h-4.5 w-4.5 text-rose-400 shrink-0" />}
              {actionMessage.type === "info" && <Loader2 className="h-4.5 w-4.5 text-indigo-400 animate-spin shrink-0" />}
              <span className="text-xs font-medium">{actionMessage.text}</span>
              <button onClick={() => setActionMessage(null)} className="ml-2 hover:opacity-80">
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        )}

        {activeTab === "dashboard" && (
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <h2 className="text-xl font-semibold text-white font-primary tracking-wide">System Metrics & Models</h2>
                <div className="flex items-center gap-2 mt-1">
                  <p className="text-xs text-neutral-500">Live hardware allocation logs, registry states, and runtime actions.</p>
                  {osBadge && (
                    <span className="inline-flex items-center gap-1 rounded-md border border-neutral-700/60 bg-neutral-800/50 px-2 py-0.5 text-[10px] font-mono text-neutral-400">
                      {osInfo?.machine === "arm64" ? "🍎" : osInfo?.system === "Windows" ? "🪟" : "🐧"} {osBadge}
                    </span>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-neutral-400">Autopoll (3s)</span>
                <button
                  onClick={() => setPolling(!polling)}
                  className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${polling ? "bg-indigo-600" : "bg-neutral-700"}`}
                >
                  <span className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${polling ? "translate-x-4" : "translate-x-0"}`} />
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="rounded-2xl border border-neutral-800 bg-[#121212]/50 p-4 flex flex-col justify-between min-h-[140px] hover:border-neutral-700/80 transition duration-200">
                <div className="flex justify-between items-start">
                  <div className="space-y-1">
                    <span className="text-xs text-neutral-500 font-medium">{gpuLabel}</span>
                    <h3 className="text-lg font-bold text-white tracking-tight">
                      {hasGpu ? `${gpuAllocatedMb} MB` : "No Device"}
                    </h3>
                  </div>
                  <div className="h-9 w-9 rounded-xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center">
                    <Database className="h-4.5 w-4.5 text-purple-400" />
                  </div>
                </div>
                <div className="mt-4 space-y-2">
                  <div className="flex justify-between text-[11px] text-neutral-400">
                    <span className="truncate">{gpuName}</span>
                    <span>{gpuPercent}%</span>
                  </div>
                  <div className="h-2 w-full bg-neutral-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-purple-500 to-indigo-500 transition-all duration-500"
                      style={{ width: `${gpuPercent}%` }}
                    />
                  </div>
                  {cudaInfo && (
                    <div className="flex justify-between text-[10px] text-neutral-500">
                      <span>Max Alloc: {cudaInfo.max_allocated_mb} MB</span>
                      <span>Total: {cudaInfo.total_vram_mb} MB</span>
                    </div>
                  )}
                  {!cudaInfo && mpsInfo && (
                    <div className="flex justify-between text-[10px] text-neutral-500">
                      <span>Driver Alloc: {mpsInfo.driver_allocated_mb ?? "N/A"} MB</span>
                      <span>Shared with System RAM</span>
                    </div>
                  )}
                  {!hasGpu && (
                    <div className="text-[10px] text-neutral-600">Models will run on CPU only. Consider enabling CUDA or MPS.</div>
                  )}
                </div>
              </div>

              <div className="rounded-2xl border border-neutral-800 bg-[#121212]/50 p-4 flex flex-col justify-between min-h-[140px] hover:border-neutral-700/80 transition duration-200">
                <div className="flex justify-between items-start">
                  <div className="space-y-1">
                    <span className="text-xs text-neutral-500 font-medium">System RAM</span>
                    <h3 className="text-lg font-bold text-white tracking-tight">
                      {ramInfo ? `${Math.round(ramInfo.used_mb / 1024)} GB` : "No Data"}
                    </h3>
                  </div>
                  <div className="h-9 w-9 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
                    <HardDrive className="h-4.5 w-4.5 text-indigo-400" />
                  </div>
                </div>
                <div className="mt-4 space-y-2">
                  <div className="flex justify-between text-[11px] text-neutral-400">
                    <span>Memory Usage</span>
                    <span>{ramInfo ? `${ramInfo.percent}%` : "0%"}</span>
                  </div>
                  <div className="h-2 w-full bg-neutral-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-indigo-500 to-blue-500 transition-all duration-500"
                      style={{ width: ramInfo ? `${ramInfo.percent}%` : "0%" }}
                    />
                  </div>
                  {ramInfo && (
                    <div className="flex justify-between text-[10px] text-neutral-500">
                      <span>Used: {Math.round(ramInfo.used_mb)} MB</span>
                      <span>Total: {Math.round(ramInfo.total_mb)} MB</span>
                    </div>
                  )}
                </div>
              </div>

              <div className="rounded-2xl border border-neutral-800 bg-[#121212]/50 p-4 flex flex-col justify-between min-h-[140px] hover:border-neutral-700/80 transition duration-200">
                <div className="flex justify-between items-start">
                  <div className="space-y-1">
                    <span className="text-xs text-neutral-500 font-medium">Processor Load</span>
                    <h3 className="text-lg font-bold text-white tracking-tight">
                      {cpuInfo ? `${Math.round(cpuInfo.percent)}%` : "No Data"}
                    </h3>
                  </div>
                  <div className="h-9 w-9 rounded-xl bg-pink-500/10 border border-pink-500/20 flex items-center justify-center">
                    <Cpu className="h-4.5 w-4.5 text-pink-400" />
                  </div>
                </div>
                <div className="mt-4 space-y-2">
                  <div className="flex justify-between text-[11px] text-neutral-400">
                    <span>CPU Util</span>
                    <span>{cpuCores > 0 ? `${cpuCores} Threads` : "0 Cores"}{cpuInfo?.cores_physical ? ` (${cpuInfo.cores_physical}P)` : ""}</span>
                  </div>
                  <div className="h-2 w-full bg-neutral-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-pink-500 to-rose-500 transition-all duration-500"
                      style={{ width: cpuInfo ? `${cpuInfo.percent}%` : "0%" }}
                    />
                  </div>
                  {diskInfo && (
                    <div className="flex justify-between text-[10px] text-neutral-500">
                      <span>Disk Free: {Math.round(diskInfo.free_mb / 1024)} GB</span>
                      <span>Total Disk: {Math.round(diskInfo.total_mb / 1024)} GB</span>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between border-b border-neutral-800 pb-2">
                <h3 className="text-sm font-semibold text-white font-primary tracking-wide flex items-center gap-2">
                  <Server className="h-4.5 w-4.5 text-indigo-400" />
                  Currently Loaded Models ({loadedModels.length})
                </h3>
              </div>

              {loadedModels.length === 0 ? (
                <div className="rounded-xl border border-dashed border-neutral-800 p-8 text-center text-neutral-500 text-xs">
                  No models are currently loaded in VRAM/CPU memory cache.
                  <br />
                  <span className="text-[10px] text-neutral-600 mt-1 block">Models load dynamically on completions, or can be triggered below.</span>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {loadedModels.map((m) => {
                    const isUnloading = unloadingModels[m.id];
                    return (
                      <div
                        key={m.id}
                        className="rounded-xl border border-neutral-800 bg-[#121212]/30 p-4 flex flex-col justify-between gap-4 hover:border-neutral-700 transition"
                      >
                        <div className="space-y-2">
                          <div className="flex items-start justify-between gap-3">
                            <div>
                              <h4 className="font-semibold text-white text-xs truncate max-w-[240px]">{m.id}</h4>
                              <p className="text-[10px] text-neutral-500 font-mono mt-0.5">{m.repo_id}</p>
                            </div>
                            <span className="inline-flex items-center gap-1 rounded bg-indigo-500/10 border border-indigo-500/20 px-1.5 py-0.5 text-[9px] font-semibold text-indigo-400">
                              <span className="h-1.5 w-1.5 rounded-full bg-indigo-400 animate-pulse" />
                              Active in Cache
                            </span>
                          </div>
                          <div className="grid grid-cols-2 gap-x-2 gap-y-1 text-[10px] text-neutral-400 font-mono">
                            <div>Backend: {m.backend}</div>
                            <div>Loaded: {new Date(m.loaded_at * 1000).toLocaleTimeString()}</div>
                          </div>
                        </div>
                        <div className="flex items-center justify-between">
                          <button
                            onClick={() => {
                              setSelectedModel(m.id);
                              setActiveTab("chat");
                            }}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-white bg-neutral-800 hover:bg-neutral-700 transition"
                          >
                            <MessageSquare className="h-3.5 w-3.5" />
                            <span>Open Chat</span>
                          </button>
                          <button
                            onClick={() => void handleUnloadModel(m.id)}
                            disabled={isUnloading}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-rose-300 hover:text-white bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/20 transition disabled:opacity-50"
                          >
                            {isUnloading ? (
                              <Loader2 className="h-3.5 w-3.5 animate-spin" />
                            ) : (
                              <Trash2 className="h-3.5 w-3.5" />
                            )}
                            <span>Unload</span>
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            <div className="space-y-6">
              <div className="space-y-3">
                <div className="flex items-center justify-between border-b border-neutral-800 pb-2">
                  <h3 className="text-sm font-semibold text-white font-primary tracking-wide flex items-center gap-2">
                    <CloudLightning className="h-4.5 w-4.5 text-indigo-400" />
                    Hugging Face / Safetensors Models ({hfModels.length})
                  </h3>
                  <span className="text-[10px] text-neutral-500 font-mono">Runs via Layer-Wise Streaming</span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {hfModels.map((m) => {
                    const isLoaded = loadedModelIds.includes(m.id);
                    const isLoading = loadingModels[m.id];
                    const isUnloading = unloadingModels[m.id];

                    return (
                      <div
                        key={m.id}
                        className={`rounded-xl border p-4 flex flex-col justify-between gap-4 transition duration-200 ${
                          isLoaded
                            ? "border-indigo-500/30 bg-indigo-500/5 hover:border-indigo-500/50"
                            : "border-neutral-800 bg-[#121212]/20 hover:border-neutral-700"
                        }`}
                      >
                        <div className="space-y-2">
                          <div className="flex items-start justify-between gap-2">
                            <div>
                              <h4 className="font-semibold text-white text-xs tracking-wide">{m.id}</h4>
                              <span className="text-[10px] text-neutral-500 font-mono">{m.repo_id}</span>
                            </div>
                            <span className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[9px] font-semibold ${
                              m.support?.status === "supported"
                                ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400"
                                : "bg-amber-500/10 border-amber-500/20 text-amber-400"
                            }`}>
                              {m.support?.status || "experimental"}
                            </span>
                          </div>
                          <p className="text-[11px] text-neutral-400 line-clamp-2 leading-relaxed">
                            {m.description || "No model description provided in registry JSON."}
                          </p>
                          <div className="flex flex-wrap gap-1.5 pt-1">
                            <span className="bg-neutral-800 text-neutral-400 border border-neutral-700 px-2 py-0.5 rounded text-[10px] font-mono">
                              Context: {m.metadata?.context_length || 2048}
                            </span>
                            {m.metadata?.memory?.active_parameters && (
                              <span className="bg-neutral-800 text-neutral-400 border border-neutral-700 px-2 py-0.5 rounded text-[10px] font-mono">
                                Active: {m.metadata?.memory?.active_parameters}
                              </span>
                            )}
                          </div>
                        </div>

                        <div className="flex items-center justify-between pt-2 border-t border-neutral-800/80">
                          <div className="flex items-center gap-1">
                            {isLoaded ? (
                              <span className="flex items-center gap-1 text-[11px] text-indigo-400 font-medium">
                                <span className="h-1.5 w-1.5 rounded-full bg-indigo-400 animate-pulse" />
                                Active in Cache
                              </span>
                            ) : (
                              <span className="text-[11px] text-neutral-500">Unloaded</span>
                            )}
                          </div>

                          <div className="flex gap-2">
                            {isLoaded ? (
                              <>
                                <button
                                  onClick={() => {
                                    setSelectedModel(m.id);
                                    setActiveTab("chat");
                                  }}
                                  className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 transition"
                                >
                                  <MessageSquare className="h-3.5 w-3.5" />
                                  <span>Chat</span>
                                </button>
                                <button
                                  onClick={() => void handleUnloadModel(m.id)}
                                  disabled={isUnloading}
                                  className="flex items-center justify-center h-8 w-8 rounded-lg text-rose-400 hover:text-rose-200 bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/20 transition disabled:opacity-50"
                                  title="Unload Model"
                                >
                                  {isUnloading ? (
                                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                  ) : (
                                    <Trash2 className="h-3.5 w-3.5" />
                                  )}
                                </button>
                              </>
                            ) : (
                              <button
                                onClick={() => void handleLoadModel(m.id)}
                                disabled={isLoading}
                                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-white bg-neutral-800 hover:bg-neutral-700 border border-neutral-700/50 transition disabled:opacity-50"
                              >
                                {isLoading ? (
                                  <>
                                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                    <span>Loading...</span>
                                  </>
                                ) : (
                                  <>
                                    <Play className="h-3.5 w-3.5 text-neutral-400" />
                                    <span>Load Model</span>
                                  </>
                                )}
                              </button>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between border-b border-neutral-800 pb-2">
                  <h3 className="text-sm font-semibold text-white font-primary tracking-wide flex items-center gap-2">
                    <HardDrive className="h-4.5 w-4.5 text-indigo-400" />
                    Ollama / GGUF Models ({ollamaModels.length})
                  </h3>
                  <span className="text-[10px] text-neutral-500 font-mono">Discovered Local Daemon</span>
                </div>

                {ollamaModels.length === 0 ? (
                  <div className="rounded-xl border border-dashed border-neutral-800 p-8 text-center text-neutral-500 text-xs">
                    No models found via the local Ollama daemon.
                    <br />
                    <span className="text-[10px] text-neutral-600 mt-1 block">Verify Ollama is installed and running on default port 11434.</span>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {ollamaModels.map((m) => (
                      <div
                        key={m.id}
                        className="rounded-xl border border-neutral-800 bg-[#121212]/20 p-4 flex flex-col justify-between gap-4 hover:border-neutral-700 transition"
                      >
                        <div className="space-y-2">
                          <div className="flex items-start justify-between gap-2">
                            <div>
                              <h4 className="font-semibold text-white text-xs tracking-wide">{m.id}</h4>
                              <span className="text-[10px] text-neutral-500 font-mono">ollama tag: {m.repo_id}</span>
                            </div>
                            <span className="inline-flex items-center rounded border border-neutral-800 bg-neutral-900 px-1.5 py-0.5 text-[9px] font-semibold text-neutral-400">
                              OLLAMA
                            </span>
                          </div>
                          <p className="text-[11px] text-neutral-400 leading-relaxed line-clamp-2">
                            {m.description || "Discovered local model installed in Ollama daemon."}
                          </p>
                          <div className="flex flex-wrap gap-1.5 pt-1">
                            <span className="bg-neutral-800 text-neutral-400 border border-neutral-700 px-2 py-0.5 rounded text-[10px] font-mono">
                              Quantization: {m.metadata?.quantization_level || "GGUF"}
                            </span>
                            <span className="bg-neutral-800 text-neutral-400 border border-neutral-700 px-2 py-0.5 rounded text-[10px] font-mono">
                              Context: {m.metadata?.context_length || 2048}
                            </span>
                          </div>
                        </div>

                        <div className="flex items-center justify-between pt-2 border-t border-neutral-800/80">
                          <span className="text-[10px] text-neutral-500">Auto-loads on generation</span>
                          <button
                            onClick={() => {
                              setSelectedModel(m.id);
                              setActiveTab("chat");
                            }}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-white bg-neutral-800 hover:bg-neutral-700 border border-neutral-700/50 transition"
                          >
                            <MessageSquare className="h-3.5 w-3.5" />
                            <span>Select & Chat</span>
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === "chat" && (
          <div className="flex-1 flex flex-col overflow-hidden relative">

            <div className="h-14 border-b border-neutral-800 px-4 flex items-center justify-between bg-[#121212]/30">
              <div className="flex items-center gap-3">
                <span className="text-xs text-neutral-400 font-medium">Active Chat Model:</span>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="bg-neutral-800 border border-neutral-700/50 rounded-xl px-3 py-1.5 text-xs text-white outline-none focus:border-indigo-500 font-medium"
                >
                  {models.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.id} ({m.backend === "ollama" ? "Ollama" : "Layer-wise"})
                    </option>
                  ))}
                  {models.length === 0 && (
                    <option value="">No models available</option>
                  )}
                </select>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => setMessages([])}
                  className="px-2.5 py-1.5 rounded-lg hover:bg-neutral-800 text-neutral-400 hover:text-neutral-200 text-xs transition"
                >
                  Clear History
                </button>
              </div>
            </div>

            <div
              ref={scrollRef}
              className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6 scrollbar-hidden"
            >
              {messages.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto p-4 space-y-4">
                  <div className="h-12 w-12 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">
                    <MessageSquare className="h-6 w-6" />
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-white">BetterAirLLM Chat Sandbox</h3>
                    <p className="text-xs text-neutral-500 mt-1 leading-relaxed">
                      Send prompts to query the selected model. If using a Hugging Face model, the server will load target weights layer-by-layer during completions.
                    </p>
                  </div>
                  {selectedModel && (
                    <div className="rounded-lg bg-neutral-900 border border-neutral-800 p-2.5 text-[11px] font-mono text-neutral-400">
                      Model ID: {selectedModel}
                    </div>
                  )}
                </div>
              ) : (
                <div className="max-w-3xl mx-auto space-y-6">
                  {messages.map((msg) => {
                    const isUser = msg.role === "user";
                    return (
                      <div key={msg.id} className={`flex gap-4 ${isUser ? "justify-end" : ""}`}>
                        {!isUser && (
                          <div className="h-8 w-8 rounded-full border border-neutral-800 bg-neutral-900 shrink-0 flex items-center justify-center overflow-hidden">
                            <CloudLightning className="h-4.5 w-4.5 text-indigo-400" />
                          </div>
                        )}
                        <div className={isUser ? "max-w-[70%]" : "flex-1 min-w-0"}>
                          <div className={`text-[10px] text-neutral-500 mb-1 ${isUser ? "text-right" : ""}`}>
                            {isUser ? "User Prompt" : selectedModel}
                          </div>
                          <div className={`p-4 rounded-2xl ${
                            isUser
                              ? "bg-indigo-600 text-white rounded-tr-none"
                              : "bg-[#121212] border border-neutral-800 text-neutral-200 rounded-tl-none markdown-prose"
                          }`}>
                            {isUser ? (
                              <p className="text-sm whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                            ) : (
                              <ReactMarkdown>{msg.content || "..."}</ReactMarkdown>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                  {isTyping && (
                    <div className="flex gap-4">
                      <div className="h-8 w-8 rounded-full border border-neutral-800 bg-neutral-900 shrink-0 flex items-center justify-center">
                        <Loader2 className="h-4.5 w-4.5 text-indigo-400 animate-spin" />
                      </div>
                      <div className="space-y-1">
                        <div className="text-[10px] text-neutral-500">{selectedModel}</div>
                        <div className="flex items-center gap-1 bg-[#121212] border border-neutral-800 px-3 py-2 rounded-2xl rounded-tl-none">
                          <span className="typing-dot" />
                          <span className="typing-dot" />
                          <span className="typing-dot" />
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="p-4 border-t border-neutral-800 bg-[#0c0c0c] shrink-0">
              <div className="max-w-3xl mx-auto flex gap-3">
                <textarea
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      void handleSendMessage();
                    }
                  }}
                  rows={1}
                  placeholder="Ask a question or request reasoning pass..."
                  className="flex-1 bg-neutral-900 border border-neutral-800/80 focus:border-indigo-500 rounded-xl px-4 py-3 text-sm text-white placeholder-neutral-500 resize-none outline-none max-h-32 min-h-[46px]"
                />
                <button
                  onClick={() => void handleSendMessage()}
                  disabled={!inputValue.trim() || isTyping}
                  className="h-[46px] px-4 rounded-xl font-semibold text-white bg-indigo-600 hover:bg-indigo-500 disabled:bg-neutral-800 disabled:text-neutral-500 transition flex items-center justify-center shrink-0"
                >
                  <Send className="h-4.5 w-4.5" />
                </button>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
