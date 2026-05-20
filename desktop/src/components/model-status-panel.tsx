"use client";

import React, { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Database,
  Loader2,
  RefreshCw,
  Server,
} from "lucide-react";

type ModelItem = {
  id: string;
  source?: string | null;
  backend?: string | null;
  format?: string | null;
  description?: string | null;
  metadata?: Record<string, unknown>;
};

type ModelsResponse = {
  data?: ModelItem[];
};

type HealthResponse = {
  loaded_model?: string | null;
  ollama?: {
    daemon_available?: boolean;
    version?: string | null;
    discovered_model_count?: number;
    running_model_count?: number;
    last_discovery_error?: string | null;
    last_refresh_used_stale?: boolean;
    warning?: string | null;
    base_url?: string;
  };
};

type CapabilitiesResponse = {
  configured_model_count?: number;
  hf_architecture_families?: Array<{ id: string; stability?: string }>;
  experimental_features?: Array<{ id: string; status?: string }>;
  known_limitations?: string[];
};

type PreflightResponse = {
  status?: "ok" | "warning" | "blocked" | string;
  warnings?: string[];
  blockers?: string[];
};

const API_BASE =
  process.env.NEXT_PUBLIC_AIRLLM_API_BASE_URL?.replace(/\/$/, "") ||
  "http://localhost:8000";

function StatusPill({ status }: { status?: string }) {
  const normalized = status || "unknown";
  const tone =
    normalized === "ok"
      ? "text-emerald-300 bg-emerald-500/10 border-emerald-500/20"
      : normalized === "blocked"
        ? "text-red-300 bg-red-500/10 border-red-500/20"
        : "text-amber-300 bg-amber-500/10 border-amber-500/20";

  return (
    <span className={`inline-flex items-center rounded-md border px-2 py-0.5 text-xs ${tone}`}>
      {normalized}
    </span>
  );
}

function Stat({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div className="min-w-0 rounded-lg border border-[var(--color-gray-800)] bg-[var(--color-gray-950)] p-3">
      <div className="mb-2 flex items-center gap-2 text-xs text-[var(--color-gray-500)]">
        {icon}
        <span>{label}</span>
      </div>
      <div className="truncate text-sm font-medium text-white">{value}</div>
    </div>
  );
}

export default function ModelStatusPanel() {
  const [models, setModels] = useState<ModelItem[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [capabilities, setCapabilities] = useState<CapabilitiesResponse | null>(null);
  const [preflights, setPreflights] = useState<Record<string, PreflightResponse>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const [modelsResponse, healthResponse, capabilitiesResponse] = await Promise.all([
        fetch(`${API_BASE}/v1/models`),
        fetch(`${API_BASE}/health`),
        fetch(`${API_BASE}/v1/capabilities`),
      ]);
      if (!modelsResponse.ok) throw new Error(`models ${modelsResponse.status}`);
      if (!healthResponse.ok) throw new Error(`health ${healthResponse.status}`);
      if (!capabilitiesResponse.ok) throw new Error(`capabilities ${capabilitiesResponse.status}`);

      const modelsJson = (await modelsResponse.json()) as ModelsResponse;
      const modelList = modelsJson.data || [];
      setModels(modelList);
      setHealth((await healthResponse.json()) as HealthResponse);
      setCapabilities((await capabilitiesResponse.json()) as CapabilitiesResponse);

      const preflightPairs = await Promise.all(
        modelList.map(async (model) => {
          try {
            const response = await fetch(
              `${API_BASE}/v1/models/${encodeURIComponent(model.id)}/preflight`,
            );
            return [model.id, (await response.json()) as PreflightResponse] as const;
          } catch (preflightError) {
            return [
              model.id,
              {
                status: "warning",
                warnings: [
                  preflightError instanceof Error
                    ? preflightError.message
                    : "Preflight request failed",
                ],
              },
            ] as const;
          }
        }),
      );
      setPreflights(Object.fromEntries(preflightPairs));
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Unable to load server status");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void load();
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  const grouped = useMemo(() => {
    const airllm = models.filter((model) => model.backend !== "ollama");
    const ollama = models.filter((model) => model.backend === "ollama");
    return { airllm, ollama };
  }, [models]);

  return (
    <div className="space-y-4 text-sm">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h3 className="text-base font-semibold text-white">Models</h3>
          <p className="mt-0.5 text-xs text-[var(--color-gray-500)]">
            Live registry, Ollama discovery, and preflight readiness.
          </p>
        </div>
        <button
          onClick={() => void load()}
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-[var(--color-gray-800)] hover:bg-[var(--color-gray-850)]"
          title="Refresh model status"
        >
          {loading ? (
            <Loader2 className="h-4 w-4 animate-spin text-[var(--color-gray-400)]" />
          ) : (
            <RefreshCw className="h-4 w-4 text-[var(--color-gray-400)]" />
          )}
        </button>
      </div>

      {error && (
        <div className="flex items-start gap-2 rounded-lg border border-red-500/20 bg-red-500/10 p-3 text-red-200">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>Server status unavailable: {error}</span>
        </div>
      )}

      <div className="grid grid-cols-2 gap-3">
        <Stat
          icon={<Database className="h-3.5 w-3.5" />}
          label="Visible models"
          value={models.length}
        />
        <Stat
          icon={<Cpu className="h-3.5 w-3.5" />}
          label="Loaded"
          value={health?.loaded_model || "none"}
        />
        <Stat
          icon={<Server className="h-3.5 w-3.5" />}
          label="Ollama"
          value={health?.ollama?.daemon_available ? "available" : "unavailable"}
        />
        <Stat
          icon={<CheckCircle2 className="h-3.5 w-3.5" />}
          label="HF families"
          value={capabilities?.hf_architecture_families?.length ?? 0}
        />
      </div>

      {(health?.ollama?.warning || health?.ollama?.last_discovery_error) && (
        <div className="rounded-lg border border-amber-500/20 bg-amber-500/10 p-3 text-xs text-amber-100">
          {health.ollama.warning || health.ollama.last_discovery_error}
        </div>
      )}

      <ModelGroup title="BetterAirLLM" models={grouped.airllm} preflights={preflights} />
      <ModelGroup title="Ollama" models={grouped.ollama} preflights={preflights} />

      {capabilities?.known_limitations?.length ? (
        <div className="rounded-lg border border-[var(--color-gray-800)] p-3">
          <div className="mb-2 text-xs font-medium text-[var(--color-gray-400)]">
            Known limitations
          </div>
          <div className="space-y-1 text-xs text-[var(--color-gray-500)]">
            {capabilities.known_limitations.slice(0, 3).map((item) => (
              <div key={item}>{item}</div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function ModelGroup({
  title,
  models,
  preflights,
}: {
  title: string;
  models: ModelItem[];
  preflights: Record<string, PreflightResponse>;
}) {
  return (
    <div>
      <div className="mb-2 text-xs font-medium text-[var(--color-gray-500)]">
        {title} ({models.length})
      </div>
      <div className="space-y-2">
        {models.length === 0 ? (
          <div className="rounded-lg border border-[var(--color-gray-800)] p-3 text-xs text-[var(--color-gray-600)]">
            No models discovered.
          </div>
        ) : (
          models.map((model) => {
            const preflight = preflights[model.id];
            const issues = [...(preflight?.blockers || []), ...(preflight?.warnings || [])];
            return (
              <div
                key={model.id}
                className="rounded-lg border border-[var(--color-gray-800)] bg-[var(--color-gray-950)] p-3"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="truncate font-medium text-white">{model.id}</div>
                    <div className="mt-1 flex flex-wrap gap-1.5 text-xs text-[var(--color-gray-500)]">
                      <span>{model.backend || "airllm"}</span>
                      <span>{model.source || "hf"}</span>
                      <span>{model.format || "checkpoint"}</span>
                    </div>
                  </div>
                  <StatusPill status={preflight?.status} />
                </div>
                {model.description && (
                  <div className="mt-2 text-xs text-[var(--color-gray-500)]">
                    {model.description}
                  </div>
                )}
                {issues.length > 0 && (
                  <div className="mt-2 space-y-1 text-xs text-amber-200">
                    {issues.slice(0, 2).map((issue) => (
                      <div key={issue} className="line-clamp-1">
                        {issue}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
