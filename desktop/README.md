# BetterAirLLM Next.js Single-Page Control Center

A high-performance, responsive **Next.js Single-Page Desktop Application** that serves as an interactive dashboard and real-time control console for the local BetterAirLLM FastAPI backend.

Built using **Next.js 16 (Turbopack)**, **React 19**, **Tailwind CSS v4**, **Framer Motion**, and **Lucide React** icons.

---

## 🎨 Premium UX & Design Aesthetics

The interface is engineered with a dark-mode-first luxury aesthetic, utilizing a curated color palette and micro-animations to deliver a smooth, high-fidelity experience:

* **Glassmorphic Panels**: Blurs and translucent borders overlaid on a deep carbon backdrop for a modern premium feel.
* **Interactive Telemetry Gauges**: Real-time responsive progress charts visualizing Host RAM, CPU, and VRAM with dynamic glowing accent colors.
* **State Transition Micro-Animations**: Smooth item expansion, modal overlays, and fade-in transitions powered by Framer Motion.
* **Compact High-Density Registry Layout**: Easy-to-scan grid system with status badges indicating loading states, model backend, and framework status.

---

## 🛠️ Key Architectural Sections

### 1. Host & GPU Telemetry

Monitors real-time telemetry polling `/v1/runtime` every few seconds:

* **CPU Metric Card**: Tracks logical/physical core count alongside real-time overall CPU usage percentage.
* **System RAM Metric Card**: Real-time active memory usage, available buffer capacity, and active memory pressure percent.
* **GPU VRAM Metric Card**: Compatible with NVIDIA CUDA allocation metrics or Apple Silicon MPS unified memory pools, with dynamic gauge coloring depending on utilization.

### 2. Live Cache Controller & Eviction Console

Visualizes active models currently occupying system hardware resources:

* Displays loaded models, backend types, dynamic loading arguments, and chronological load timestamps.
* One-click **Unload** command (`POST /v1/models/{model_id}/unload`) to clean Python references and invoke hardware cache flushes (e.g., `torch.cuda.empty_cache()` or `torch.mps.empty_cache()`) instantly.

### 3. Unified Registry card deck

Unifies both downloaded local/remote Safetensors weights and active Ollama daemon instances:

* Dynamic scanning of the Hugging Face hub repositories and local tags via the Ollama endpoint proxy.
* Pre-caching controls (`POST /v1/models/{model_id}/load`) to warm up caches before starting text generation sessions.

### 4. Real-time Chat & Streaming Playground

* Supports real-time server-sent events (SSE) chat streaming (`POST /v1/chat/completions`) using native Next.js stream processing.
* Features interactive styling including full Markdown rendering, code snippets syntax highlighting, adjustable token generation limits, and chat session history wiping.

---

## ⚙️ Environmental Configuration

You can customize the API connection address and runtime parameters via environment variables. Create a `.env.local` file in the `desktop/` directory:

```env
# Custom local server endpoint override (defaults to http://localhost:8000)
NEXT_PUBLIC_BETTERAIRLLM_API_BASE_URL=http://127.0.0.1:8000
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have **Node.js (v18+)** and **npm** installed.

### Installation

1. Navigate to the desktop app folder:

   ```bash
   cd desktop
   ```

2. Install local workspace dependencies:

   ```bash
   npm install
   ```

### Development Server

Launch Next.js in development mode with Turbopack acceleration enabled:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your web browser. The app will automatically connect to your FastAPI backend server at `http://localhost:8000`.

### Production Deployment

To compile a highly optimized, fully tree-shaken static production bundle:

```bash
npm run build
```

To run the built production bundle:

```bash
npm run start
```

---

## 💡 Troubleshooting

* **Connection Status is Offline**: Verify that your FastAPI backend is active and running (`python server.py`). Check for CORS warnings in your web browser console.
* **Host RAM/VRAM Metrics Empty**: Ensure you are running on an operating system with fully configured system drivers (NVIDIA driver for CUDA, standard Apple Metal libraries for MPS).
* **Port Conflict**: If port `3000` is already in use, you can run the app on a custom port:

  ```bash
  npx next dev -p 3001
  ```
