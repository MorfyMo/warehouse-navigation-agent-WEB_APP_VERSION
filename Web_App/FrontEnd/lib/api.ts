// API configuration and utilities for connecting to Python backend
// const API_BASE_URL_backend = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080"
const NEXT_PUBLIC_WS_URL=process.env.NEXT_PUBLIC_WS_URL ?? "wss://warehouse-rl-api.fly.dev"

// to avoid hard coding for things before /api
export const API =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") ?? "";

export const dynamic = 'force-dynamic'; // disables static optimization
// export const revalidate = 0;

export interface TrainingConfig {
  algorithm: "dqn" | "ppo"
  episodes: number
  learning_rate: number
  environment: "navigation" | "warehouse"
  warehouse_config?: {
    total_packages: number
    agent_types: ("normal" | "helper" | "collector")[]
  }
}

export interface TrainingStatus {
  is_training: boolean
  current_episode: number
  episodes: number
  current_reward: number
  progress: number
}

export interface EnvironmentState {
  agent_positions: { x: number; y: number; type: string; id: string }[]
  packages: { x: number; y: number; id: string; status: "shelf" | "carried" | "receiving" | "transfer" }[]
  obstacles: { x: number; y: number; width: number; height: number }[]
  receiving_region: { x: number; y: number; width: number; height: number }
  transfer_region: { x: number; y: number; width: number; height: number }
  shelves: { x: number; y: number; width: number; height: number }[]
}

async function withTimeout(input: RequestInfo, init: RequestInit = {}, timeoutMs = 15000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(new DOMException("Timeout", "TimeoutError")), timeoutMs);
  try {
    const res = await fetch(input, { cache: "no-store", ...init, signal: controller.signal });
    return res;
  } finally {
    clearTimeout(id);
  }
}

export const wsPaths = {
  training: (id: string) => `/ws/training/${id}`,
  plot2d:  (id: string) => `/ws/plot/${id}`,
  plot3d:  (id: string) => `/ws/plot3d/${id}`,
  layout: (id: string) => `/ws/layout/${id}`,
  layout2d: (id:string) => `/ws/vis2d/${id}`
};

// to help centralized the behaviors, we add this fetch function
type FetchOptions = RequestInit & {timeoutMs?: number; signal?: AbortSignal};

async function fetchJSON<T>(url: string, init: FetchOptions={}): Promise<T>{
  const {timeoutMs, signal, ...rest}=init;
  const ctl = new AbortController();

  const onAbort = () =>ctl.abort(signal?.reason ?? undefined);
  if (signal) signal.addEventListener("abort", onAbort);

  const timer = timeoutMs ? setTimeout(()=>ctl.abort(), timeoutMs):null;

  try{
    const response = await fetch(url,{...rest, signal:ctl.signal, cache: "no-store"});
    if(!response.ok) throw new Error(`${response.status}${response.statusText}`);
    return response.json() as Promise<T>;
  } finally {
    if(signal) signal.removeEventListener("abort", onAbort)
    if(timer) clearTimeout(timer);
  }
}

// this function tries to retrieve the result from env_init in "api_server" file
export async function envInit(sessionId:string|null){
  const r = await fetch(`${API_BASE_URL}/api/env_init/${sessionId}`,{
    method: "GET",
    cache: "no-store",
    // credentials: "include",
  });
  if(!r.ok) throw new Error(`env_init ${r.status}`);
  return r.json() as Promise<{is_ready:boolean}>
}

export async function waitForEnv(
  sessionId: string|null,
  opts: {timeoutMs?: number; intervalMs?: number}={},
){
  const timeoutMs = opts.timeoutMs ?? 30_000;
  const intervalMs = opts.intervalMs ?? 500;
  const start = Date.now();
  for (;;){
    const {is_ready} = await envInit(sessionId);
    if(is_ready) return true;
    if (Date.now()-start>timeoutMs){
      throw new Error("env_init timed out");
    }
    await new Promise((r)=> setTimeout(r, intervalMs));
  }
}

// API functions
export const api = {
  // Start training
  async startTraining(config: TrainingConfig,opts?: { signal?: AbortSignal },timeoutMs=180_000): Promise<{ success: boolean; session_id: string }> {
  //   const response = await withTimeout(`${API_BASE_URL}/api/training/start`, {
  //     method: "POST",
  //     headers: { "Content-Type": "application/json" },
  //     body: JSON.stringify(config),
  //     signal: opts?.signal,
  //   },
  //   Number(process.env.NEXT_PUBLIC_START_TRAINING_TIMEOUT_MS ?? 60000)
  // );
  //   if (!response.ok) throw new Error(await response.text());
  //   return response.json()
  // },
    return fetchJSON<{success:boolean; session_id:string}>(
      `${API_BASE_URL}/api/training/start`,
      {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(config),
        timeoutMs,
        // credentials: "include",
      }
    );
  },

  // Stop training
  async stopTraining(sessionId: string, opts?: { signal?: AbortSignal }, timeoutMs=20_000): Promise<{ success: boolean }> {
  //   const response = await withTimeout(`${API_BASE_URL}/api/training/stop/${sessionId}`, {
  //     method: "POST",
  //     headers: { "Content-Type": "application/json" },
  //     body: JSON.stringify({ session_id: sessionId }),
  //     keepalive: true,
  //     signal: opts?.signal,
  //   },
  //   10000
  // );
  //   if(!response.ok) throw new Error(await response.text());
  //   return response.json()
  // },
    return fetchJSON<{ success:boolean; session_id:string}>(
      `${API_BASE_URL}/api/training/stop/${sessionId}`,
      {
        method:"POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
        keepalive: true,
        timeoutMs,
        signal: opts?.signal,
        // credentials: "include",
      }
    )
  },

  // Get training status
  async getTrainingStatus(sessionId: string, opts?: { signal?: AbortSignal },timeoutMs=10_000): Promise<TrainingStatus> {
    const response = await withTimeout(`${API_BASE_URL}/api/training/status/${sessionId}`,
    { signal: opts?.signal },
    10000
  );
    if (!response.ok) throw new Error(await response.text());
    return response.json()
  },
  //   return fetchJSON<{success:boolean;session_id:string}>(
  //     `${API_BASE_URL}/api/training/status/${sessionId}`,
  //     {timeoutMs,}

  //   )
  // },

  // Get environment state
  async getEnvironmentState(sessionId: string, opts?: { signal?: AbortSignal }): Promise<EnvironmentState> {
    const response = await withTimeout(`${API_BASE_URL}/api/environment/state/${sessionId}`,
    { signal: opts?.signal },
    10000
    );
    if(!response.ok) throw new Error(await response.text());
    return response.json()
  },

  // Get matplotlib plot
  async getMatplotlibPlot(plotType: string, sessionId: string, opts?: { signal?: AbortSignal }): Promise<string> {
    const response = await withTimeout(`${API_BASE_URL}/api/plots/${plotType}/${sessionId}`, { signal: opts?.signal },
      20000
    );
    const text=await response.text();
    //this is for debug purpose
    if(!response.ok){
      console.error("Backend returned error:", text);
      throw new Error(text);
    }

    // const data = await response.json()
    const data = JSON.parse(text)
    return data.image_base64 // Base64 encoded image
  },

  // //to get the slider info(specific or the episode slider)
  // async setRenderEpisode(sessionId: string, selected_episode: number): Promise<void> {
  //   await fetch(`${API_BASE_URL}/api/session/${sessionId}/set-episode`, {
  //     method: "POST",
  //     headers: {
  //       "Content-Type": "application/json"
  //     },
  //     body: JSON.stringify({ selected_episode })
  //   })
  // }
  // fetch('/api/layout_modification', {
  //   method: 'POST',
  //   headers: {
  //     'Content-Type': 'application/json'
  //   },
  //   body: JSON.stringify({
  //     sessionId: sessionId,
  //     layout: modCode
  //   })
  // }),

}

// WebSocket connection for real-time updates
export class TrainingWebSocket {
  private ws: WebSocket | null = null
  private sessionId: string

  constructor(sessionId: string) {
    this.sessionId = sessionId
  }

  connect(onMessage: (data: any) => void, onError?: (error: Event) => void) {
    const wsUrl = `${API_BASE_URL.replace("http", "ws")}/ws/training/${this.sessionId}`
    this.ws = new WebSocket(wsUrl)

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if(data.type==="ping"){
        this.ws?.send(JSON.stringify({ type: "pong", ts: Date.now() }));
        return;
      }

      onMessage(data)
    }

    this.ws.onerror = (error) => {
      console.error("WebSocket error:", error)
      onError?.(error)
    }

    this.ws.onclose = () => {
      console.log("WebSocket connection closed")
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }
}

export function connectToLivePlot(
  sessionId: string,
  onImageUpdate: (imageBase64: string) => void,
  onError?: (error: string) => void
): WebSocket {
  const ws = new WebSocket(`${NEXT_PUBLIC_WS_URL}/ws/plot/${sessionId}`);

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if(data.type==="ping"){
      ws.send(JSON.stringify({ type: "pong", ts: Date.now() }));
      return;
    }

    if (data.image_base64) {
      onImageUpdate(data.image_base64);
    } else if (data.error && onError) {
      onError(data.error);
    }
  };

  ws.onerror = (e) => {
    if (onError) onError("WebSocket connection error");
    console.error("WebSocket error", e);
  };

  ws.onclose = () => {
    console.log("WebSocket closed");
  };

  return ws;
}
