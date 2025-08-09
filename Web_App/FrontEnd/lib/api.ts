// API configuration and utilities for connecting to Python backend
// const API_BASE_URL_backend = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

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

// API functions
export const api = {
  // Start training
  async startTraining(config: TrainingConfig): Promise<{ success: boolean; session_id: string }> {
    const response = await fetch(`${API_BASE_URL}/api/training/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    })
    return response.json()
  },

  // Stop training
  async stopTraining(sessionId: string): Promise<{ success: boolean }> {
    const response = await fetch(`${API_BASE_URL}/api/training/stop/${sessionId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    })
    return response.json()
  },

  // Get training status
  async getTrainingStatus(sessionId: string): Promise<TrainingStatus> {
    const response = await fetch(`${API_BASE_URL}/api/training/status/${sessionId}`)
    return response.json()
  },

  // Get environment state
  async getEnvironmentState(sessionId: string): Promise<EnvironmentState> {
    const response = await fetch(`${API_BASE_URL}/api/environment/state/${sessionId}`)
    return response.json()
  },

  // Get matplotlib plot
  async getMatplotlibPlot(plotType: string, sessionId: string): Promise<string> {
    const response = await fetch(`${API_BASE_URL}/api/plots/${plotType}/${sessionId}`);
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
  const ws = new WebSocket(`ws://localhost:8000/ws/plot/${sessionId}`);

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
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
