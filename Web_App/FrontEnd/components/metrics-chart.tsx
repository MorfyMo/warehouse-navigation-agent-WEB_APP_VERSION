"use client"

import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Download } from "lucide-react"
import { useTheme } from "next-themes"
import { useEffect, useState } from "react"

interface MetricsChartProps {
  algorithm: "dqn" | "ppo"
  isTraining: boolean
  progress: number
  sessionId?: string | null
  mode: string
  // rewardData: number[]
  // lossData: number[]
  // epsilonData: number[]
  // actorlossData: number[]
  // criticlossData: number[]
}

//the parameters after sessionId are the parameters for metric chart
export function MetricsChart({ algorithm, isTraining, progress, sessionId, mode}: MetricsChartProps) {
  const [rewardData, setRewardData] = useState<number[]>([])
  const [lossData, setLossData] = useState<number[]>([])
  const [epsilonData, setEpsilonData] = useState<number[]>([])
  const [actorlossData, setActorlossData] = useState<number[]>([])
  const [criticlossData, setCriticlossData] = useState<number[]>([])
  const [matplotlibPlot, setMatplotlibPlot] = useState<string | null>(null)
  const [mounted, setMounted] = useState(false)
  const { theme } = useTheme()
  const [isRendering, setIsRendering]=useState(false)
  const [start,setStart]=useState(false)

  // Avoid hydration mismatch
  useEffect(() => {
    setMounted(true)
  }, [])

  // // Fetch matplotlib plots from backend
  // useEffect(() => {
  //   if (!sessionId || !isTraining) return

  //   const fetchPlots = async () => {
  //     try {
  //       const plotImage = await api.getMatplotlibPlot("training_progress",sessionId)
  //       setMatplotlibPlot(plotImage)
  //     } catch (error) {
  //       console.log("Matplotlib plots not available, using simulated data")
  //     }
  //   }

  //   const interval = setInterval(fetchPlots, 2000)
  //   return () => clearInterval(interval)
  // }, [sessionId, isTraining])

  useEffect(() => {
    if (!sessionId || !isTraining) return;

    //this assumes that we just directly return if we don't have 2D
    const route = mode === "2D" ? `ws://localhost:8000/ws/plot/${sessionId}` : `ws://localhost:8000/ws/plot3d/${sessionId}`;

    const socket = new WebSocket(route);

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("[WS Plot Message]", data);
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.image_base64) {
        setMatplotlibPlot(data.image_base64);
      }

      // if (data.type === "metrics_update") {
      if(data.type==="render"){
        setIsRendering(true)
        if (typeof data.reward === "number") {
          setRewardData((prev) => [...prev.slice(-49), data.reward])
        }
        if (typeof data.loss === "number" && algorithm === "dqn") {
          setLossData((prev) => [...prev.slice(-49), data.loss])
        }
        if (typeof data.epsilon === "number" && algorithm === "dqn") {
          setEpsilonData((prev) => [...prev.slice(-49), data.epsilon])
        }
        if (typeof data.loss === "number" && algorithm === "ppo") {
          setLossData((prev) => [...prev.slice(-49), data.actor_loss])
        }
        if (typeof data.epsilon === "number" && algorithm === "ppo") {
          setEpsilonData((prev) => [...prev.slice(-49), data.critic_loss])
        }
      }
      else if(data.type==="ready"){
        setStart(true)
      }
      else{
        setIsRendering(false)
        setStart(false)
      }
    };

    return () => socket.close();
  }, [sessionId, isTraining]);

//the simulation version:
  // useEffect(() => {
  //   if (!isTraining) return

  //   const interval = setInterval(() => {
  //     setRewardData((prev) => {
  //       const newReward =
  //         algorithm === "dqn" ? Math.random() * 100 + progress * 2 : Math.random() * 120 + progress * 2.5
  //       return [...prev.slice(-49), newReward]
  //     })

  //     setLossData((prev) => {
  //       const newLoss = Math.max(0, 10 - progress * 0.1 + Math.random() * 2)
  //       return [...prev.slice(-49), newLoss]
  //     })

  //     if (algorithm === "dqn") {
  //       setEpsilonData((prev) => {
  //         const newEpsilon = Math.max(0.01, 1 - progress * 0.01)
  //         return [...prev.slice(-49), newEpsilon]
  //       })
  //     }
  //   }, 500)

  //   return () => clearInterval(interval)
  // }, [isTraining, algorithm, progress])

  const maxReward = Math.max(...rewardData, 100)
  const maxLoss = Math.max(...lossData, 10)
  const safeMaxLoss = Math.max(1e-8, maxLoss); // Avoid division by zero

  const minRewardy = Math.min(...rewardData, 100)
  const minLossy = Math.min(...lossData, 100)
  const minActory = Math.min(...actorlossData, 100)
  const minCriticy = Math.min(...criticlossData, 100)

  // Determine if we're in light mode
  const isLightMode = mounted && theme === "light"

  const downloadPlot = () => {
    if (matplotlibPlot) {
      const link = document.createElement("a")
      link.href = `data:image/png;base64,${matplotlibPlot}`
      link.download = `${algorithm}_training_plot.png`
      link.click()
    }
  }

  return (
    <div className="space-y-4">
      {/* Matplotlib Plot from Backend */}
      {matplotlibPlot && (
        <Card className="p-4">
          <div className="flex justify-between items-center mb-3">
            <h4 className="font-semibold">Training Progress (2D)</h4>
            <Button variant="outline" size="sm" onClick={downloadPlot}>
              <Download className="h-4 w-4 mr-2" />
              Download
            </Button>
          </div>
          <div className="flex justify-center">
            <img
              src={`data:image/png;base64,${matplotlibPlot}`}
              alt="Training Progress"
              className="max-w-full h-auto rounded border"
            />
          </div>
        </Card>
      )}

      {/* Real-time Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* Reward Chart */}
        <Card className="p-4">
          <h4 className="font-semibold mb-3">Episode Reward</h4>
          <div className={isLightMode ? "h-32 relative bg-gray-50 rounded" : "h-32 relative bg-gray-900/50 rounded"}>
            <svg className="w-full h-full">
              {rewardData.length > 1 && (
                <polyline
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth="2"
                  points={rewardData
                    .map(
                      (reward, index) =>{
                        // `${(index / (rewardData.length - 1)) * 100},${100 - (reward / maxReward) * 80}`,
                      const x=(index / (rewardData.length - 1)) * 100;
                      // const y = 100 - (reward / maxReward) * 80;
                      const y=100-((reward-minRewardy)/maxReward)*80;
                      return `${x},${y}`
                      }
                    )
                    .join(" ")}
                />
              )}
            </svg>
            {/* {isRendering===false && (
                <div className="absolute inset-0 flex items-center justify-center text-sm text-gray-500">
                  In Middle of Episode...
                </div>
            )} */}

            <div
              className={
                isLightMode
                  ? "absolute bottom-1 left-1 text-xs text-gray-500"
                  : "absolute bottom-1 left-1 text-xs text-muted-foreground"
              }
            >
              Current: {rewardData[rewardData.length - 1]?.toFixed(1) || "0.0"}
            </div>
          </div>
        </Card>


        {/* Separating DQN and PPO specific items */}
        {/* DQN */}
        {/* Loss Chart */}
        {algorithm === "dqn" && (
          <Card className="p-4">
            <h4 className="font-semibold mb-4">Episode Training Loss</h4>
            <div className="text-xs text-gray-500 -mt-4 leading-tight whitespace-nowrap mb-1">update per episode</div>
            <div className={isLightMode ? "h-32 relative bg-gray-50 rounded" : "h-32 relative bg-gray-900/50 rounded"}>
              <svg className="w-full h-full">
                {lossData.length > 1 && (
                  <polyline
                    fill="none"
                    stroke="#ef4444"
                    strokeWidth="2"
                    points={lossData
                      // .map((loss, index) =>`${(index / (lossData.length - 1)) * 100},${100 - (loss / maxLoss) * 80}`)
                      .map((loss,index) =>
                      {
                        const x = (index / (lossData.length - 1)) * 100;
                        const y = 100 - ((loss - minLossy )/ maxLoss) * 80;
                        return `${x},${y}`
                      })
                      .join(" ")}
                  />
                )}
              </svg>
              {lossData.length <= 1 && isRendering && (
                <div className="absolute inset-0 flex items-center justify-center text-sm text-gray-500">
                  In Middle of Episode...
                </div>
              )}

              <div
                className={
                  isLightMode
                    ? "absolute bottom-1 left-1 text-xs text-gray-500"
                    : "absolute bottom-1 left-1 text-xs text-muted-foreground"
                }
              >
                Current: {lossData[lossData.length - 1]?.toFixed(3) || "0.000"}
              </div>
            </div>
          </Card>
        )}


        {/* Algorithm-specific metrics */}
        {algorithm === "dqn" && (
          <Card className="p-4">
            <h4 className="font-semibold mb-3">Epsilon (Exploration)</h4>
            <div className={isLightMode ? "h-32 relative bg-gray-50 rounded" : "h-32 relative bg-gray-900/50 rounded"}>
              <svg className="w-full h-full">
                {epsilonData.length > 1 && (
                  <polyline
                    fill="none"
                    stroke="#10b981"
                    strokeWidth="2"
                    points={epsilonData
                      .map((epsilon, index) => `${(index / (epsilonData.length - 1)) * 100},${100 - epsilon * 80}`)
                      .join(" ")}
                  />
                )}
              </svg>

              {lossData.length <= 1 && isRendering && (
                <div className="absolute inset-0 flex items-center justify-center text-sm text-gray-500">
                  In Middle of Episode...
                </div>
              )}

              <div
                className={
                  isLightMode
                    ? "absolute bottom-1 left-1 text-xs text-gray-500"
                    : "absolute bottom-1 left-1 text-xs text-muted-foreground"
                }
              >
                Current: {epsilonData[epsilonData.length - 1]?.toFixed(3) || "1.000"}
              </div>
            </div>
          </Card>
        )}

        {/* PPO: originally policy entropy */}
        {/* {algorithm === "ppo" && (
          <Card className="p-4">
            <h4 className="font-semibold mb-3">Policy Entropy</h4>
            <div
              className={
                isLightMode
                  ? "h-32 relative bg-gray-50 rounded flex items-center justify-center"
                  : "h-32 relative bg-gray-900/50 rounded flex items-center justify-center"
              }
            >
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">{(Math.random() * 2 + 1).toFixed(2)}</div>
                <div className={isLightMode ? "text-xs text-gray-500" : "text-xs text-muted-foreground"}>
                  Current Entropy
                </div>
              </div>
            </div>
          </Card>
        )} */}
        {algorithm === "ppo" && (
          <Card className="p-4">
            <h4 className="font-semibold mb-3">Episode Actor Loss</h4>
            <div className="text-xs text-gray-500 -mt-4 leading-tight whitespace-nowrap mb-1">update per episode</div>
            <div className={isLightMode ? "h-32 relative bg-gray-50 rounded" : "h-32 relative bg-gray-900/50 rounded"}>
              <svg className="w-full h-full">
                {actorlossData.length > 1 && (
                  <polyline
                    fill="none"
                    stroke="#DFC57B"
                    strokeWidth="2"
                    points={actorlossData
                      // .map((loss, index) => `${(index / (actorlossData.length - 1)) * 100},${100 - (loss / maxLoss) * 80}`)
                      // .map((loss, index) => `${(index / (actorlossData.length - 1)) * 100},${100 - ((loss - minActory )/ maxLoss) * 80}`)
                      .map((actorloss,index)=>{
                        const x = (index / (actorlossData.length - 1)) * 100;
                        const y = 100 - ((actorloss - minActory )/ safeMaxLoss) * 80;
                        return `${x},${y}`
                      })
                      .join(" ")}
                  />
                )}
              </svg>

              {actorlossData.length <= 1 && isRendering && (
                <div className="absolute inset-0 flex items-center justify-center text-sm text-gray-500">
                  In Middle of Episode...
                </div>
              )}
              
              <div
                className={
                  isLightMode
                    ? "absolute bottom-1 left-1 text-xs text-gray-500"
                    : "absolute bottom-1 left-1 text-xs text-muted-foreground"
                }
              >
                Current: {actorlossData[actorlossData.length - 1]?.toFixed(3) || "0.000"}
              </div>
            </div>
          </Card>
        )}

        {algorithm === "ppo" && (
          <Card className="p-4">
            <h4 className="font-semibold mb-3">Episode Critic Loss</h4>
            <div className="text-xs text-gray-500 -mt-4 leading-tight whitespace-nowrap mb-1">update per episode</div>
            <div className={isLightMode ? "h-32 relative bg-gray-50 rounded" : "h-32 relative bg-gray-900/50 rounded"}>
              <svg className="w-full h-full">
                {criticlossData.length > 1 && (
                  <polyline
                    fill="none"
                    stroke="#C5B3EF"
                    strokeWidth="2"
                    points={criticlossData
                      // .map((loss, index) => `${(index / (criticlossData.length - 1)) * 100},${100 - (loss / maxLoss) * 80}`)
                      // .map((loss, index) => `${(index / (criticlossData.length - 1)) * 100},${100 - ((loss - minCriticy) / maxLoss) * 80}`)
                      .map((criticloss,index)=>{
                        const x = (index / (criticlossData.length - 1)) * 100;
                        const y = 100 - ((criticloss - minCriticy )/ safeMaxLoss) * 80;
                        return `${x},${y}`
                      })
                      .join(" ")}
                  />
                )}
              </svg>

              {criticlossData.length <= 1 && isRendering && (
                <div className="absolute inset-0 flex items-center justify-center text-sm text-gray-500">
                  In Middle of Episode...
                </div>
              )}

              <div
                className={
                  isLightMode
                    ? "absolute bottom-1 left-1 text-xs text-gray-500"
                    : "absolute bottom-1 left-1 text-xs text-muted-foreground"
                }
              >
                Current: {criticlossData[criticlossData.length - 1]?.toFixed(3) || "0.000"}
              </div>
            </div>
          </Card>
        )}
      </div>
    </div>
  )
}
