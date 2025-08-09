"use client"

import { useEffect, useState } from "react"
import { useTheme } from "next-themes"
import { api } from "@/lib/api"

interface Agent {
  id: string
  x: number
  y: number
  type: "normal" | "helper" | "collector"
  carrying_package?: string
}

interface Package {
  id: string
  x: number
  y: number
  status: "shelf" | "carried" | "receiving" | "transfer"
}

interface WarehouseVisualizationProps {
  algorithm: "dqn" | "ppo"
  isTraining: boolean
  totalPackages: number
  sessionId: string | null
}

export function WarehouseVisualization({
  algorithm,
  isTraining,
  totalPackages,
  sessionId,
}: WarehouseVisualizationProps) {
  const [agents, setAgents] = useState<Agent[]>([
    { id: "normal1", x: 50, y: 50, type: "normal" },
    { id: "helper1", x: 80, y: 50, type: "helper" },
    { id: "collector1", x: 350, y: 250, type: "collector" },
  ])

  const [packages, setPackages] = useState<Package[]>([])
  const [episode, setEpisode] = useState(1)
  const [packagesDelivered, setPackagesDelivered] = useState(0)
  const [mounted, setMounted] = useState(false)
  const { theme } = useTheme()

  // Initialize packages on shelves
  useEffect(() => {
    const initialPackages: Package[] = []
    const shelfPositions = [
      { x: 100, y: 100 },
      { x: 130, y: 100 },
      { x: 160, y: 100 },
      { x: 100, y: 130 },
      { x: 130, y: 130 },
      { x: 160, y: 130 },
      { x: 250, y: 100 },
      { x: 280, y: 100 },
      { x: 310, y: 100 },
      { x: 250, y: 130 },
      { x: 280, y: 130 },
      { x: 310, y: 130 },
    ]

    for (let i = 0; i < Math.min(totalPackages, 50); i++) {
      const shelfPos = shelfPositions[i % shelfPositions.length]
      initialPackages.push({
        id: `package_${i}`,
        x: shelfPos.x + (i % 3) * 10,
        y: shelfPos.y + Math.floor(i / 3) * 5,
        status: "shelf",
      })
    }
    setPackages(initialPackages)
  }, [totalPackages])

  // Avoid hydration mismatch
  useEffect(() => {
    setMounted(true)
  }, [])

  // Fetch real-time data from backend if available
  useEffect(() => {
    if (!isTraining || !sessionId) return

    const interval = setInterval(async () => {
      try {
        const envState = await api.getEnvironmentState(sessionId)
        if (envState.agent_positions) {
          setAgents(
            envState.agent_positions.map((pos) => ({
              id: pos.id,
              x: pos.x,
              y: pos.y,
              type: pos.type as "normal" | "helper" | "collector",
            })),
          )
        }
        if (envState.packages) {
          setPackages(envState.packages)
        }
      } catch (error) {
        // Fallback to simulation
        simulateMovement()
      }
    }, 200)

    return () => clearInterval(interval)
  }, [isTraining, sessionId])

  // Simulation fallback
  const simulateMovement = () => {
    setAgents((prev) =>
      prev.map((agent) => {
        let newX = agent.x
        let newY = agent.y

        if (agent.type === "collector") {
          // Collector moves in receiving/transfer regions
          newX = Math.max(320, Math.min(400, agent.x + (Math.random() - 0.5) * 10))
          newY = Math.max(200, Math.min(300, agent.y + (Math.random() - 0.5) * 10))
        } else {
          // Normal and helper agents move in warehouse area
          newX = Math.max(20, Math.min(300, agent.x + (Math.random() - 0.5) * 15))
          newY = Math.max(20, Math.min(180, agent.y + (Math.random() - 0.5) * 15))
        }

        return { ...agent, x: newX, y: newY }
      }),
    )

    // Simulate package delivery
    if (Math.random() < 0.05) {
      setPackagesDelivered((prev) => Math.min(prev + 1, totalPackages))
      if (Math.random() < 0.02) {
        setEpisode((prev) => prev + 1)
        setPackagesDelivered(0)
      }
    }
  }

  // Determine if we're in light mode
  const isLightMode = mounted && theme === "light"

  // Define warehouse regions
  const warehouseRegions = {
    shelves: [
      { x: 90, y: 90, width: 90, height: 60 },
      { x: 240, y: 90, width: 90, height: 60 },
    ],
    receiving: { x: 320, y: 200, width: 80, height: 60 },
    transfer: { x: 320, y: 270, width: 80, height: 30 },
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between text-sm">
        <span>Episode: {episode}</span>
        <span>
          Packages Delivered: {packagesDelivered}/{totalPackages}
        </span>
        <span>Algorithm: {algorithm.toUpperCase()}</span>
      </div>

      <div
        className={
          isLightMode
            ? "relative bg-gray-100 rounded-lg overflow-hidden"
            : "relative bg-gray-900/50 rounded-lg overflow-hidden"
        }
        style={{ height: "320px" }}
      >
        {/* Grid background */}
        <svg className="absolute inset-0 w-full h-full">
          <defs>
            <pattern id="warehouse-grid" width="20" height="20" patternUnits="userSpaceOnUse">
              <path d="M 20 0 L 0 0 0 20" fill="none" stroke="currentColor" strokeOpacity="0.1" strokeWidth="1" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#warehouse-grid)" />
        </svg>

        {/* Warehouse Regions */}
        {/* Shelves */}
        {warehouseRegions.shelves.map((shelf, index) => (
          <div
            key={`shelf-${index}`}
            className="absolute bg-amber-200/60 dark:bg-amber-800/40 border-2 border-amber-400 dark:border-amber-600 rounded"
            style={{
              left: shelf.x,
              top: shelf.y,
              width: shelf.width,
              height: shelf.height,
            }}
          />
        ))}

        {/* Receiving Region (Yellow) */}
        <div
          className="absolute bg-yellow-200/80 dark:bg-yellow-800/60 border-2 border-yellow-500 dark:border-yellow-600 rounded"
          style={{
            left: warehouseRegions.receiving.x,
            top: warehouseRegions.receiving.y,
            width: warehouseRegions.receiving.width,
            height: warehouseRegions.receiving.height,
          }}
        />

        {/* Transfer Region (Gray) */}
        <div
          className="absolute bg-gray-300/80 dark:bg-gray-700/60 border-2 border-gray-500 dark:border-gray-600 rounded"
          style={{
            left: warehouseRegions.transfer.x,
            top: warehouseRegions.transfer.y,
            width: warehouseRegions.transfer.width,
            height: warehouseRegions.transfer.height,
          }}
        />

        {/* Packages */}
        {packages.slice(0, 20).map((pkg) => (
          <div
            key={pkg.id}
            className={`absolute w-2 h-2 rounded-sm ${
              pkg.status === "shelf"
                ? "bg-blue-500"
                : pkg.status === "receiving"
                  ? "bg-green-500"
                  : pkg.status === "transfer"
                    ? "bg-purple-500"
                    : "bg-orange-500"
            }`}
            style={{
              left: pkg.x,
              top: pkg.y,
            }}
          />
        ))}

        {/* Agents */}
        {agents.map((agent) => (
          <div
            key={agent.id}
            className={`absolute rounded-full flex items-center justify-center text-white font-bold text-xs transition-all duration-200 ${
              agent.type === "normal" ? "bg-blue-600" : agent.type === "helper" ? "bg-green-600" : "bg-purple-600"
            }`}
            style={{
              left: agent.x - 8,
              top: agent.y - 8,
              width: 16,
              height: 16,
            }}
          >
            {agent.type === "normal" ? "N" : agent.type === "helper" ? "H" : "C"}
          </div>
        ))}

        {/* Labels */}
        <div className="absolute top-2 left-2 text-xs font-semibold">
          <div className="bg-white/80 dark:bg-black/80 px-2 py-1 rounded">Shelves</div>
        </div>
        <div className="absolute top-2 right-2 text-xs font-semibold">
          <div className="bg-white/80 dark:bg-black/80 px-2 py-1 rounded">Receiving</div>
        </div>
        <div className="absolute bottom-2 right-2 text-xs font-semibold">
          <div className="bg-white/80 dark:bg-black/80 px-2 py-1 rounded">Transfer</div>
        </div>
      </div>

      <div className={isLightMode ? "text-xs text-gray-600 space-y-1" : "text-xs text-muted-foreground space-y-1"}>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 bg-blue-600 rounded-full"></div>
            <span>Normal Agent</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 bg-green-600 rounded-full"></div>
            <span>Helper Agent</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 bg-purple-600 rounded-full"></div>
            <span>Collector Agent</span>
          </div>
        </div>
        <p>Robots navigate the warehouse to move packages from shelves to the receiving region efficiently.</p>
      </div>
    </div>
  )
}
