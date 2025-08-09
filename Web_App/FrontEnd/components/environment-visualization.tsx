"use client"

import { useEffect, useState } from "react"
import { useTheme } from "next-themes"

interface EnvironmentVisualizationProps {
  algorithm: "dqn" | "ppo"
  isTraining: boolean
}

export function EnvironmentVisualization({ algorithm, isTraining }: EnvironmentVisualizationProps) {
  const [agentPosition, setAgentPosition] = useState({ x: 50, y: 50 })
  const [targetPosition] = useState({ x: 350, y: 100 })
  const [obstacles] = useState([
    { x: 150, y: 120, width: 60, height: 60 },
    { x: 250, y: 200, width: 80, height: 40 },
    { x: 100, y: 250, width: 40, height: 80 },
  ])
  const [episode, setEpisode] = useState(1)
  const [reward, setReward] = useState(0)
  const [mounted, setMounted] = useState(false)
  const { theme } = useTheme()

  // Avoid hydration mismatch
  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    if (!isTraining) return

    const interval = setInterval(() => {
      setAgentPosition((prev) => {
        const newX = Math.max(20, Math.min(380, prev.x + (Math.random() - 0.5) * 20))
        const newY = Math.max(20, Math.min(280, prev.y + (Math.random() - 0.5) * 20))

        // Calculate reward based on distance to target
        const distance = Math.sqrt(Math.pow(newX - targetPosition.x, 2) + Math.pow(newY - targetPosition.y, 2))
        const newReward = Math.max(0, 100 - distance * 0.5)
        setReward(Math.round(newReward))

        return { x: newX, y: newY }
      })

      // Occasionally reset episode
      if (Math.random() < 0.02) {
        setEpisode((prev) => prev + 1)
        setAgentPosition({ x: 50, y: 50 })
      }
    }, 200)

    return () => clearInterval(interval)
  }, [isTraining, targetPosition.x, targetPosition.y])

  // Determine if we're in light mode
  const isLightMode = mounted && theme === "light"

  return (
    <div className="space-y-4">
      <div className="flex justify-between text-sm">
        <span>Episode: {episode}</span>
        <span>Current Reward: {reward}</span>
        <span>Algorithm: {algorithm.toUpperCase()}</span>
      </div>

      <div
        className={
          isLightMode
            ? "relative bg-gray-100 rounded-lg overflow-hidden"
            : "relative bg-gray-900/50 rounded-lg overflow-hidden"
        }
        style={{ height: "300px" }}
      >
        {/* Grid background */}
        <svg className="absolute inset-0 w-full h-full">
          <defs>
            <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
              <path d="M 20 0 L 0 0 0 20" fill="none" stroke="currentColor" strokeOpacity="0.1" strokeWidth="1" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>

        {/* Obstacles */}
        {obstacles.map((obstacle, index) => (
          <div
            key={index}
            className={isLightMode ? "absolute bg-red-400 rounded" : "absolute bg-red-600/70 rounded"}
            style={{
              left: obstacle.x,
              top: obstacle.y,
              width: obstacle.width,
              height: obstacle.height,
            }}
          />
        ))}

        {/* Target */}
        <div
          className={
            isLightMode
              ? "absolute bg-green-400 rounded-full flex items-center justify-center text-white font-bold"
              : "absolute bg-green-600/80 rounded-full flex items-center justify-center text-white font-bold"
          }
          style={{
            left: targetPosition.x - 15,
            top: targetPosition.y - 15,
            width: 30,
            height: 30,
          }}
        >
          T
        </div>

        {/* Agent */}
        <div
          className={
            algorithm === "dqn"
              ? isLightMode
                ? "absolute bg-blue-500 rounded-full flex items-center justify-center text-white font-bold transition-all duration-200"
                : "absolute bg-blue-600/90 rounded-full flex items-center justify-center text-white font-bold transition-all duration-200"
              : isLightMode
                ? "absolute bg-purple-500 rounded-full flex items-center justify-center text-white font-bold transition-all duration-200"
                : "absolute bg-purple-600/90 rounded-full flex items-center justify-center text-white font-bold transition-all duration-200"
          }
          style={{
            left: agentPosition.x - 10,
            top: agentPosition.y - 10,
            width: 20,
            height: 20,
          }}
        >
          A
        </div>

        {/* Path visualization */}
        {isTraining && (
          <div
            className="absolute border-2 border-dashed border-gray-400 rounded-full opacity-30"
            style={{
              left: agentPosition.x - 30,
              top: agentPosition.y - 30,
              width: 60,
              height: 60,
            }}
          />
        )}
      </div>

      <div className={isLightMode ? "text-xs text-gray-600 space-y-1" : "text-xs text-muted-foreground space-y-1"}>
        <p>
          <span className="font-semibold">A:</span> Agent | <span className="font-semibold">T:</span> Target |{" "}
          <span className={isLightMode ? "font-semibold text-red-500" : "font-semibold text-red-400"}>Red:</span>{" "}
          Obstacles
        </p>
        <p>The agent learns to navigate from start position to target while avoiding obstacles.</p>
      </div>
    </div>
  )
}
