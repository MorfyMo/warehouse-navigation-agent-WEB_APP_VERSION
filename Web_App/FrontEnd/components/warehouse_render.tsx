//this file is for rendering with the default matplotlib version from our rendering from RL
"use client"

// import { updateDqn, updatePpo } from "@/components/performance-chart";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useEffect, useState } from "react";

interface WarehouseRenderProps {
    sessionId: string
    isTraining: boolean
    mode: string
}


export function WarehouseRender({ sessionId, isTraining, mode}: WarehouseRenderProps) {
    const [imageData, setImageData] = useState<string | null>(null)

  // Poll image from backend every 2 seconds
    // useEffect(() => {
    // if (!isTraining || !sessionId) return

    // const fetchImage = async () => {
    //     try {
    //     // const data = await api.getMatplotlibPlot(sessionId, "warehouse_render")  // <- this should trigger return64web=True
    //     const data = await api.getMatplotlibPlot(sessionId, "warehouse_render")
    //     setImageData(data)
    //     } catch (err) {
    //     console.error("Failed to fetch warehouse render image:", err)
    //     }
    // }

    // const interval = setInterval(fetchImage, 2000)
    // return () => clearInterval(interval)
    // }, [sessionId, isTraining])
    useEffect(() => {
        if (!isTraining || !sessionId ||sessionId==="null") return;

        // const url=(mode==="2D")?`ws://localhost:8000/ws/plot/${sessionId}`:`ws://localhost:8000/ws/plot3d/${sessionId}`
        // const socket = new WebSocket(url)
        if (mode!=="2D") return;
        const socket = new WebSocket(`ws://localhost:8000/ws/plot/${sessionId}`)

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data)
            // updateDqn(data)
            // updatePpo(data)
            if (data.image_base64) {
            setImageData(data.image_base64)
        }
        }

        socket.onerror = (err) => {
            console.error("WebSocket error:", err)
            console.log("WebSocket readyState:", socket.readyState);
        }

        return () => socket.close()
    }, [sessionId, isTraining, mode])


    return (
    // <Card className="p-4 mt-4">
    //p-4 means the padding(the space between edge and the content) is 1rem(16px)
    //mt-16 means margin-top 4 rem(64 px) ; this "mt-16" controls the distance from the top to the plot
    <Card className="p-4 mt-16">
        <div className="flex justify-between items-center mb-3">
        <h4 className="font-semibold">Warehouse Visualization</h4>
        {imageData && (
        <Button
            variant="outline"
            size="sm"
            onClick={() => {
                const link = document.createElement("a")
                link.href = `data:image/png;base64,${imageData}`
                link.download = "warehouse_render.png"
                link.click()
            }}
            >
            Download Image
            </Button>
        )}
        </div>

        {imageData ? (
        <img
            src={`data:image/png;base64,${imageData}`}
            alt="Warehouse Rendering"
            // className="w-full h-auto rounded border"
            className="w-full h-auto rounded border mt-1"
        />
        ) : (
        <div className="text-sm text-muted-foreground text-center">Waiting for visualization...</div>
        )}
    </Card>
    )
}

