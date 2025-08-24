//this file is for rendering with the default matplotlib version from our rendering from RL
"use client"

// import { updateDqn, updatePpo } from "@/components/performance-chart";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useWS } from "@/hooks/useWS";
import { wsPaths } from "@/lib/api";
import { useCallback, useEffect, useRef, useState } from "react";

// const NEXT_PUBLIC_WS_URL="wss://warehouse-rl-api.fly.dev"
const NEXT_PUBLIC_WS_URL=process.env.NEXT_PUBLIC_WS_URL!

interface WarehouseRenderProps {
    sessionId: string
    isTraining: boolean
    mode: string
    envReady: boolean
    // URL?: string|null|undefined
}

function validId(id: unknown): id is string{
    return typeof id==="string" && id!=="undefined" && id!== "null" && id.length>0 && id!== undefined && id!== null
}

export function WarehouseRender({ sessionId, isTraining, mode, envReady}: WarehouseRenderProps) {
    const [imageData, setImageData] = useState<string | null>(null)

    const handlePlotMsg = useCallback((event: MessageEvent)=>{
        if(typeof event.data !== "string") return;
        let data: any;
        try { data=JSON.parse(event.data); } catch {return;}

        if (data.type === "ping"){
            plotSendRef.current?.({type:"pong", ts: Date.now()});
            return;
        }
        if (typeof data.image_base64 === "string"){
            setImageData(data.image_base64);
        }

    },[]);

    const plotPath =
        validId(sessionId) && isTraining && mode==="2D" && envReady
            ? wsPaths.layout2d(sessionId)
            : undefined;
    // const plotPath =
    //     mode === "2D"
    //         ?(typeof URL === "string" && URL.length >0
    //             ? URL
    //             : (validId(sessionId) && isTraining && envReady ? wsPaths.plot2d(sessionId) : undefined))
    //         : undefined;


    const {status: plotStatus, send: sendPlot } = useWS({
        path: plotPath,
        onMessage: handlePlotMsg,
        throttleMs: 250,
    });

    const plotSendRef = useRef<typeof sendPlot | null>(null);
    useEffect(()=>{
        plotSendRef.current = sendPlot;
        return () => { plotSendRef.current = null; };
    }, [sendPlot]);

    useEffect(()=>{
        if (!plotPath) setImageData(null);
    }, [plotPath]);

    const source = imageData?.startsWith("data:")
        ? imageData
        : `data:image/jpeg;base64,${imageData}`;

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
                // link.href = `data:image/jpeg;base64,${imageData}`
                // link.download = "warehouse_render.jpeg"
                const href = imageData?.startsWith("data:")
                    ? imageData
                    : `data:image/jpeg;base64,${imageData}`;
                link.href = href;
                link.download = "warehouse_render.jpeg"
                link.click()
            }}
            >
            Download Image
            </Button>
        )}
        </div>

        {/* const src =  */}

        {imageData ? (
        <img
            // src={`data:image/jpeg;base64,${imageData}`}
            src = {source}
            alt="Warehouse Rendering"
            className="w-full h-auto rounded border mt-1"
        />
        ) : (
        <div className="text-sm text-muted-foreground text-center">Waiting for visualization...</div>
        )}
    </Card>
    )
}

