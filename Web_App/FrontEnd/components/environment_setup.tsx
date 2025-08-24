'use client';

import { useWS } from "@/hooks/useWS";
import { wsPaths } from "@/lib/api";
import { OrbitControls } from '@react-three/drei';
import { Canvas } from '@react-three/fiber';
import { useCallback, useEffect, useRef, useState } from 'react';
import WarehouseScene from './warehouse_scene';

const isClient = typeof window !== "undefined";
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080"
// const NEXT_PUBLIC_WS_URL="wss://warehouse-rl-api.fly.dev"
const NEXT_PUBLIC_WS_URL =process.env.NEXT_PUBLIC_WS_URL!

export const dynamic = 'force-dynamic'; // disables static optimization
// export const revalidate = 0;

interface CanvasProps{
    sessionId: string
    isTraining: boolean
    dimension: string
    // envReady: boolean
}

function validId(id: unknown): id is string{
    return typeof id==="string" && id!=="undefined" && id!== "null" && id.length>0
}

export default function CanvasWrapper({sessionId,isTraining,dimension}:CanvasProps) {
    const [envinit,setEnvInit]=useState(false);
    const [grid, setGrid] = useState<number[][]>([]);

    useEffect(()=>{
        if(!validId(sessionId)||!isTraining||dimension!=="3D") return;

        let alive = true;
        const ctl = new AbortController();

        const checkEnvInit = async()=>{
            try{
                const response = await fetch(`${API_BASE_URL}/api/env_init/${sessionId}`, {
                    cache: 'no-store',
                    signal: ctl.signal,
                });
                const data = await response.json();
                if(data.is_ready){
                    setEnvInit(true);
                }
            } catch {
                console.log("Error");
            }
        };

        checkEnvInit();
        const id = window.setInterval(checkEnvInit, 1500);
        // const id = window.setInterval(envReady, 1500);

        return () => {
            alive = false;
            clearInterval(id);
            ctl.abort();
            // reset when deps change
            setEnvInit(false);
        };
    },[sessionId,isTraining,dimension]);

    const handleLayoutMsg = useCallback((event: MessageEvent)=>{
        if(typeof event.data !== 'string') return;
        let data: any;
        try{
            data = JSON.parse(event.data);
        } catch {
            return;
        }

        if (data.type === "ping"){
            layoutSendRef.current?.({type:"pong", ts: Date.now()});
            return;
        }

        if(Array.isArray(data?.layout)){
            setGrid(data.layout);
        }

    }, []);

    const layoutPath =
        validId(sessionId) && isTraining && envinit && dimension ==="3D"
            // ?`/ws/layout/${sessionId}`
            ? wsPaths.layout(sessionId)
            : undefined;

    const { status: layoutStatus, send: sendLayout } = useWS({
        path: layoutPath,
        onMessage: handleLayoutMsg,
        throttleMs: 100,
    })

    const layoutSendRef = useRef<typeof sendLayout | null>(null);
    const sentReadyRef = useRef(false);
    useEffect(() => {
        layoutSendRef.current = sendLayout;
        return () => {
            layoutSendRef.current = null;
        };
    }, [sendLayout]);

    useEffect(()=>{
        if (!layoutPath){
            sentReadyRef.current = false;
            return;
        }
        if(layoutStatus === "open" && !sentReadyRef.current){
            sendLayout({type:"ready"});
            sentReadyRef.current = true;
        }
    }, [layoutStatus, layoutPath, sendLayout]);


    return (
    <Canvas camera={{ position: [0, 30, 0], fov: 30 }}>
        <ambientLight intensity={2} />
        <pointLight position={[10, 10, 10]} />

        {grid.length>0&&(<WarehouseScene grid={grid} />)}
        <OrbitControls />
    </Canvas>
    );
}