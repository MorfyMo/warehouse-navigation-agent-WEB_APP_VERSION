'use client';

import { useWS } from "@/hooks/useWS";
import { wsPaths } from "@/lib/api";
import { OrbitControls } from '@react-three/drei';
import { Canvas } from '@react-three/fiber';
import { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import WarehouseScene from './warehouse_scene';

const isClient = typeof window !== "undefined";
// const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080"
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL
// const NEXT_PUBLIC_WS_URL="wss://api.rl-navigation.com"
const NEXT_PUBLIC_WS_URL =process.env.NEXT_PUBLIC_WS_URL!

export const dynamic = 'force-dynamic'; // disables static optimization
// export const revalidate = 0;

interface CanvasProps{
    sessionId: string
    isTraining: boolean
    dimension: string
    envReady: boolean
}

function validId(id: unknown): id is string{
    return typeof id==="string" && id!=="undefined" && id!== "null" && id.length>0
}

export default function CanvasWrapper({sessionId,isTraining,dimension, envReady}:CanvasProps) {
    // const [envinit,setEnvInit]=useState(false);
    const [grid, setGrid] = useState<number[][]>([]);

    // useEffect(()=>{
    //     if(!validId(sessionId)||!isTraining||dimension!=="3D") return;

    //     let alive = true;
    //     const ctl = new AbortController();

    //     const checkEnvInit = async()=>{
    //         try{
    //             const url = `${API_BASE_URL}/api/env_init/${sessionId}`;
    //             const response = await fetch(url, {
    //                 cache: 'no-store',
    //                 signal: ctl.signal
    //             })
    //             if(!response.ok){
    //                 console.warn("[env_init] not ok", response.status,url);
    //                 return;
    //             }
    //             // the above block replaced the first block
    //             const data = await response.json();
    //             console.log("[env_init] poll",{url,is_ready: data?.is_ready})
    //             if(data?.is_ready){
    //                 setEnvInit(true);
    //                 console.log("[env_init] is_ready ->", data.is_ready);
    //             }
    //         } catch (e: any){
    //             if (e?.name !== 'AbortError')
    //             console.log('[env_init] fetch error:',e);
    //         }
    //     };

    //     checkEnvInit();
    //     const id = window.setInterval(checkEnvInit, 1500);
    //     // const id = window.setInterval(envReady, 1500);

    //     return () => {
    //         alive = false;
    //         clearInterval(id);
    //         ctl.abort();
    //         // reset when deps change
    //         // setEnvInit(false);
    //     };
    // },[sessionId,isTraining,dimension]);

    const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
    const ctxLostRef = useRef(false);

    const glFactory = useCallback((canvas: HTMLCanvasElement | OffscreenCanvas) => {
        if(!rendererRef.current){
            rendererRef.current = new THREE.WebGLRenderer({
                canvas: canvas as unknown as HTMLCanvasElement,
                antialias: true,
                alpha: false,
                powerPreference: 'high-performance',
                preserveDrawingBuffer: false,
            });
            if(rendererRef.current.domElement instanceof HTMLCanvasElement){
                const el = rendererRef.current.domElement as HTMLCanvasElement;
                rendererRef.current.setPixelRatio(window.devicePixelRatio);
                rendererRef.current.setSize(el.clientWidth, el.clientHeight, false);
            }
        }
        return rendererRef.current!;
    }, []);

    const onCreated = useCallback(({gl}:{gl:THREE.WebGLRenderer}) => {
        const canvas = gl.domElement as HTMLCanvasElement;

        const onLost = (e: Event) => {
            e.preventDefault();
            ctxLostRef.current = true;
        };

        const onRestored = () => {
            ctxLostRef.current = false;
            gl.resetState();
        }

        canvas.addEventListener('webglcontextlost',onLost,{passive: false});
        canvas.addEventListener('webglcontextrestored',onRestored,{passive: true});

        return () => {
            canvas.removeEventListener('webglcontextlost',onLost);
            canvas.removeEventListener('webglcontextrestored',onRestored);
        };
    },[]);

    useEffect(() => {
        return () => {
            const r = rendererRef.current;
            if(r) {
                const canvas = r.domElement as HTMLCanvasElement;
                canvas?.removeEventListener?.('webglcontextlost',()=>{});
                canvas?.removeEventListener?.('webglcontextrestored',()=>{});
                try { r.forceContextLoss?.(); } catch {}
                try { r.dispose(); } catch {}
                rendererRef.current = null;
                ctxLostRef.current = false;
            }
        };
    }, []);

    const lastHashRef = useRef<string>('');
    const handleLayoutMsg = useCallback((event: MessageEvent)=>{
        console.log("[3D] Received WebSocket message:", event.data);
        console.log("[3D] WebSocket connection state:", layoutSendRef.current ? "connected" : "disconnected");
        if(typeof event.data !== 'string') return;

        let data: any;
        try{
            data = JSON.parse(event.data);
            console.log("[3D received layout data at the frontend] data =", data);
            console.log("[3D received layout data at the frontend] data.type =", data.type);
            console.log("[3D received layout data at the frontend] data.layout =", data.layout);
        } catch {
            console.log("[3D] Failed to parse WebSocket message:", event.data);
            return;
        }

        if (data.type === "ping"){
            // Temporarily commented for useWS:
            layoutSendRef.current?.({type:"pong", ts: Date.now()});
            console.log("[3D] Layout WebSocket sent pong message");
            return;
        }

        // Temporarily COMMENT this below:
        // if(Array.isArray(data?.layout)){
        //     // in this block we modify to avoid identical-frame updates
        //     const rows = data.layout.length;
        //     const cols = rows ? data.layout[0].length :0;
        //     let h=0;
        //     for(let r=0; r<rows;r++){
        //         const row = data.layout[r] as number[];
        //         for (let c = 0; c<cols; c++) h = (h*31+(row[c]|0))|0;
        //     }
        //     const nextHash = `${rows}x${cols}:${h}`;
        //     if(nextHash !== lastHashRef.current){
        //         lastHashRef.current = nextHash;
        //         // originally we just sent and set the Grid in this way(which might cause duplicates)
        //         setGrid(data.layout);
        //     }
        // }
        
        // V1: set Grid function
        // if(data?.type === "render" &&
        //     Array.isArray(data.layout) &&
        //     data.layout.every((row:any) => Array.isArray(row))){
        //         setGrid(data.layout);
        //     }

        // V2: the suggested version
        if (data.type === "render" && Array.isArray(data.layout)){
            console.log("[3D] Setting grid with render data:", data.layout);
            console.log("[3D] Layout dimensions:", data.layout.length, "x", data.layout[0]?.length);
            setGrid(data.layout);
        } else if (Array.isArray(data.layout)){
            console.log("[3D] Setting grid with layout data:", data.layout);
            console.log("[3D] Layout dimensions:", data.layout.length, "x", data.layout[0]?.length);
            setGrid(data.layout);
        } else {
            console.log("[3D] No layout data to process, data.type =", data.type, "data.layout =", data.layout);
            console.log("[3D] WebSocket connection state:", layoutSendRef.current ? "connected" : "disconnected");
        }
        
    }, []);

    // const [layoutGate, setLayoutGate]= useState(false);
    // useEffect(()=>{
    //     if(isTraining && envinit && dimension ==="3D"){
    //         const t = setTimeout(() => setLayoutGate(true), 400);
    //         // return () => { clearTimeout(t); setLayoutGate(false); };
    //         return () => { clearTimeout(t); };
    //     }
    // }, [envinit, isTraining, dimension]);

    // V1:
    // const layoutPath =
    //     validId(sessionId) && layoutGate
    //         // ?`/ws/layout/${sessionId}`
    //         ? wsPaths.layout(sessionId)
    //         : undefined;

    // V2:
    // const layoutPath = useMemo(()=>{
    //     if(!validId(sessionId)) return undefined;
    //     if(!isTraining || !envinit || dimension !== "3D") return undefined;
    //     return wsPaths.layout(sessionId);
    // }, [sessionId, isTraining, envinit, dimension]);

    // V3:
    const layoutPath =
        validId(sessionId) && isTraining && dimension==="3D" && envReady
            ? wsPaths.layout(sessionId)
            : undefined;

    console.log("[layout guard check]", {
        sessionId,
        isTraining,
        envReady,
        dimension,
        layoutPath,
    });

    const sentReady = useRef(false);
    // this below useWS block is temporarily commented to test whether it is useWS problem:
    const { status: layoutStatus, send: sendLayout } = useWS({
        path: layoutPath,
        onMessage: handleLayoutMsg,
        throttleMs: 16, // ~60fps instead of 10fps
        onOpen: () => {
            console.log("[3D] Layout WebSocket opened, status:", layoutStatus);
            if (!sentReady.current) {
                sendLayout({ type: "ready" });
                sentReady.current = true;
            }
        }
    })

    const layoutSendRef = useRef<typeof sendLayout | null>(null);
    const sentReadyRef = useRef(false);
    useEffect(() => {
        layoutSendRef.current = sendLayout;
        return () => {
            layoutSendRef.current = null;
        };
    }, [sendLayout]);

    // Debug WebSocket status
    useEffect(() => {
        console.log("[3D] Layout WebSocket status changed:", layoutStatus);
    }, [layoutStatus]);

    // useEffect(()=>{
    //     if (!layoutPath){
    //         sentReadyRef.current = false;
    //         return;
    //     }
    //     if(layoutStatus === "open" && !sentReadyRef.current){
    //         sendLayout({ type:"ready" });
    //         sentReadyRef.current = true;
    //     }
    // }, [layoutStatus, layoutPath, sendLayout]);

    // "gl={glFactory} onCreated={onCreated}" taken out
    return (
    <Canvas camera={{ position: [0, 30, 0], fov: 30 }}>
        <ambientLight intensity={1.5} />
        <pointLight position={[10, 10, 10]} />

        {grid.length>0&&(<WarehouseScene grid={grid} />)}
        <OrbitControls />
    </Canvas>
    );
}