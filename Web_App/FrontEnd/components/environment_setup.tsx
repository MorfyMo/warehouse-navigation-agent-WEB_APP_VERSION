'use client';

import { useWS } from "@/hooks/useWS";
import { wsPaths } from "@/lib/api";
import { OrbitControls } from '@react-three/drei';
import { Canvas } from '@react-three/fiber';
import { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
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
                    // credentials: "include",
                });
                const data = await response.json();
                if(data.is_ready){
                    setEnvInit(true);
                    console.log("[env_init] is_ready ->", data.is_ready);
                }
            } catch (e: any){
                if (e?.name !== 'AbortError')
                console.log('env_init fetch error:',e);
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
            // in this block we modify to avoid identical-frame updates
            const rows = data.layout.length;
            const cols = rows ? data.layout[0].length :0;
            let h=0;
            for(let r=0; r<rows;r++){
                const row = data.layout[r] as number[];
                for (let c = 0; c<cols; c++) h = (h*31+(row[c]|0))|0;
            }
            const nextHash = `${rows}x${cols}:${h}`;
            if(nextHash !== lastHashRef.current){
                lastHashRef.current = nextHash;
                // originally we just sent and set the Grid in this way(which might cause duplicates)
                setGrid(data.layout);
            }
        }
    }, []);

    const [layoutGate, setLayoutGate]= useState(false);
    useEffect(()=>{
        if(isTraining && envinit && dimension ==="3D"){
            const t = setTimeout(() => setLayoutGate(true), 400);
            return () => { clearTimeout(t); setLayoutGate(false); };
        }
    }, [envinit, isTraining, dimension]);

    const layoutPath =
        validId(sessionId) && layoutGate
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
        console.log("[layout] path =", layoutPath, "status =", layoutStatus);
    }, [layoutStatus, layoutPath, sendLayout]);

    return (
    <Canvas gl={glFactory} onCreated={onCreated} camera={{ position: [0, 30, 0], fov: 30 }}>
        <ambientLight intensity={2} />
        <pointLight position={[10, 10, 10]} />

        {grid.length>0&&(<WarehouseScene grid={grid} />)}
        <OrbitControls />
    </Canvas>
    );
}