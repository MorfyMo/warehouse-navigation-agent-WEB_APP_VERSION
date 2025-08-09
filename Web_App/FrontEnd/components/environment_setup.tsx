'use client';

import { OrbitControls } from '@react-three/drei';
import { Canvas } from '@react-three/fiber';
import { useEffect, useState } from 'react';
import WarehouseScene from './warehouse_scene';

const isClient = typeof window !== "undefined";

interface CanvasProps{
    sessionId: string
    isTraining: boolean
    dimension: string
}

export default function CanvasWrapper({sessionId,isTraining,dimension}:CanvasProps) {
    const [envinit,setEnvInit]=useState(false);
    const [grid, setGrid] = useState<number[][]>([]);

    useEffect(()=>{
        const checkEnvInit = async()=>{
            if(!sessionId||!isTraining||dimension!=="3D") return;
            const response = await fetch(`http://localhost:8000/api/env_init/${sessionId}`);
            const data = await response.json();
            if(data.is_ready){
                setEnvInit(true);
            }
        }
        checkEnvInit();
    },[sessionId,isTraining,dimension]);

    useEffect(() => {
        console.log("Inside the second useEffect")
        console.log("Check session/env/dim", { sessionId, isTraining, envinit, dimension });

        if(!sessionId||!isTraining||!envinit) return;
        if(dimension!=="3D") return;

        console.log("Attempting to connect to layout WebSocket...");
        const socket = new WebSocket(`ws://localhost:8000/ws/layout/${sessionId}`); // Or use NEXT_PUBLIC_API_URL with ws


        socket.onopen = () => {
            socket.send(JSON.stringify({ type: "ready" }));
            console.log("[3D WS connected]");
        };

        socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                console.log("[3D WS message]",data);
                // const parsed = data.layout.map((row: string) => row.split('').map(Number));
                const parsed = data.layout;
                console.log("Parsed layout:", parsed.length, parsed[0]?.length);
                setGrid(parsed);
            } catch (err) {
                console.error("Failed to parse layout:", err);
            }
        };

        socket.onerror = (err) => {
            console.error("WebSocket error:", err);
            setTimeout(()=>{
                const socket = new WebSocket(`ws://localhost:8000/ws/layout/${sessionId}`);
            },2000);
        };

        socket.onclose = (event) => {
            console.warn("WebSocket layout closed", event);
            if (!socket.readyState || socket.readyState === WebSocket.CLOSED) {
                console.warn("Closed before open — connection may not have succeeded");
            }
        };

        // return () => {
        //     try{
        //         if(socket.readyState===WebSocket.OPEN||socket.readyState === WebSocket.CONNECTING){
        //             socket.close();
        //         }
        //     }
        //     catch(err){
        //         console.warn("WebSocket cleanup error",err);
        //     }
        // };
        return ()=>socket.close();
    }, [sessionId,isTraining, envinit,dimension]); //runs the code only when any of the things inside this bracket changes

    return (
    <Canvas camera={{ position: [0, 30, 0], fov: 30 }}>
        <ambientLight intensity={1} />
        <pointLight position={[10, 10, 10]} />

        {grid.length>0&&(<WarehouseScene grid={grid} />)}
        <OrbitControls />
    </Canvas>
    );
}