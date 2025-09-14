'use client'

import { useEffect } from "react";
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "https://api.rl-navigation.com"
// const NEXT_PUBLIC_WS_URL="wss://api.rl-navigation.com"
const NEXT_PUBLIC_WS_URL = process.env.NEXT_PUBLIC_WS_URL!

export let lastStatsDqn={average_reward: 0.0, time: "0.0s"}
export let lastStatsPpo = {normalized_reward: 0.0, time: "0.0s"}

export function updateDqn(data:any){
    if (data?.reward !== undefined && data?.reward !== null && !data?.stopped) {
        lastStatsDqn.average_reward = data.reward;
    }
    if (data?.time_text !== undefined && data?.time_text !== null && !data?.stopped) {
        lastStatsDqn.time = data.time_text;
    }
}

export function updatePpo(data:any){
    if (data?.reward !== undefined && data?.reward !== null && !data?.stopped) {
        lastStatsPpo.normalized_reward = data.reward;
    }
    if (data?.time_text !== undefined && data?.time_text !== null && !data?.stopped) {
        lastStatsPpo.time = data.time_text;
    }
}

interface UpdateProps{
    sessionId: string
}

export function UpdateDqn({sessionId}:UpdateProps){
    useEffect(() => {
        const socket = new WebSocket(`${NEXT_PUBLIC_WS_URL}/ws/plot/${sessionId}`);

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if(data.type==="ping"){
                socket.send(JSON.stringify({ type: "pong", ts: Date.now() }));
                return;
            }
            updateDqn(data); // updates lastValidDqnStats
        };

        return () => {
            socket.close();
        };
    }, [sessionId]);
}

export function UpdatePpo({sessionId}:UpdateProps){
    useEffect(() => {
        const socket = new WebSocket(`${NEXT_PUBLIC_WS_URL}/ws/plot/${sessionId}`);

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if(data.type==="ping"){
                socket.send(JSON.stringify({ type: "pong", ts: Date.now() }));
                return;
            }
            updatePpo(data); // updates lastValidDqnStats
        };

        return () => {
            socket.close();
        };
    }, [sessionId]);
}