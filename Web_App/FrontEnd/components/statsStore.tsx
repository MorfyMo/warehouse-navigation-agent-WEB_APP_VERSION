import { useEffect } from "react";

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
        const socket = new WebSocket(`ws://localhost:8000/ws/plot/${sessionId}`);

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateDqn(data); // updates lastValidDqnStats
        };

        return () => {
            socket.close();
        };
    }, [sessionId]);
}

export function UpdatePpo({sessionId}:UpdateProps){
    useEffect(() => {
        const socket = new WebSocket(`ws://localhost:8000/ws/plot/${sessionId}`);

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updatePpo(data); // updates lastValidDqnStats
        };

        return () => {
            socket.close();
        };
    }, [sessionId]);
}