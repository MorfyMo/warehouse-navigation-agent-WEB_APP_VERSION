"use client"

import { useEffect, useState } from "react";
// import { lastStatsDqn, lastStatsPpo } from "./ui/stats-store";
import { lastStatsDqn, lastStatsPpo } from "../app/page";


// const lastStatsDqn = useRef({average_reward: 0.0, time: "0.0s"});
// const lastStatsPpo = useRef({normalized_reward: 0.0, time: "0.0s"});
// let lastStatsDqn = {average_reward: 0.0, time: "0.0s"};
// let lastStatsPpo = {normalized_reward: 0.0, time: "0.0s"};


interface PerformanceChartProps{
    // socket: WebSocket | null
    algorithm: "dqn"|"ppo"
    isTraining: boolean
    sessionId?: string|null
    mode: "2D"|"3D"
}

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

export function PerformanceChartDqn({algorithm, isTraining, sessionId, mode}: PerformanceChartProps){

    const [dqnStats, setDqnStats]=useState({
        average_reward: lastStatsDqn.average_reward,
        time: lastStatsDqn.time
    });


    useEffect(() => {
        if (!isTraining) {
            setDqnStats({
                average_reward: lastStatsDqn.average_reward,
                time: lastStatsDqn.time
            });
        }
    }, [isTraining]);

    useEffect(() => {
        setDqnStats({
            average_reward: lastStatsDqn.average_reward,
            time: lastStatsDqn.time
        });
    }, [algorithm, mode]);


    useEffect(() => {
        if(!sessionId||sessionId==="null"||mode!=="2D") return;

        if(algorithm!=="dqn") return;

        const socket = new WebSocket(`ws://localhost:8000/ws/plot/${sessionId}`)

        socket.onmessage = (event)=>{
            const data = JSON.parse(event.data);
            console.log("🌛🌛🌛 updated dqn data",data)

            if(!sessionId||algorithm!=="dqn"||!isTraining){
                setDqnStats(lastStatsDqn);
                return
            }
            
            if(data.error||data.stopped){
                setDqnStats(lastStatsDqn);
                return
            }

            if (data.error === "Session not found") {
                console.warn("❌ Session not found, skipping update");
                return;
            }

            if(data.current_algo==="dqn" && algorithm === "dqn"){

                const updateStats = {
                    average_reward: data.reward!==undefined && data.reward!==null && data.reward!==0 && !data.stopped ? data.reward : lastStatsDqn.average_reward,
                    // average_reward: data.reward!==undefined && data.reward!==null && !data.stopped ? data.reward : lastStatsDqn.average_reward,
                    time: data.time_text!==undefined && data.time_text!==null && !data.stopped ? data.time_text : lastStatsDqn.time
                }

                lastStatsDqn.average_reward = updateStats.average_reward
                lastStatsDqn.time = updateStats.time
                setDqnStats(updateStats);
            }
        }

        return() => {
            socket.close();
            socket.onmessage=null;
            console.log("DQN performance socket closed");
        }

    }, [algorithm, mode]);

    // return(
    //     <div className="space-y-4">
    //         <div className="flex justify-between">
    //             <span>Average Reward:</span>
    //             <span className="font-mono">{dqnStats.average_reward}</span>
    //         </div>
    //         {/* <div className="flex justify-between">
    //             <span>Episodes to Convergence:</span>
    //             <span className="font-mono">1,250</span>
    //         </div> */}
    //         <div className="flex justify-between">
    //             <span>Training Time:</span>
    //             <span className="font-mono">{dqnStats.time}</span>
    //         </div>
    //     </div>
    // )

    return null
}

export function PerformanceChartPpo({algorithm, isTraining, sessionId, mode}: PerformanceChartProps){

    const [ppoStats, setPpoStats]=useState({
        normalized_reward: lastStatsPpo.normalized_reward,
        time: lastStatsPpo.time
    });


    useEffect(() => {
        if (!isTraining) {
            setPpoStats({
                normalized_reward: lastStatsPpo.normalized_reward,
                time: lastStatsPpo.time
            });
        }
    }, [isTraining]);

    useEffect(() => {
        setPpoStats({
            normalized_reward: lastStatsPpo.normalized_reward,
            time: lastStatsPpo.time
        });
    }, [algorithm, mode]);


    useEffect(() => {
        if(!sessionId||sessionId==="null"||mode!=="2D") return;

        if(algorithm!=="ppo") return;

        const socket = new WebSocket(`ws://localhost:8000/ws/plot/${sessionId}`)

        socket.onmessage = (event)=>{
            const data = JSON.parse(event.data);
            console.log("Updated ppo data",data)

            if(!sessionId||algorithm!=="ppo"||!isTraining){
                setPpoStats(lastStatsPpo);
                return
            }

            if (data.error === "Session not found") {
                console.warn("❌ Session not found, skipping update");
                return;
            }

            if(data.current_algo==="ppo" && algorithm==="ppo"){

                const updateStats={
                    normalized_reward: data.reward!==null && data.reward!==undefined && data.reward!==0 && !data.stopped ? data.reward : lastStatsPpo.normalized_reward,
                    time: data.time_text!==null && data.time_text!==undefined && !data.stopped ? data.time_text : lastStatsPpo.time
                }

                lastStatsPpo.normalized_reward = updateStats.normalized_reward
                lastStatsPpo.time = updateStats.time
                setPpoStats(updateStats);
                // setPpoStats({normalized_reward:data.reward, time: data.time_text})
            }

        }

        return() => {
            socket.close();
            socket.onmessage=null;
            console.log("PPO socket closed");
        }


    }, [algorithm, mode]);

    // return(
    //     <div className="space-y-4">
    //             <div className="flex justify-between">
    //                 <span>Normalized Reward:</span>
    //                 <span className="font-mono">{ppoStats.normalized_reward}</span>
    //             </div>
    //             {/* <div className="flex justify-between">
    //                 <span>Episodes to Convergence:</span>
    //                 <span className="font-mono">980</span>
    //             </div> */}
    //             <div className="flex justify-between">
    //                 <span>Training Time:</span>
    //                 <span className="font-mono">{ppoStats.time}</span>
    //             </div>
    //         </div>
    // )
    return null

}

export function PerformanceChartDqn3d({algorithm, isTraining, sessionId, mode}: PerformanceChartProps){

    const [dqnStats, setDqnStats]=useState({
        average_reward: lastStatsDqn.average_reward,
        time: lastStatsDqn.time
    });

    useEffect(() => {
        if (!isTraining) {
            setDqnStats({
                average_reward: lastStatsDqn.average_reward,
                time: lastStatsDqn.time
            });
        }
    }, [isTraining]);

    useEffect(() => {
        setDqnStats({
            average_reward: lastStatsDqn.average_reward,
            time: lastStatsDqn.time
        });
    }, [algorithm, mode]);


    useEffect(() => {
        if(!sessionId||sessionId==="null"||mode!="3D") return;

        if(algorithm!=="dqn") return;

        const socket = new WebSocket(`ws://localhost:8000/ws/plot3d/${sessionId}`)

        socket.onmessage = (event)=>{
            const data = JSON.parse(event.data);
            console.log("🌛🌛🌛 updated dqn data",data)

            if(!sessionId||algorithm!=="dqn"||!isTraining){
                setDqnStats(lastStatsDqn);
                return
            }
            
            if(data.error||data.stopped){
                setDqnStats(lastStatsDqn);
                return
            }

            if (data.error === "Session not found") {
                console.warn("❌ Session not found, skipping update");
                return;
            }

            if(data.current_algo==="dqn" && algorithm === "dqn"){

                const updateStats = {
                    average_reward: data.reward!==undefined && data.reward!==null && data.reward!==0 && !data.stopped ? data.reward : lastStatsDqn.average_reward,
                    time: data.time_text!==undefined && data.time_text!==null && !data.stopped ? data.time_text : lastStatsDqn.time
                }

                lastStatsDqn.average_reward = updateStats.average_reward
                lastStatsDqn.time = updateStats.time
                setDqnStats(updateStats);
            }
        }

        return() => {
            socket.close();
            socket.onmessage=null;
            console.log("DQN performance socket closed");
        }

    }, [algorithm, mode]);

    // return(
    //     <div className="space-y-4">
    //         <div className="flex justify-between">
    //             <span>Average Reward:</span>
    //             <span className="font-mono">{dqnStats.average_reward}</span>
    //         </div>
    //         {/* <div className="flex justify-between">
    //             <span>Episodes to Convergence:</span>
    //             <span className="font-mono">1,250</span>
    //         </div> */}
    //         <div className="flex justify-between">
    //             <span>Training Time:</span>
    //             <span className="font-mono">{dqnStats.time}</span>
    //         </div>
    //     </div>
    // )
    return null
}

export function PerformanceChartPpo3d({algorithm, isTraining, sessionId, mode}: PerformanceChartProps){

    const [ppoStats, setPpoStats]=useState({
        normalized_reward: lastStatsPpo.normalized_reward,
        time: lastStatsPpo.time
    });

    useEffect(() => {
        if (!isTraining) {
            setPpoStats({
                normalized_reward: lastStatsPpo.normalized_reward,
                time: lastStatsPpo.time
            });
        }
    }, [isTraining]);

    useEffect(() => {
        setPpoStats({
            normalized_reward: lastStatsPpo.normalized_reward,
            time: lastStatsPpo.time
        });
    }, [algorithm, mode]);


    useEffect(() => {
        if(!sessionId||sessionId==="null"||mode!=="3D") return;

        if(algorithm!=="ppo") return;
        
        const socket = new WebSocket(`ws://localhost:8000/ws/plot3d/${sessionId}`)

        socket.onmessage = (event)=>{
            const data = JSON.parse(event.data);
            console.log("Updated ppo data",data)

            if(!sessionId||algorithm!=="ppo"||!isTraining){
                setPpoStats(lastStatsPpo);
                return
            }

            if (data.error === "Session not found") {
                console.warn("❌ Session not found, skipping update");
                return;
            }

            if(data.current_algo==="ppo" && algorithm==="ppo"){

                const updateStats={
                    normalized_reward: data.reward!==null && data.reward!==undefined && data.reward!==0 && !data.stopped ? data.reward : lastStatsPpo.normalized_reward,
                    time: data.time_text!==null && data.time_text!==undefined && !data.stopped ? data.time_text : lastStatsPpo.time
                }

                lastStatsPpo.normalized_reward = updateStats.normalized_reward
                lastStatsPpo.time = updateStats.time
                setPpoStats(updateStats);
                
            }
        }

        return() => {
            socket.close();
            socket.onmessage=null;
            console.log("PPO socket closed");
        }

    }, [algorithm, mode]);

    // return(
    //     <div className="space-y-4">
    //             <div className="flex justify-between">
    //                 <span>Normalized Reward:</span>
    //                 <span className="font-mono">{ppoStats.normalized_reward}</span>
    //             </div>
    //             {/* <div className="flex justify-between">
    //                 <span>Episodes to Convergence:</span>
    //                 <span className="font-mono">980</span>
    //             </div> */}
    //             <div className="flex justify-between">
    //                 <span>Training Time:</span>
    //                 <span className="font-mono">{ppoStats.time}</span>
    //             </div>
    //         </div>
    // )
    return null

}