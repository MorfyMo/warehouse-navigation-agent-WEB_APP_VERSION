'use client'

// this file is intended for the comment page
import { useEffect, useState } from "react";
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "https://api.rl-navigation.com"
// const NEXT_PUBLIC_WS_URL="wss://api.rl-navigation.com"
const NEXT_PUBLIC_WS_URL=process.env.NEXT_PUBLIC_WS_URL!

export const dynamic = 'force-dynamic'; // disables static optimization
// export const revalidate = 0;

interface CommentProp{
    message: string
    username: string
    time: string
}

export default function Comments(){
    const [comment,setComment] = useState<CommentProp[]>([]);
    const [username,setUsername] = useState("Mr/Ms_Anonymous");
    const [message,setMessage]=useState("");

    useEffect(()=>{
        fetch(`${API_BASE_URL}/api/msg/`)
            .then((res)=>res.json())
            .then((data)=>{
                console.log("Fetched comment:", data);
                if(Array.isArray(data)){
                    setComment(data);
                }
                else{
                    setComment([]);
                }
            });
    },[]);

    const submitComment = async() =>{
        await fetch(`${API_BASE_URL}/api/add_msg/`,{
            method: "POST",
            headers: {"Content-Type":"application/json"},
            body: JSON.stringify({
                // session_id: sessionId,
                message,
                username,
                time: new Date().toISOString(),
            })
        });

        setMessage("");

        const res = await fetch(`${API_BASE_URL}/api/msg/`, { cache: 'no-store' })
        const data = await res.json()
        setComment(data);

    };

    return(
        <div className="p-4 border-t mt-4">
        {/* <div className="p-4 mt-4"> */}
            {/* <h2 className="text-xl font-semibold mb-2">Suggestions</h2> */}

            <div className="max-h-64 overflow-y-auto">
                {
                    comment.map((c,i)=>(
                    <div key={i} className="border-t py-2">
                        <p className="text-sm">
                            <strong>{c.username}</strong> at {""}
                            {new Date(c.time).toLocaleString()}
                        </p>
                        <p>{c.message}</p>
                    </div>
                    ))}
            </div>

            <div className="flex w-full gap-x-4 mb-2">
                {/* deal with user name */}
                <input
                type="text"
                placeholder="Mr/Ms_Anonymous"
                value={username}
                onChange={(e)=>setUsername(e.target.value)}
                className="border rounded px-2 py-1 mr-2"
                />
                {/* the message */}
                <input
                type="text"
                placeholder=" slience is gold (?)"
                value={message}
                onChange={(e)=>setMessage(e.target.value)}
                className="border rounded flex-grow px-2 py-1 w-1/2"
                />
                {/* the sending button */}
                <button
                onClick={submitComment}
                className="bg-gray-700 text-white p-4 px-3 py-1 rounded">Send</button>
            </div>
        </div>
    );

}