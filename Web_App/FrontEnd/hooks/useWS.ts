'use client'
import { useEffect, useMemo, useRef, useState } from "react";
import { wsManager } from "./wsManager";

type Status = "idle"|"connecting"|"open"|"closed"|"error";

type UseWSOpts = {
    path?: string;                           // e.g. `/ws/plot/${sessionId}`
    onMessage?: (evt: MessageEvent) => void; // optional message handler
    throttleMs?: number;                     // drop bursts (e.g. 33 for ~30fps)
    // binaryType?: "blob" | "arraybuffer";
    onOpen?: () => void;
};

export function useWS({ path, onMessage, throttleMs = 0, onOpen }: UseWSOpts) {
    const [status, setStatus] = useState<Status>("idle");
    const wsRef = useRef<WebSocket | null>(null);
    const lastRef = useRef(0);
    const onMessageRef = useRef(onMessage);
    onMessageRef.current = onMessage;

    const url = useMemo(() => {
        // if (!path) return undefined;
        if(!path){
            if(process.env.NODE_ENV !== "production") console.warn("useWS idle: no path");
            // console.warn("WS skipped: no path");
            return undefined;
        }
        const base = process.env.NEXT_PUBLIC_WS_URL ?? "wss://warehouse-rl-api.fly.dev";
        console.log("WS connecting", { fullUrl:`${base}${path}` });
        return `${base}${path}`;
    }, [path]);

    // this block is added[test]
    // const queue = useRef<string[]>([]);
    // const flush = useCallback(() => {
    //     const ws = wsRef.current;
    //     if(!ws || ws.readyState !== WebSocket.OPEN) return;
    //     while (queue.current.length){
    //         const msg = queue.current.shift()!;
    //         try { ws.send(msg) } catch(e) { console.error("WS send failed",e); break; }
    //     }
    // }, []);

    // altered send and put here[test]
    // const send = useCallback((data: string| ArrayBufferLike | Blob | ArrayBufferView | object) => {
    //     const raw = (typeof data === "object" && !(data instanceof Blob) && !(data instanceof ArrayBuffer) && !ArrayBuffer.isView(data))
    //     ? JSON.stringify(data)
    //     : (data as any);
    //     const ws = wsRef.current;
    //     if(ws && ws.readyState === WebSocket.OPEN){
    //         try { ws.send(raw); return true; } catch (e) { console.error("WS send failed", e); return false; }
    //     }
    //     queue.current.push(raw);
    //     return false;
    // },[]);

    useEffect(() => {
        if (!url) {
            setStatus("idle");
            return;
        }

        const ws = wsManager.get(url);
        // const ws = wsManager.open(url,{onMessage})
        // if (binaryType) ws.binaryType = binaryType;
        wsRef.current = ws;
        setStatus("connecting");
        // setStatus(ws.readyState === WebSocket.OPEN ? "open" : "connecting");

        const handleOpen = () => {
            setStatus("open");
            try { onOpen?.(); } catch (e) { console.error(e); }
            // try {ws.send(JSON.stringify({type:"ready"})); } catch {}
            // flush();
        };
        const handleClose = (evt: CloseEvent) => {
            console.warn("WS closed", {
            url,
            code: evt.code,
            reason: evt.reason,
            wasClean: evt.wasClean,
            });
            setStatus("closed");
            console.warn("WS closed (hook)",{url, code: evt.code, reason: evt.reason, wasClean: evt.wasClean})
        };
        const handleError = (e: Event) => {
            console.error("WS error", { url, error: e });
            setStatus("error")
        };

    const handleMessage = (evt: MessageEvent) => {
        console.log("WS message received in useWS hook", { url, data: evt.data, hasHandler: !!onMessageRef.current });
        if (!onMessageRef.current) return;
        if (throttleMs > 0) {
            const now = performance.now();
            if (now - lastRef.current < throttleMs) {
                console.log("WS message throttled", { url, throttleMs, timeSinceLast: now - lastRef.current });
                return;
            }
            lastRef.current = now;
        }
        console.log("WS message passed to handler", { url });
        onMessageRef.current(evt);
    };

    setStatus(ws.readyState === WebSocket.OPEN ? "open":"connecting");

    // COMMENTED OUT: These event listeners conflict with WebSocket manager
    // ws.addEventListener("open", handleOpen);
    // ws.addEventListener("close", handleClose);
    // ws.addEventListener("error", handleError);
    
    // Only keep message listener - other events are handled by WebSocket manager
    ws.addEventListener("message", handleMessage);
    console.log("WS message listener attached", { url, readyState: ws.readyState });

    // try {ws.send(JSON.stringify({type:"ready"})); } catch {}

        return () => {
        // COMMENTED OUT: These event listener removals are no longer needed
        // ws.removeEventListener("open", handleOpen);
        // ws.removeEventListener("close", handleClose);
        // ws.removeEventListener("error", handleError);
        
        // Only remove message listener - other events are handled by WebSocket manager
        ws.removeEventListener("message", handleMessage);
        
        // this condition is added so that we don't kill CONNECTING sockets
        if( ws.readyState !== WebSocket.CONNECTING){
            wsManager.release(url);
            wsRef.current = null;
        }
        };
    }, [url, throttleMs, onMessage, onOpen]);
    // }, [url, throttleMs, binaryType, onMessage, onOpen]);

  // convenience send helpers (auto-JSON for objects)[temporarily COMMENTED]
    const send = (data: string | ArrayBufferLike | Blob | ArrayBufferView | object) => {
        const ws = wsRef.current;
        if (!ws || ws.readyState !== WebSocket.OPEN) return false;
        if (typeof data === "object" && !(data instanceof Blob) && !(data instanceof ArrayBuffer) && !(ArrayBuffer.isView(data))) {
            ws.send(JSON.stringify(data));
        } else {
            ws.send(data as any);
        }
            return true;
    };

    return { ws: wsRef.current, status, send };
}
