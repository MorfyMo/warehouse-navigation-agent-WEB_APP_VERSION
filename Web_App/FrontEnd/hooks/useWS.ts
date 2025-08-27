'use client'
import { useEffect, useMemo, useRef, useState } from "react";
import { wsManager } from "./wsManager";

type Status = "idle"|"connecting"|"open"|"closed"|"error";

type UseWSOpts = {
    path?: string;                           // e.g. `/ws/plot/${sessionId}`
    onMessage?: (evt: MessageEvent) => void; // optional message handler
    throttleMs?: number;                     // drop bursts (e.g. 33 for ~30fps)
    binaryType?: "blob" | "arraybuffer";
};

export function useWS({ path, onMessage, throttleMs = 0, binaryType }: UseWSOpts) {
    const [status, setStatus] = useState<Status>("idle");
    const wsRef = useRef<WebSocket | null>(null);
    const lastRef = useRef(0);
    const onMessageRef = useRef(onMessage);
    onMessageRef.current = onMessage;

    const url = useMemo(() => {
        // if (!path) return undefined;
        if(!path){
            if(process.env.NODE_ENV !== "production") console.warn("useWS idle: no path");
            return undefined;
        }
        const base = process.env.NEXT_PUBLIC_WS_URL ?? "wss://warehouse-rl-api.fly.dev";
        return `${base}${path}`;
    }, [path]);

    useEffect(() => {
        if (!url) {
            setStatus("idle");
            return;
        }
        setStatus("connecting");

        const ws = wsManager.get(url);
        if (binaryType) ws.binaryType = binaryType;
        wsRef.current = ws;

        const handleOpen = () => {
            setStatus("open");
            try {ws.send(JSON.stringify({type:"ready"})); } catch {}
        };
        const handleClose = (evt: CloseEvent) => {
            console.warn("WS closed", {
            url,
            code: evt.code,
            reason: evt.reason,
            wasClean: evt.wasClean,
            });
            setStatus("closed");
        };
        const handleError = (e: Event) => {
            console.error("WS error", { url, error: e });
            setStatus("error")
        };

    const handleMessage = (evt: MessageEvent) => {
        if (!onMessageRef.current) return;
        if (throttleMs > 0) {
            const now = performance.now();
            if (now - lastRef.current < throttleMs) return;
            lastRef.current = now;
        }
        onMessageRef.current(evt);
    };

    setStatus(ws.readyState === WebSocket.OPEN ? "open":"connecting");

    ws.addEventListener("open", handleOpen);
    ws.addEventListener("close", handleClose);
    ws.addEventListener("error", handleError);
    ws.addEventListener("message", handleMessage);

    try {ws.send(JSON.stringify({type:"ready"})); } catch {}

        return () => {
        ws.removeEventListener("open", handleOpen);
        ws.removeEventListener("close", handleClose);
        ws.removeEventListener("error", handleError);
        ws.removeEventListener("message", handleMessage);
        wsManager.release(url);
        wsRef.current = null;
        };
    }, [url, throttleMs, binaryType, onMessage]);

  // convenience send helpers (auto-JSON for objects)
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
