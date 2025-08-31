'use client'
// wsManager.ts — guarantees one physical WebSocket per URL across the app
// type Sub = {
//     onOpen?: (e: Event) => void;
//     onMessage?: (e: MessageEvent) => void;
//     onError?: (e: Event) => void;
//     onClose?: (e: CloseEvent) => void;
//     binaryType?: "blob"|"arraybuffer";
// }

// type Entry = {
//     ws: WebSocket;
//     subs: Set<Sub>;
//     backoffMs: number;
//     timerId: number|null;
// };

const GRACE_MS =3000;
const MAX_BACKOFF_MS = 3000;

class WSManager {
    private sockets = new Map<string, WebSocket>();
    private refs = new Map<string, number>();
    private timers = new Map<string, number>();
    private backoff = new Map<string, number>(); // ms
    private pingTimers = new Map<string, number>();

    private scheduleReconnect(url: string) {
        // don't reconnect if nobody wants it anymore
        if ((this.refs.get(url) ?? 0) <= 0) return;
        if (this.sockets.get(url)) return;
        // avoid multiple timers
        if (this.timers.has(url)) return;

        const cur = this.backoff.get(url) ?? 500;        // start at 0.5s
        const next = Math.min(cur * 2, MAX_BACKOFF_MS);            // cap at 3s
        this.backoff.set(url, next);

        const id = window.setTimeout(() => {
        this.timers.delete(url);
        // only reconnect if still wanted and not already connected
        if ((this.refs.get(url) ?? 0) > 0 && !this.sockets.get(url)) {
            const ws = new WebSocket(url);
            ws.binaryType="arraybuffer";
            this.sockets.set(url, ws);

            // re-attach the same lifecycle listeners as in get(url) above:
            ws.addEventListener("open", () => {
                console.info("WS open (reconnect)", { url });

                // send the "ready" signal to backend
                try {ws.send(JSON.stringify({type:"ready"}));} catch {}

                this.backoff.set(url, 500); // reset backoff on success

                // // now we add the heartbeat into the function(since we removed the previous heartbeat in api_Server)
                // const pingId = setInterval(()=>{
                //     if(ws.readyState === WebSocket.OPEN){
                //         try {ws.send(JSON.stringify({type: "ping", ts:Date.now()}));}
                //         catch{}
                //     }
                // },15000);
                // this.pingTimers.set(url,pingId as unknown as number);

                // have the subscription
                try{
                    if (url.includes("/ws/vis2d/")){
                        ws.send(JSON.stringify({type:"subscribe",topic:"2d_subs"}));
                    }
                    else if (url.includes("/ws/plot/")){
                        ws.send(JSON.stringify({type:"subscribe",topic:"2d_metrics_subs"}));
                    }
                    else if (url.includes("/ws/layout/")){
                        ws.send(JSON.stringify({type:"subscribe",topic:"3d_subs"}));
                    }
                    else if(url.includes("/ws/plot3d/")){
                        ws.send(JSON.stringify({type:"subscribe",topic:"3d_metrics_subs"}));
                    }
                    else if(url.includes("/ws/training/")){
                        ws.send(JSON.stringify({type:"subscribe",topic:"training"}));
                    }
                } catch {}
            });

            ws.addEventListener("error", (e) => {
            console.error("WS error (reconnect)", { url, error: e });
            });

            ws.addEventListener("close", (evt) => {
                console.warn("WS closed (reconnect)", { url, code: evt.code, reason: evt.reason, wasClean: evt.wasClean });
                this.sockets.delete(url);

                // const pingId = this.pingTimers.get(url);
                // if(pingId){
                //     clearInterval(pingId);
                //     this.pingTimers.delete(url);
                // }
                // this.sockets.delete(url);

                const refs = this.refs.get(url)??0;
                const transient =
                    evt.code === 1001 ||
                    evt.code === 1006 ||
                    evt.code === 1011 ||
                    evt.code === 1012 ||
                    evt.code === 1013;

                // ensure that only reconnect with it should(not exiting)
                if(refs>0 && transient) this.scheduleReconnect(url); // try again if still referenced
            });

            ws.addEventListener("open",()=>{
                try {ws.send(JSON.stringify({type:"ready"})); } catch {}

                const pingId = setInterval(()=>{
                    if(ws.readyState === WebSocket.OPEN){
                        try {ws.send(JSON.stringify({type: "ping", ts:Date.now()}));}
                        catch{}
                    }
                },15000);
                this.pingTimers.set(url,pingId as unknown as number);
            });

            ws.addEventListener("close",()=>{
                const pingId = this.pingTimers.get(url);
                if(pingId){
                    clearInterval(pingId);
                    this.pingTimers.delete(url);
                }
            })
        }
        }, cur);
        this.timers.set(url, id);
    }

    get(url: string): WebSocket {
        //this.ref: components using a given WebSocket URL
        // const c = (this.refs.get(url)??0)+1
        // this.refs.set(url,c)

        // existing socket
        const s = this.sockets.get(url);
        if (s && s.readyState !== WebSocket.CLOSED) {
            this.refs.set(url, (this.refs.get(url) ?? 0) + 1);
            return s;
        }

        // cancel pending close
        const pending = this.timers.get(url);
        if(pending){
            clearTimeout(pending);
            this.timers.delete(url);
        }

        const ws = new WebSocket(url);
        ws.binaryType = "arraybuffer";
        this.sockets.set(url, ws);
        // this below line has already been covered by the first condition(initially)[thus COMMENTED for now]:
        this.refs.set(url, (this.refs.get(url) ?? 0) + 1);

        ws.addEventListener("open", () => {
            console.info("WS open", { url });

            // send "ready" signal to backend
            try {ws.send(JSON.stringify({type:"ready"})); } catch {}

            // this.backoff.set(url, 500);
            // now we add the heartbeat (instead of adding heartbeat at api_Server side)
            const pingId = setInterval(()=>{
                if(ws.readyState === WebSocket.OPEN){
                    try {ws.send(JSON.stringify({type:"ping",ts:Date.now()}))}
                    catch{}
                }
            },15000);
            this.pingTimers.set(url,pingId as unknown as number);
        });

        ws.addEventListener("error", (e) => {
            console.error("WS error (manager)", { url, error: e });
        });

        ws.addEventListener("close", (evt) => {
            console.warn("WS closed (manager)", {
                url,
                code: evt.code,
                reason: evt.reason,
                wasClean: evt.wasClean,
            });
            // need to add the heartbeat cancel
            const pingId = this.pingTimers.get(url);
            if(pingId){
                clearInterval(pingId);
                // this.pingTimers.delete(url);
            }
            this.pingTimers.delete(url);
            this.sockets.delete(url);

            // const shouldReconnect =
            //     (this.refs.get(url) ?? 0)>0 &&
            //     evt.wasClean === false &&
            //     evt.code !== 1000 && evt.code !== 1001;
            const refs = this.refs.get(url)??0;
            const transient =
                evt.code === 1001 ||
                evt.code === 1006 ||
                evt.code === 1011 ||
                evt.code === 1012 ||
                evt.code === 1013;

            if(refs>0 && transient) this.scheduleReconnect(url);
        });
            
        return ws;
    }

    release(url: string) {
        // const n = (this.refs.get(url) ?? 0) - 1;
        const n =Math.max(0,((this.refs.get(url) ?? 0) - 1));
        this.refs.set(url,n)
        if ( n > 0 ) return;

        if(this.timers.has(url)) return;
        const id = window.setTimeout(() => {
            this.timers.delete(url);
            // this.refs.delete(url);
            // const t = this.timers.get(url);
            // if(t){clearTimeout(t); this.timers.delete(url);}
            // this.backoff.delete(url);
            if((this.refs.get(url) ?? 0) == 0){
                const ws = this.sockets.get(url);
                if(ws && ws.readyState !== WebSocket.CLOSED){
                    try{ws.close(1000,"no-subscribers");}catch{}
                }
                this.sockets.delete(url);

                const pingId = this.pingTimers.get(url);
                if(pingId){
                    clearInterval(pingId);
                }
                this.pingTimers.delete(url);
            }
        },GRACE_MS);
            this.timers.set(url,id);
        }
            
}

export const wsManager = new WSManager();
