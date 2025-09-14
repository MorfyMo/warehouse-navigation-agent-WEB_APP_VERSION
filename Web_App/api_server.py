# Web_App/api_server.py
import contextlib
import sys
import os
import hashlib
import queue
import asyncio
import base64
import io
from typing import Dict, Any, Optional
import threading
import time
import ast
from fastapi.websockets import WebSocketState
from pydantic import BaseModel
from datetime import datetime


# Add your project root to Python path so we can import your modules
project_root=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Now we can import your existing modules
# TODO: import statements
#import the previous main module
# from main_module import *

# # from DQN_test.your_dqn_file import DQN related Class
# from DQN_test.DQN_module import Agent as DQNAgent, DQN_args_parser, writer as dqn_writer
# dqn_args,dqn_parser=DQN_args_parser()

# # from PPO_test.your_ppo_file import PPO related Class
# from PPO_test.PPO_module import Agent as PPOAgent, PPO_args_parser, writer as ppo_writer
# ppo_args,ppo_parser=PPO_args_parser()

# from env.your_env_file import Environment related Class
# from Env.env_module import Environment
import matplotlib
matplotlib.use('Agg') #non_interactive backend for web
import matplotlib.pyplot as plt

import logging, traceback
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, APIRouter, Request, logger
from fastapi import Body, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from starlette.websockets import WebSocketDisconnect, WebSocketState
from starlette.routing import Route, WebSocketRoute, Mount
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Scope, Receive, Send
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import uvicorn
import json
import copy

app = FastAPI(title="Warehouse Navigation API")

# import os
# from fastapi import Request, Response
# from starlette.middleware.base import BaseHTTPMiddleware

# def _instance_id() -> str:
#     return os.environ.get("FLY_MACHINE_ID") or os.environ.get("FLY_ALLOC_ID", "local-dev")

# STICKY_COOKIE = "fly_instance"

# class FlyStickyMiddleware(BaseHTTPMiddleware):
#     async def dispatch(self, request: Request, call_next):
#         want = request.cookies.get(STICKY_COOKIE)
#         have = _instance_id()
#         q = request.query_params.get("instance")
#         if q:
#             want = q
#         if want and want != have:
#             # Tell Fly to replay this request to the desired instance
#             return Response(status_code=409, headers={"fly-replay": f"instance={want}"})
#         resp: Response = await call_next(request)
#         # Set/refresh the cookie so future requests stick
#         resp.set_cookie(STICKY_COOKIE, have, secure=True, samesite="none", httponly=False, path="/")
#         return resp

frame_queues: dict[str, asyncio.Queue] = {}
# MIN_PERIOD = 0.1 #10 Hz cap

origins = [
    "https://rl-navigation.com", #the frontend link
    "https://api.rl-navigation.com",#the backend link
    "http://localhost:3000", #the local host version
]

class LogWSBaseMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app
    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "websocket":
            print("[ASGI] WS incoming:", scope.get("path"))
        await self.app(scope, receive, send)

# Enable CORS for the frontend
#this creates the FastAPI application(application initialization)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
# app.add_middleware(FlyStickyMiddleware)
app.add_middleware(LogWSBaseMiddleware)

api = APIRouter(prefix="/api")

@api.get("/health")
def health():
    """Simple health check - should always respond quickly"""
    try:
        # Quick check without heavy operations
        return {"ok": True, "timestamp": time.time()}
    except Exception as e:
        # Even if there's an error, try to return something
        return {"ok": False, "error": str(e), "timestamp": time.time()}

@api.get("/health/detailed")
def health_detailed():
    import psutil
    import os
    
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()
        
        # Get session details
        session_details = {}
        total_websocket_connections = 0
        for session_id, session in active_sessions.items():
            # Count WebSocket connections
            subs = session.get("subs", {})
            ws_connections = sum(subs.values())
            total_websocket_connections += ws_connections
            
            session_details[session_id] = {
                "status": session.get("status", "unknown"),
                "algorithm": session.get("config", {}).algorithm if session.get("config") else "unknown",
                "has_thread": session.get("training_thread") is not None,
                "thread_alive": session.get("training_thread").is_alive() if session.get("training_thread") else False,
                "websocket_connections": ws_connections,
                "subscribers": subs
            }
        
        return {
            "ok": True,
            "memory_mb": round(memory_info.rss / 1024 / 1024, 2),
            "memory_percent": round(psutil.virtual_memory().percent, 2),
            "cpu_percent": round(cpu_percent, 2),
            "active_sessions": len(active_sessions),
            "total_websocket_connections": total_websocket_connections,
            "session_details": session_details,
            "timestamp": time.time()
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@api.post("/health/cleanup")
def health_cleanup():
    """Force garbage collection and memory cleanup"""
    try:
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024
        
        # Force garbage collection
        collected = gc.collect()
        
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_freed = memory_before - memory_after
        
        return {
            "ok": True,
            "memory_before_mb": round(memory_before, 2),
            "memory_after_mb": round(memory_after, 2),
            "memory_freed_mb": round(memory_freed, 2),
            "objects_collected": collected,
            "timestamp": time.time()
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

# app.include_router(api)
# Store active training sessions
#This is a "threading.Thread" object, which does have ".is_alive()" function
"""threading.Thread:
a class from Python's built-in threading module
1) allow running code in parallel(in separate thread of execution)
2) this means it can run alongside other code
3) useful for doing other tasks(background) without blocking the main program
"""
init_layout=["1111111111111111111111111111111111111111111111111111111111111111111111",
"1000002000000000000000000000008000000000000000000000000000000000000001",
"1000000000000000000000000000000000000000000000000000000000000000000001",
"1000311300033333333333333333333333333333333000051111111111111111150001",
"1000311300011111111111111111111111111111111000011000000000000000110001",
"1000311300011111111111111111111111111111111000011000000000000000110001",
"1000311300033333333333333333333333333333333000011000000000000011110001",
"1000311300000008000000000020000000000000000000011000000000000111110001",
"1000311300000000000000000000000000000000000000011000000000000111110001",
"1000311300033333333333333333333333333333333000011000000000000111110001",
"1000311300011111111111111111111111111111111000011000000000000011110001",
"1000311300011111111111111111111111111111111000011000000000000000110001",
"1000311300033333333333333333333333333333333000051100001111111111150001",
"1000000000000000000000000000020000000000000000000000000000000000000001",
"1000000000000000000000000000000000000000000000000000000000000000000001",
"1000444444444444444444444444444400000000000000000000000000000000000001",
"1000444444444444444444444444444444444444444444444444444444444440000001",
"1000444444444444444444444444444444444444444444444444444444444440000001",
"1111666666666666666666966666666666666666666666666666666666666661111111"]

active_sessions: Dict[str, Dict[str, Any]] = {}
#this is used to store the modified layout if any
layoutModified= {
    "status":"not_modified",
    "layout":None
    }
layoutRaw={
    "layout":init_layout
}

#this is intend for storing all the comments(currently does not contain database management)
# messages={}
messages=[]

class StopFlag:
    def __init__(self):
        self._stop=False
    
    def __call__(self):
        return self._stop
    
    def stop(self):
        self._stop=True

# Request/Response models
class TrainingConfig(BaseModel):
    algorithm: str  # "dqn" or "ppo"
    episodes: int
    learning_rate: float
    environment: str  # "navigation" or "warehouse"
    warehouse_config: Optional[Dict[str, Any]] = None

class TrainingResponse(BaseModel):
    success: bool
    session_id: str
    message: str
    # instance: str
    
class LayoutRequest(BaseModel):
    # session_id: str
    layout: str

class Comment(BaseModel):
    # session_id: str
    message: str
    username: str
    time: datetime = datetime.utcnow()

print(">>> STARTUP: api_server loaded")

# Basic endpoints
@app.get("/")
async def root():
    return {"message": "RL Project API is running!", "status": "connected"}

# @app.get("/api/test")
@api.get("/test")
async def test_connection():
    return {"status": "connected", "message": "Frontend can reach backend"}

@app.get("/api/__corscheck")
async def cors_check(request: Request):
    return {
        "received_origin": request.headers.get("origin"),
        "allowed": origins,
    }

# this is intended for keeping the program alive
async def heartbeat(ws: WebSocket, interval: float = 15.0):
    """App-level heartbeat. Sends a small JSON ping periodically."""
    try:
        # while True:
        #     await ws.send_json({"type": "ping", "ts": time.time()})
        #     await asyncio.sleep(interval)
        return ws.application_state is WebSocketState.CONNECTED
    except WebSocketDisconnect:
        # Client dropped
        pass
    except Exception:
        # NOTE: since safe_send might cause races, we simply remove the send thing(shown below)
        # await safe_send(ws,"ERROR: Env not initialized.")
        # await ws.close()
        # return
        pass

# this function is a helper function that help to check whether the app is connected as we expected
def ws_connected(ws: WebSocket) -> bool:
    return (ws.application_state == WebSocketState.CONNECTED and ws.client_state == WebSocketState.CONNECTED)

def get_queue(session_id: str) -> asyncio.Queue:
    q = frame_queues.get(session_id)
    if q is None:
        q = asyncio.Queue(maxsize=1)
        frame_queues[session_id] = q
    return q

async def push_latest(session_id: str, payload: dict | bytes):
    q = get_queue(session_id)
    if q.full():
        try: q.get_nowait()
        except: pass
    await q.put(payload)

# this function is design for asyncio queue to add new element to queue

# def put_drop(q: "queue.Queue", item) -> bool:
#     try:
#         q.put_nowait(item)
#         # print(f"[put_drop] Successfully added item to queue: {type(item)}")
#         # log.info(f"[put_drop] Successfully added item to queue: {type(item)}")
#         return True
#     except queue.Full:
#         # print(f"[put_drop] Queue is full, removing oldest item")
#         # log.info(f"[put_drop] Queue is full, removing oldest item")
#         try:
#             q.get_nowait()     # drop oldest
#             # print(f"[put_drop] Drop the oldest item")
#             # log.info(f"[put_drop] Drop the oldest item")
#         except queue.Empty:
#             pass
#         try:
#             q.put_nowait(item) # try once more
#             # print(f"[put_drop] Successfully added item after removing oldest: {type(item)}")
#             # log.info(f"[put_drop] Successfully added item after removing oldest: {type(item)}")
#             return True
#         except queue.Full:
#             # print(f"[put_drop] Failed to add item after removing oldest: {type(item)}")
#             # log.info(f"[put_drop] Failed to add item after removing oldest: {type(item)}")
#             return False
        
async def put_drop(q: asyncio.Queue, item) -> bool:
    try:
        q.put_nowait(item)
        return True
    except asyncio.QueueFull:
        try:
            q.get_nowait()     # drop oldest
        except asyncio.QueueEmpty:
            pass
        try:
            q.put_nowait(item) # try once more
            return True
        except asyncio.QueueFull:
            return False
        
def stop_helper(session, session_id=None):
    # Stop training thread if running
    if session.get("training_thread") and session["training_thread"].is_alive():
        session["stop_requested"] = True
    
    stop_flag=session.get("stop_flag")
    if(stop_flag):
        try:
            stop_flag.stop()
        except Exception:
            pass
    
    session["status"] = "stopped"
    
    # Clean up session from active_sessions if session_id provided
    if session_id and session_id in active_sessions:
        log.info(f"[CLEANUP] Removing session {session_id} from active_sessions")
        del active_sessions[session_id]
        log.info(f"[CLEANUP] Remaining active sessions: {len(active_sessions)}")

def _walk(prefix, routes):
    for r in routes:
        if isinstance(r, WebSocketRoute):
            print(f"[ROUTE][WS]  {prefix}{r.path}")
        elif isinstance(r, Route):
            methods = ",".join(sorted((r.methods or [])))
            print(f"[ROUTE][HTTP] {methods:12} {prefix}{r.path}")
        elif isinstance(r, Mount):
            _walk(prefix + r.path, r.routes)
        
# previously use "queue.Queue"
async def drain_queue_to_ws(websocket, send_lock: asyncio.Lock, q: asyncio.Queue,*,poll_timeout_sec: float=1.0, drop_stale: bool = True):
    # consecutive_fails=0
    log.info(f"[drain_queue_to_ws] Starting queue drain for queue: {q}")
    log.info(f"[drain_queue_to_ws] Queue size: {q.qsize()}")
    log.info(f"[drain_queue_to_ws] WebSocket state: {websocket.application_state}")
    try:
        # while websocket.application_state is WebSocketState.CONNECTED:
        while ws_connected(websocket):
            log.info(f"[drain_queue_to_ws] Loop iteration - Queue size: {q.qsize()}")
            log.info(f"[drain_queue_to_ws] About to call q.get()")
            log.info(f"[drain_queue_to_ws] WebSocket state: {websocket.application_state}")
            
            # Get an item from the queue
            item = None
            try:
                # First try to get an item immediately if available
                try:
                    item = q.get_nowait()
                    log.info(f"[drain_queue_to_ws] Retrieved item from queue (immediate): {type(item)}")
                except asyncio.QueueEmpty:
                    # Queue is empty, wait for an item with timeout
                    log.info("[drain_queue_to_ws] Queue is empty, waiting for items...")
                    item = await asyncio.wait_for(q.get(), timeout=poll_timeout_sec)
                    log.info(f"[drain_queue_to_ws] Retrieved item from queue (after wait): {type(item)}")
            except asyncio.TimeoutError:
                log.info("[drain_queue_to_ws] Timeout waiting for queue item, continuing...")
                log.info(f"[drain_queue_to_ws] WebSocket state after timeout: {websocket.application_state}")
                if websocket.application_state is not WebSocketState.CONNECTED:
                    log.info("[drain_queue_to_ws] WebSocket disconnected during timeout, breaking loop")
                    break
                continue
            
            if item is None:
                log.info("[drain_queue_to_ws] No item retrieved, continuing...")
                # if websocket.application_state is not WebSocketState.CONNECTED:
                if not ws_connected(websocket):
                    log.info("[drain_queue_to_ws] WebSocket disconnected, breaking loop")
                    break
                await asyncio.sleep(0.1)
                continue
            
            if drop_stale:
                try:
                    while True:
                        q.get_nowait()  # Drop stale items, but keep the current item
                        log.info("[drain_queue_to_ws] Dropped stale item")
                except asyncio.QueueEmpty:
                    log.info("[drain_queue_to_ws] No more stale items to drop")
                    pass
            
            log.info(f"[drain_queue_to_ws] Sending item: {type(item)} - {str(item)[:100]}...")
            log.info(f"[drain_queue_to_ws] WebSocket state before send: {websocket.application_state}")
            ok = await safe_send(websocket, item, lock=send_lock)
            if not ok:
                log.info("[drain_queue_to_ws] Failed to send item - breaking out of loop")
                log.info(f"[drain_queue_to_ws] WebSocket state when send failed: {websocket.application_state}")
                break
            else:
                log.info("[drain_queue_to_ws] Successfully sent item to WebSocket")
                log.info(f"[drain_queue_to_ws] WebSocket state after send: {websocket.application_state}")
            
            # Check if WebSocket is still connected after processing
            # if websocket.application_state is not WebSocketState.CONNECTED:
            if not ws_connected(websocket):
                log.info("[drain_queue_to_ws] WebSocket disconnected after processing item, breaking loop")
                break
    # finally:
    #     with contextlib.suppress(Exception):
    #         await try_close(websocket,1000,"exception from drain_queue")
    # except asyncio.CancelledError:
    #     pass
    except Exception as e:
        log.info(f"[drain_queue_to_ws] exception: {repr(e)}")
        traceback.print_exc()
        try:
            log.info("[drain_queue_to_ws] closing websocket due to exception")
            await try_close(websocket,1000,"exception from drain_queue")
        except Exception:
            log.info("[drain_queue_to_ws] failed to close websocket due to exception")
            pass
    finally:
        log.info(f"[drain_queue_to_ws] Exiting drain_queue_to_ws. WebSocket state: {websocket.application_state}, Queue size: {q.qsize()}")
            
# this function is intended to receive the heartbeat from frontend(ws routes)
async def ws_receiver(ws: WebSocket, send_lock: asyncio.Lock, ready_event: asyncio.Event, session: dict, kind: str):
    log.info(f"[ws_receiver] ENTRY: Starting receiver for {kind}")
    unsubscribed = False
    unsubscribed_topic = None
    try:
        # Don't automatically increment subscriber count - wait for subscribe message
        # session["subs"][kind]=session["subs"].get(kind,0)+1
        log.info(f"[ws_receiver] Started receiver for {kind}")
        # while ws.application_state is WebSocketState.CONNECTED:
        while ws_connected(ws):
            try:
                log.info(f"[ws_receiver] Waiting for message from {kind}")
                data=await ws.receive_text()
                log.info(f"[ws_receiver] Received raw data from {kind}: {data}")
                try:
                    msg = json.loads(data)
                    log.info(f"[ws_receiver] Parsed message from {kind}: {msg}")
                except Exception as e:
                    log.info(f"[ws_receiver] Failed to parse message from {kind}: {e}")
                    # msg = {"type":"text","data":data}
                    continue
                    
                t = msg.get("type") if isinstance(msg,dict) else None
                log.info(f"[ws_receiver] Received message type: {t} for {kind}")
                if t=="ping":
                    # Handle ping messages (can be frequent, so don't log every one)
                    if send_lock is not None:
                        await safe_send(ws,{"type":"pong","ts":time.time()},lock=send_lock)
                    else:
                        await safe_send(ws,{"type":"pong","ts":time.time()})
                elif t=="ready":
                    # Handle duplicate ready messages gracefully
                    if not ready_event.is_set():
                        ready_event.set()
                        log.info(f"[ws_receiver] Ready event set for {kind}")
                    else:
                        log.info(f"[ws_receiver] Duplicate ready message received for {kind}, ignoring")
                elif t=="subscribe":
                    topic = msg.get("topic")
                    if topic:
                        session["subs"][topic] = session["subs"].get(topic, 0) + 1
                        log.info(f"[ws_receiver] Incremented {topic} subscribers to {session['subs'][topic]}")
                    else:
                        log.info(f"[ws_receiver] Subscribe message missing topic: {msg}")
                elif t=="unsubscribe":
                    topic = msg.get("topic")
                    if topic in session["subs"]:
                        session["subs"][topic]=session["subs"].get(topic,1)-1
                    unsubscribed = True
                    unsubscribed_topic = topic
                    break
                elif t=="stop":
                    stop_helper(session)
                    break
            except WebSocketDisconnect:
                # session["stop_requested"] = True
                print("[WS] ws_receiver has a WebSocketDisconnect")
                break
            except asyncio.CancelledError:
                break
            except Exception:
                continue
    except Exception:
        pass
    finally:
        # Only decrement if we actually unsubscribed from a specific topic
        if unsubscribed and unsubscribed_topic:
            session["subs"][unsubscribed_topic]=max(0,session["subs"].get(unsubscribed_topic,1)-1)
            log.info(f"[ws_receiver] Finally block: decremented {unsubscribed_topic} subscribers to {session['subs'][unsubscribed_topic]}")
        else:
            log.info(f"[ws_receiver] Finally block: no decrement needed (no unsubscribe message received)")

async def check_loop(websocket: WebSocket, session: dict, sender_task: asyncio.Task, receiver_task: asyncio.Task, send_lock: asyncio.Lock):
    while True:
        if( websocket.application_state is WebSocketState.CONNECTED
            and session.get("training_thread") is not None
            and session["training_thread"].is_alive()
            and not session.get("stop_requested")):
            sender_task.cancel()
            receiver_task.cancel()
            break
        else:
            await asyncio.sleep(0.1)

log = logging.getLogger("uvicorn.error")
# Unique marker for this build/process
_MARK = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:12]
log.info(f"[BOOT] api_server.py hash={_MARK} pid={os.getpid()} time={datetime.utcnow().isoformat()}Z")

# After app/router setup, dump just the WS routes once
def _dump_routes():
    from starlette.routing import WebSocketRoute, Route
    ws = [r.path for r in app.router.routes if isinstance(r, WebSocketRoute)]
    http = [r.path for r in app.router.routes if isinstance(r, Route)]
    log.info(f"[ROUTES] WS={ws}")
    log.info(f"[ROUTES] HTTP={http}")
_dump_routes()

# @app.post("/api/training/start")
#config: the input argument that holdes training settings(i.e. algorithms, episodes, etc.)
@api.post("/training/start")
async def start_training(config: TrainingConfig) -> TrainingResponse:
    try:
        log.info("[TRAINING START] Starting training initialization...")
        log.info(f"[TRAINING START] Config: algorithm={config.algorithm}, environment={config.environment}")
        
        # Check resource limits before starting
        import psutil
        import os
        process = psutil.Process(os.getpid())
        current_memory = process.memory_info().rss / 1024 / 1024
        
        # Don't start new training if memory is already high
        if current_memory > 2500:  # 2.5GB threshold
            log.warning(f"[TRAINING START] Memory already high: {current_memory:.2f} MB, refusing to start training")
            raise HTTPException(status_code=503, detail=f"Server memory usage too high ({current_memory:.2f} MB). Please wait and try again.")
        
        # Check if too many active sessions
        if len(active_sessions) >= 3:  # Limit concurrent training sessions
            log.warning(f"[TRAINING START] Too many active sessions: {len(active_sessions)}")
            raise HTTPException(status_code=503, detail="Too many concurrent training sessions. Please wait for one to complete.")
        
        log.info("[TRAINING START] Importing Environment module...")
        from Env.env_module import Environment
        log.info("[TRAINING START] Environment module imported successfully")
        
        # from DQN_test.your_dqn_file import DQN related Class
        log.info("[TRAINING START] Importing DQN module...")
        from DQN_test.DQN_module import Agent as DQNAgent, DQN_args_parser, writer as dqn_writer
        # dqn_args,dqn_parser=DQN_args_parser()
        dqn_parser = DQN_args_parser()
        dqn_args = dqn_parser.parse_args([])
        log.info("[TRAINING START] DQN module imported successfully")

        # from PPO_test.your_ppo_file import PPO related Class
        log.info("[TRAINING START] Importing PPO module...")
        from PPO_test.PPO_module import Agent as PPOAgent, PPO_args_parser, writer as ppo_writer
        # ppo_args,ppo_parser=PPO_args_parser()
        ppo_parser = PPO_args_parser()
        ppo_args = ppo_parser.parse_args([])
        log.info("[TRAINING START] PPO module imported successfully")
        
        # hb_task = None
        # Generate session ID
        session_id = f"session_{int(time.time())}"
        log.info(f"[TRAINING START] Created session: {session_id}")
        
        # Create environment based on config
        log.info("[TRAINING START] Creating environment...")
        if config.environment == "warehouse":
            total_packages = config.warehouse_config.get("total_packages") if config.warehouse_config else 146
            if(layoutModified["status"]=="layout_modified"):
                updated_layout=layoutModified["layout"]
                log.info("Modified Layout in Start Training!")
                env = Environment(layout=updated_layout, done_standard=total_packages)
            else:#if the layout has not been modified
                env = Environment(done_standard=total_packages)
        else:#navigation environment(just in case if there is another option)
            # For navigation environment, use default settings
            env = Environment(done_standard=146)  # we don't alter the goal right now
        log.info("[TRAINING START] Environment created successfully")
        
        # Create agent based on algorithm
        log.info(f"[TRAINING START] Creating {config.algorithm} agent...")
        
        # Monitor memory before agent creation
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024
        log.info(f"[TRAINING START] Memory before agent creation: {memory_before:.2f} MB")
        
        if config.algorithm == "dqn":
            # Update DQN args with user config(only when there are really input from the users)
            if config.learning_rate:
                dqn_args.learning_rate = config.learning_rate
            #if not, we just use the default value
            log.info("[TRAINING START] Creating DQN agent...")
            agent = DQNAgent(dqn_args, env, dqn_writer, False, False,True)  # ThreeD=False for web
            log.info("[TRAINING START] DQN agent created successfully")
        elif config.algorithm == "ppo":
            # Update PPO args with user config(if there really exist user input)
            if config.learning_rate:
                ppo_args.actor_lr = config.learning_rate
                ppo_args.critic_lr = config.learning_rate * 2
            #if not, we just use the default value from PPO module
            log.info("[TRAINING START] Creating PPO agent...")
            agent = PPOAgent(ppo_args, env, ppo_writer, False,True)  # ThreeD=False for web
            log.info("[TRAINING START] PPO agent created successfully")
        else:
            raise ValueError(f"Unknown algorithm: {config.algorithm}")
        
        # Monitor memory after agent creation
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_increase = memory_after - memory_before
        log.info(f"[TRAINING START] Memory after agent creation: {memory_after:.2f} MB (+{memory_increase:.2f} MB)")
        
        # Check if memory usage is too high (conservative threshold for 4GB container)
        if memory_after > 2800:  # 2.8GB threshold (leave 1.2GB buffer)
            log.warning(f"[TRAINING START] High memory usage detected: {memory_after:.2f} MB")
            # Force garbage collection
            import gc
            gc.collect()
            
            # Check memory after cleanup
            memory_after_cleanup = process.memory_info().rss / 1024 / 1024
            log.info(f"[TRAINING START] Memory after cleanup: {memory_after_cleanup:.2f} MB")
            
            # If still too high, warn about potential issues
            if memory_after_cleanup > 3200:  # 3.2GB critical threshold
                log.error(f"[TRAINING START] CRITICAL: Memory usage still too high: {memory_after_cleanup:.2f} MB")
                log.error("[TRAINING START] Training may fail due to memory constraints")
        
        # Store session info
        active_sessions[session_id] = {
            "config": config,
            "env": env,
            "agent": agent,
            # "total_episodes": 1000,
            "status": "ready",
            "stop_flag": StopFlag(),
            "progress": 0,
            "current_episode": 0,
            "training_thread": None,
            "progress_q": asyncio.Queue(maxsize=1), #this manages the progress(make training push updates once and WS routes await fresh itmes)
            "metrics2d": asyncio.Queue(maxsize=1), #this is the 2d metrics queue
            "metrics3d": asyncio.Queue(maxsize=1), #this is the 3d metrics queue
            "frame_q": asyncio.Queue(maxsize=1), #this is for plot streaming at the visualization time(2d)
            "frame3d": asyncio.Queue(maxsize=1), #this is for plot streaming for 3d
            # "2d_subs": 0,
            # "3d_subs":0,
            "subs":{
                "2d_subs":0,
                "2d_metrics_subs":0,
                "3d_subs":0,
                "3d_metrics_subs":0,
                "training":0,
                },
            "progress_log": {
                "episode":0,
                "progress":0,
                "reward":0,
                "delivered":0,
                "image":None,
                "time":0,
                "loss":0.0,
                "epsilon":0.0,
                "actor_loss":0.0,
                "critic_loss":0.0,
                "time_text":"0.0s",
                "current_algo":None
                } #this is for the parameters needed for the render function
        }
        
        # # we add this line here to set the stopflag(originally set in websocket endpoint)
        # if "stop_flag" not in active_sessions[session_id]:
        #     active_sessions[session_id]["stop_flag"]=StopFlag()
        
        return TrainingResponse(
            success=True,
            session_id=session_id,
            message=f"Training session created with {config.algorithm.upper()} algorithm"
        )
        
    except Exception as e:
        import traceback, sys
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

# this function is intended to show whether the environment has been initiallzed(whether we have decided "start_training")
@api.get("/env_init/{session_id}")
async def env_init(session_id: str):
    session=active_sessions.get(session_id)
    # this below line for env and stop_flag are added to ensure that the plot is shown as expected:
    # return {"is_ready":session is not None and session.get("env") is not None}
    return {"is_ready":session is not None and session.get("env") is not None and session.get("stop_flag") is not None}
            # "instance":_instance_id()}

@api.post("/training/stop/{session_id}")
async def stop_training(session_id: str):
    if session_id not in active_sessions:
        # raise HTTPException(status_code=404, detail="Session not found")
        # currently we treat the already quiet as success:
        return {"success": True, "message":"No such session(already stopped)"}
    
    session = active_sessions[session_id]
    stop_helper(session, session_id)
    return {"success": True, "message": "Training stopped"}

#this is intended for getting stop_status
def stop_status():
    session=active_sessions["active_session"]
    if(session["status"]=="stopped"):
        return True
    return False

@api.get("/training/status/{session_id}")
async def get_training_status(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    return {
        "session_id": session_id,
        "status": session["status"],
        "progress": session["progress"],
        "current_episode": session["current_episode"],
        "algorithm": session["config"].algorithm,
        "environment": session["config"].environment
    }

#this function is used to received the altered layout
@api.post("/layout_modification")
async def modify_layout(data: LayoutRequest=Body(...)): #this means to pass a data in the format of the "LayoutRequeset" to the parameter field
    print("🔥🔥🔥 Received layout modification request")
    # to make sure python know that we are modifying the global level dictionary(not a local one)
    global layoutModified
    try:
        # row_layout=data.layout.strip()
        # raw_layout='\n'.join(row_layout)
        
        # # Safe eval for tuples/lists of strings
        parsed_layout = ast.literal_eval(data.layout)
        layoutRaw["layout"]=parsed_layout
        
        raw_layout="\n".join(parsed_layout)
        
        # Now you can pass it to your environment(this is used in just in set environment part)
        # active_sessions[session_id]["env"] = active_sessions[session_id]["env"](parsed_layout)
        layoutModified["status"]="layout_modified"
        layoutModified["layout"]=copy.deepcopy(raw_layout)
        print("🌛🌛🌛Modified layout received and stored:")
        print(layoutModified)

        # return {"status": "success"}
        return {"status": "success"}
    
    except Exception as e:
        return {"There is error with modified layout:": str(e)}

#this is intended for creating a comment area
@api.post("/add_msg/")
async def AddComment(comment: Comment):
    # if comment.username not in messages:
    #     messages[comment.username]=[] #this means to create a section to store all the messages related with this session inside the message
    messages.append(comment)
    return {"status":"success"}

# this is intended to get all the messages from a specific session
@api.get("/msg/")
async def GetComment():
    # try:
    #     return messages.get(user,[])
    # except:
    #     raise Exception(f"404:Comments invalid!")
    return messages

#currently we don't need this so much - this is just for returning a static snapshot(not for continuous update)
# @app.get("/api/matplotlib/{session_id}")
@api.get("/plots/{plot_type}/{session_id}")
async def get_matplotlib_plot(plot_type: str, session_id: str):
    """This is the original version of using matplotlib render from the environment render
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    #to make sure the "plot_type" is not empty(because of the issue of the path url, we have to do this here)
    #this means that if "plot_type" is False, emtpy, etc, we use "training_progess"
    plot_type = plot_type or "training_progress"
    
    try:
        session = active_sessions[session_id]
        env = session["env"]
        #get the parameters for render function inside environment class
        episode=session["progress_log"]["episode"]
        current_progress=session["progress_log"]["progress"] #this is the progress counted in percentage(the number beside '%')
        total_reward=session["progress_log"]["reward"]
        total_time=session["progress_log"]["time"]
        # current_delivered=session["progress_log"]["delivered"]
        current_delivered=env.count_delivered()
        return4web=True
        
        if session["progress_log"] is None:
            raise HTTPException(status_code=400, detail="Plot not available yet. Training may not have started.")
        
        # Create a matplotlib plot using your existing visualization
        img_shown=env.render(mode="human", close=False, reward=total_reward, count_time=total_time, delivered=current_delivered, ThreeD_vis=False,return64web=return4web)

        #we decode the string back to raw image bytes(this is done because we need to pass it to the stream)
        byte_version=base64.b64decode(img_shown)
        
        # Convert plot to base64 string
        buffer = io.BytesIO(byte_version)
        # buffer.seek(0) #we don't need this here because we are already putting 'byte_version' into the buffer
        base64_str=base64.b64encode(byte_version).decode("utf-8")
        return {"image_base64": base64_str}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate plot: {str(e)}")

app.include_router(api)

# def get_ws_lock(session):
#     lock = session.get("_send_lock")
#     if lock is None:
#         lock = asyncio.Lock()
#         session["_send_lock"] = lock
#     return lock

# This below version is the updated(the better version of lock that can be separated by the route):
def get_ws_lock(session,route_name: str) -> asyncio.Lock:
    locks = session.setdefault("_route_locks",{})
    lock = locks.get(route_name)
    if not lock:
        lock = asyncio.Lock()
        locks[route_name] = lock
    return lock

async def _send(ws: WebSocket, payload):
    if isinstance(payload, (dict, list)):
        await ws.send_json(payload)
    elif isinstance(payload, bytes):
        await ws.send_bytes(payload)
    elif isinstance(payload, str):
        await ws.send_text(payload)
    else:
        await ws.send_text(json.dumps(payload, default=str))

# this function is to prevent send after close(for normal close sessions)
async def try_close(ws: WebSocket, code: int=1000, reason:str=""):
    try:
        if ws.application_state is WebSocketState.CONNECTED:
            await ws.close(code=code, reason=reason)
        # await ws.close(code=code, reason=reason)
    except Exception:
        pass

# this is used to ensure safesend with no delay
async def safe_send(ws: WebSocket, message, lock: asyncio.Lock | None = None, timeout: float = 0.25) -> bool:
    # Bail if not connected
    if not ws_connected(ws):
        log.info("[safe_send] WebSocket not connected, returning False")
        return False

    async def _forward():
        await _send(ws,message)

    try:
        if lock is not None:
            async with lock:
                # Re-check inside the lock in case state changed
                if not ws_connected(ws):
                    log.info("[safe_send] WebSocket not connected inside lock, returning False")
                    return False
                if timeout is None:
                    await _forward()
                else:
                    await asyncio.wait_for(_forward(),timeout=timeout)
                log.info("[safe_send] Successfully sent message with lock")
                return True
        else:
            if timeout is None:
                await _forward()
            else:
                await asyncio.wait_for(_forward(), timeout=timeout)
            log.info("[safe_send] Successfully sent message without lock")
            return True
    except WebSocketDisconnect:
        log.info("[safe_send] WebSocket disconnected during send")
        return False
    except Exception as e:
        log.info(f"[safe_send] Error sending message: {e}")
        return False

@app.on_event("startup")
async def show_routes():
    print("=== ROUTES AT STARTUP ===")
    _walk("", app.router.routes)
    print("=========================")
    
    # Debug: Log WebSocket routes specifically
    log = logging.getLogger("uvicorn.error")
    log.info("=== WEBSOCKET ROUTES ===")
    for route in app.router.routes:
        if hasattr(route, 'path') and '/ws/' in route.path:
            log.info(f"WebSocket route: {route.path}")
    log.info("========================")

@app.websocket("/ws/layout/{session_id}")
async def websocket_layout(websocket: WebSocket, session_id:str):
    # Temporarily added to check the issue(below):
    log = logging.getLogger("uvicorn.error")
    log.info(f"[WS ROUTE] layout ENTER session={session_id} pid={os.getpid()}")
    
    print("[WS ROUTE] Entered layout websocket route")
    print("[WS LAYOUT] entered", session_id, type(session_id).__name__)
    print("[WS ROUTE] layout attempt", session_id, "origin=", websocket.headers.get("origin"))
    
    # Debug: Log that the route is being called
    log.info(f"[WS LAYOUT] Route called for session: {session_id}")
    print(f"[WS LAYOUT] Route called for session: {session_id}")
    
    # Debug: Log all active sessions
    log.info(f"[WS LAYOUT] Active sessions: {list(active_sessions.keys())}")
    log.info(f"[WS LAYOUT] Requested session: {session_id}")
    
    # await websocket.accept()
    
    if session_id not in active_sessions:
        log.info(f"[layout sessionid] no session id for layout route. Available sessions: {list(active_sessions.keys())}")
        await try_close(websocket,1008,"3D layout but no session")
        return
    
    origin = websocket.headers.get("origin")
    if origin not in origins:
        log.info("[origin not exist] the origin of layout route is not existed")
        await try_close(websocket,1008,"3D plot but no origin(origin not allowed)")
        return
    
    # previously we have the environment check and stop_flag check here[NECESSARY]
    await websocket.accept()
    send_lock = asyncio.Lock()

    # env & session & stop_flag check
    session = active_sessions.get(session_id)
    for _ in range(60):
        session = active_sessions.get(session_id)
        if not session:
            break
        environment = session.get("env")
        stop_flag = session.get("stop_flag")
        
        if environment is not None and stop_flag is not None and ("progress_log" in session):
            break
        
        await safe_send(websocket, {"type":"not ready"}, lock = send_lock)
        await asyncio.sleep(0.1)
        
    session = active_sessions.get(session_id)
    if not session or not session.get("stop_flag") or ("progress_log" not in session):
        # NOTE: here we temporarily commented these things and replace with "not ready"
        # this helps to keep use alive but not open the socket
        log.info("[layout close] either no session, or no stop_flag, or no progress_log")
        await try_close(websocket,1011,"3D layout not ready")
        return

    env = session.get("env")
    stop_flag = session.get("stop_flag")
    if env is None or stop_flag is None:
        # await safe_send(websocket, {"type":"error", "error":"environemnt not ready"}, lock = send_lock)
        log.info("[layout close] either no environment, or no stop_flag")
        await try_close(websocket,1008,"3D layout but no env and no stop_flag")
        # await safe_send(websocket,{"type":"not ready"},lock=send_lock)
        return
    await safe_send(websocket, {"type":"ready"}, lock = send_lock)
    log.info("[WS LAYOUT] Ready message sent, proceeding to create ws_receiver task")
    
    # send_lock = asyncio.Lock()
    
    # Temporarily we want to move this below block to the upper level[COMMENTED]
    ready_event = asyncio.Event()
    log.info("[WS LAYOUT] Ready event created, about to create ws_receiver task")
    # this below line is to receive the heartbeat from the frontend(replace the heartbeat)
    log.info("[WS LAYOUT] Creating ws_receiver task for 3d_subs")
    receiver_task = asyncio.create_task(ws_receiver(websocket, send_lock, ready_event, session, "3d_subs"))
    log.info("[WS LAYOUT] ws_receiver task created")
    
    # hb_task = asyncio.create_task(heartbeat(websocket, interval=15.0))
    try:
        # previously use "queue.Queue"
        frame_q: asyncio.Queue = session.get("frame3d")
        sender_task = asyncio.create_task(drain_queue_to_ws(websocket, send_lock, frame_q))
        
        try:
            await asyncio.wait_for(ready_event.wait(),timeout=8.0)
        except asyncio.TimeoutError:
            print("[WS] did not receive 'ready' in time")
            log.info("[layout error] timeout error")
            # await try_close(websocket,1000,"timeout and not ready from the client")
            # receiver_task.cancel()
            # with contextlib.suppress(asyncio.CancelledError): await receiver_task
            # return
            await safe_send(websocket, {"type":"not ready"}, lock = send_lock)
        
        print("[READY EVENT] 3D layout DID receive 'ready'")
        log.info("[LAYOUT READY] the layout IS READY!")
        # try:
        #     done, pending = await asyncio.wait(
        #         {asyncio.create_task(websocket.receive_json())},
        #         timeout=8.0,
        #         return_when = asyncio.FIRST_COMPLETED,
        #     )
        #     for t in pending:
        #         t.cancel()
        #     if done:
        #         msg = list(done)[0].result()
        #         if msg.get("type")!="ready":
        #             print(f"[3d layout] Unexpected first message: {msg}")
        # except Exception as e:
        #     print(f"[3d layout] Did not receive 'ready' in time: {e}")

        # original check block

        # have a FPS cap
        FPS = 10
        min_dt = 1.0/FPS
        last = 0.0

        # Wait for subscribers before starting the main loop
        log.info("[Layout WS] Waiting for 3d_subs subscribers...")
        while websocket.application_state is WebSocketState.CONNECTED and session["subs"].get("3d_subs", 0) == 0:
            await asyncio.sleep(0.1)
        
        if websocket.application_state is not WebSocketState.CONNECTED:
            log.info("[Layout WS] WebSocket disconnected while waiting for subscribers")
            return
        
        log.info(f"[Layout WS] Found {session['subs'].get('3d_subs', 0)} 3d_subs subscribers, starting main loop")
        log.info(f"[Layout WS] Pre-loop check - WebSocket state: {websocket.application_state}, env.done: {env.done}, stop_flag: {stop_flag()}")
        log.info(f"[Layout WS] Pre-loop check - Session subs: {session['subs']}")

        # Main loop: stream layout until training is done
        loop_count = 0
        last_layout_hash = None
        log.info(f"[Layout WS] About to start main loop - WebSocket state: {websocket.application_state}, env.done: {env.done}, stop_flag: {stop_flag()}")
        
        while websocket.application_state is WebSocketState.CONNECTED and not env.done and not stop_flag():
            try:
                loop_count += 1
                log.info(f"[Layout WS] Loop iteration {loop_count} - Checking for layout changes...")
                log.info(f"[Layout WS] WebSocket state: {websocket.application_state}, env.done: {env.done}, stop_flag: {stop_flag()}")
                log.info(f"[Layout WS] Training status: {session.get('status', 'unknown')}")
                log.info(f"[Layout WS] Episode: {session.get('progress_log', {}).get('episode', 'unknown')}")
                log.info(f"[Layout WS] Delivered: {session.get('progress_log', {}).get('delivered', 'unknown')}")
                
                now = time.time()
                if now - last < min_dt:
                    await asyncio.sleep(min_dt - (now - last))
                    continue
                last = now
                
                # debugging line
                log.info(f"[Layout WS] Checking subscribers: 3d_subs = {session['subs'].get('3d_subs', 0)}")
                log.info(f"[Layout WS] Session subs: {session['subs']}")
                # now we add this to our queue
                if(session["subs"].get("3d_subs",0)>0):
                    try:
                        layout_3d = env.stream_layout()
                        
                        # Create a hash of the layout to detect changes
                        import hashlib
                        layout_str = str(layout_3d.get('layout', []))
                        current_layout_hash = hashlib.md5(layout_str.encode()).hexdigest()
                        
                        # Only send if layout has changed
                        if current_layout_hash != last_layout_hash:
                            log.info(f"[Layout WS] Layout changed! Hash: {current_layout_hash[:8]}...")
                            log.info(f"[Layout WS] Generated layout data: {type(layout_3d)} - {str(layout_3d)[:100]}...")
                            # Send the layout data in the correct format expected by frontend
                            success = await put_drop(frame_q, layout_3d)
                            log.info(f"[Layout WS] Layout queued successfully: {success}")
                            log.info(f"[Layout WS] Frame queue size: {frame_q.qsize()}")
                            last_layout_hash = current_layout_hash
                            
                            # Yield control to allow WebSocket tasks to process the queue
                            await asyncio.sleep(0)
                        else:
                            log.info(f"[Layout WS] Layout unchanged, skipping send. Hash: {current_layout_hash[:8]}...")
                    except Exception as layout_error:
                        log.error(f"[Layout WS] Error in layout generation: {layout_error}")
                        # Continue the loop even if layout generation fails
                        await asyncio.sleep(0.5)
                else:
                    log.info(f"[Layout WS] No 3d_subs subscribers: {session['subs'].get('3d_subs', 0)}")
                    # Adjust speed if needed
                    await asyncio.sleep(0.2)
            except WebSocketDisconnect:
                log.info("Client disconnected (image update)")
                log.info("[layout disconnected]")
                break
            except Exception as e:
                log.info(f"Render/WebSocket error: {e}")
                log.info("[layout render/websocket error]")
                try:
                    await safe_send(websocket,{"error": str(e)},lock=send_lock)
                    # await websocket.send_text(json.dumps({"error": str(e)}))
                except:
                    print("Client disconnected during error handling")
                    break
            
            # WebSocket tasks are already running concurrently
            
            await asyncio.sleep(0.5)
        
        # Log why the loop exited
        log.info(f"[Layout WS] Loop exited after {loop_count} iterations")
        log.info(f"[Layout WS] Exit reason - WebSocket state: {websocket.application_state}, env.done: {env.done}, stop_flag: {stop_flag()}")
        log.info("[layout close] normal close")
        await try_close(websocket,1000,"normal close for 3d layout")
        return
    except WebSocketDisconnect:
        print("[Layout WS] Disconnected.")
        log.info("[direct disconnect]")
    except Exception as e:
        log.info("[layout internal error]")
        reason = (str(e) or "internal error for 3d plot")[:120]
        try:
            await try_close(websocket,1011,reason)
        except Exception:
            pass
    finally:
        log.info("[layout final block]")
        # if hb_task:
        #     hb_task.cancel()
        try:
            log.info("layout tasks cancelled at final")
            sender_task.cancel()
            receiver_task.cancel()
            with contextlib.suppress(asyncio.CancelledError): await receiver_task
        except Exception:
            pass
        # if(websocket.application_state==WebSocketState.CONNECTED):
        #     await websocket.close(code=1000, reason="normal close for layout")
        await try_close(websocket,1000,"normal close for 3d layout")
        return


@app.websocket("/ws/plot3d/{session_id}")
async def metrics3d(websocket: WebSocket, session_id:str):
    print(f"[BACKEND] WebSocket layout route hit: session_id = {session_id}")
    # await websocket.accept()

    if session_id not in active_sessions:
        # this accept line is added later(just to test)
        # await websocket.accept()
        # await websocket.close(code=1008,reason="3D layout but no session")
        await try_close(websocket,1008,"3D metrics but no session")
        return
    
    origin = websocket.headers.get("origin")
    if origin not in origins:
        # await websocket.accept()
        # await websocket.close(code=1008,reason="3D metrics but we don't have origin")
        await try_close(websocket,1008,"3D metrics but we don't have origin")
        return

    # [NECESSARY]
    await websocket.accept()
    
    # env & session & stop_flag check

    send_lock = asyncio.Lock()
    
    session = active_sessions.get(session_id)
    for _ in range(50):  # ~5s total
        session = active_sessions.get(session_id)
        if session and session.get("stop_flag") and "progress_log" in session:
            break
        await asyncio.sleep(0.1)
    
    if not session or not session.get("stop_flag") or "progress_log" not in session:
        # await safe_send(websocket, {"type":"error","error":"not ready"},lock=send_lock)
        # await websocket.close(code=1011, reason="not ready for 3d metric showing")
        await try_close(websocket,1011,"not ready for 3d metric showing")
        return
    
    env = session.get("env") if session else None
    stop_flag = session.get("stop_flag")
    if env is None or stop_flag is None:
        # await websocket.accept()
        # await websocket.close(code=1008, reason="3D metrics but no env or no stop_flag")
        await try_close(websocket,1008,"3D metrics but no env or no stop_flag")
        return
    
    # this line is intended to check whether the sending thing works here
    log.info("[BACKEND] WebSocket accepted")

    # Temporarily we want to move the below block to upper level[COMMENTED]
    ready_event = asyncio.Event()
    log.info("[WS PLOT3D] Ready event created, about to create ws_receiver task")
    # hb_task = asyncio.create_task(heartbeat(websocket, interval=15.0))
    log.info("[WS PLOT3D] Creating ws_receiver task for 3d_metrics_subs")
    receiver_task = asyncio.create_task(ws_receiver(websocket, send_lock,ready_event,session,"3d_metrics_subs"))
    log.info("[WS PLOT3D] ws_receiver task created")
    
    await safe_send(websocket, {"type": "ready", "session_id": session_id}, lock=send_lock)
    try:
        # previously use "queue.Queue"
        progress_q: asyncio.Queue = session.get("metrics3d")
        if progress_q is None:
            await try_close(websocket, 1011, "[WS] 3d metrics queue missing")
            return
        sender_task = asyncio.create_task(drain_queue_to_ws(websocket, send_lock, progress_q))
        
        try:
            await asyncio.wait_for(ready_event.wait(),timeout=8.0)
        except asyncio.TimeoutError:
            print("[WS] 3d metrics didn't receive")
            # await try_close(websocket,1000,"timeout and not ready from the client")
            # receiver_task.cancel()
            # with contextlib.suppress(asyncio.CancelledError): await receiver_task
            # return
            await safe_send(websocket, {"type":"not ready"}, lock = send_lock)

        # original check block
        
        # await env.stream_layout(websocket)
        while websocket.application_state is WebSocketState.CONNECTED and not env.done and not stop_flag():
            try:
                log.info("[WS PLOT3D] Getting inside the loop")
                # await asyncio.sleep(0.2)
                progress=session["progress_log"]
                delivered_package=env.count_delivered()
                
                # await asyncio.sleep(0.5)
                
                #let's assume we have successfully did the visulization
                # if not await safe_send(websocket,{"type":"render","stopped":stop_flag(),**active_sessions[session_id]["progress_log"],"number_delivered":delivered_package}, lock=send_lock):
                #     break
                if(session["subs"].get("3d_metrics_subs",0)>0):
                    await put_drop(progress_q,{"type":"render","stopped":stop_flag(),**active_sessions[session_id]["progress_log"],"number_delivered":delivered_package})
                    log.info("[WS PLOT3D] metrics3d_q enqueued")
                    
                    # Yield control to allow WebSocket tasks to process the queue
                    await asyncio.sleep(0)
                else:
                    await asyncio.sleep(0.2)
            except WebSocketDisconnect:
                log.info("[WS PLOT3D] Client disconnected")
                print("Client disconnected")
                break
            except Exception as e:
                log.info("[WS PLOT3D] 3D Render/WEbSocket error:{e}")
                print(f"3D Render/WEbSocket error:{e}")
                try:
                    # await asyncio.sleep(0.01)
                    await safe_send(websocket,{"error": str(e)},lock=send_lock)
                    # await websocket.send_text(json.dumps({"error": str(e)}))
                except:
                    log.info("[WS PLOT3D] Client disconnected during error handling")
                    print("Client disconnected during error handling")
                    break
            
            # WebSocket tasks are already running concurrently
            
            # 5 updates/sec, adjustable
            # await asyncio.sleep(0.2)

        # await websocket.close(code=1008, reason="3D metrics closed but should not close")
        await try_close(websocket,1000,"3D metrics closed - it should be closed")
        return
    except Exception as e:
        log.info("[WS PLOT3D] Unexpected error: {e}")
        print(f"Unexpected error: {e}")
        try:
            await asyncio.sleep(0.01)
            # await safe_send(websocket,{"error": str(e)},lock=send_lock)
            # await websocket.send_text(json.dumps({"error": str(e)}))
            await try_close(websocket,1011,"3D metrics but there is exception")
        except:
            pass
        # await try_close(websocket,1008,"3D metrics but there is exception")
        # return
    finally:
        # if hb_task:
        #     hb_task.cancel()
        try:
            sender_task.cancel()
            receiver_task.cancel()
            with contextlib.suppress(asyncio.CancelledError): await receiver_task
        except Exception:
            pass
        # if(websocket.application_state==WebSocketState.CONNECTED):
        #     await websocket.close(code=1000, reason="normal close for 3D plot metrics")
        await try_close(websocket,1000,"normal close for 3D plot metrics")
        return
            
"""this function is used to compute the metrics for 2D"""
@app.websocket("/ws/plot/{session_id}")
async def metrics2d(websocket: WebSocket, session_id: str):
    print(f"[WebSocket] Connected for session {session_id}")
    # await websocket.accept()

    # hb_task = None
    if session_id not in active_sessions:
        # await websocket.accept()
        await try_close(websocket,1008,"2D plot but no valid session_id")
        return

    origin = websocket.headers.get("origin")
    if origin not in origins:
        # await websocket.accept()
        await try_close(websocket,1008,"2D plot but no origin")
        return
    
    # [NECESSARY]
    await websocket.accept()
    send_lock = asyncio.Lock()
    
    session = active_sessions.get(session_id)
    for _ in range(60):  # up to ~6s with 0.1s sleeps
        session = active_sessions.get(session_id)
        if session and session.get("stop_flag") and ("progress_log" in session):
            break
        await asyncio.sleep(0.1)
    
    if not session or not session.get("stop_flag") or ("progress_log" not in session):
        await safe_send(websocket, {"type":"error","error":"not ready"},lock=send_lock)
        # await websocket.close(code=1011,reason="not ready for 2d metrics")
        await try_close(websocket,1011,"not ready for 2d metrics")
        return
    
    env = session.get("env")
    stop_flag=session.get("stop_flag")
    if env is None or stop_flag is None:
        # await websocket.accept()
        if(env is None):
            # await websocket.close(code=1008, reason="2D plot but no env")
            await try_close(websocket,1008,"2D plot but no env")
            return
        elif(stop_flag is None):
            # await websocket.close(code=1008, reason="2D plot but no stop_flag")
            await try_close(websocket,1008,"2D plot but no stop_flag")
            return
        
        # await websocket.close(code=1008, reason="2D plot but no env or no stop_flag")
        await try_close(websocket,1008,"2D plot but no env or no stop_flag")
        return
    
    # send_lock = get_ws_lock(session,"plot")
    ready_event = asyncio.Event()
    receiver_task = asyncio.create_task(ws_receiver(websocket, send_lock, ready_event, session,"2d_metrics_subs"))
    # hb_task = asyncio.create_task(heartbeat(websocket, interval=15.0))
    try:
        # previously use "queue.Queue"
        progress_q: asyncio.Queue = session.get("metrics2d")
        sender_task = asyncio.create_task(drain_queue_to_ws(websocket,send_lock,progress_q))
        
        try:
            await asyncio.wait_for(ready_event.wait(),timeout=8.0)
        except asyncio.TimeoutError:
            print("[WS] 2d metrics didn't receive anything")
            # await try_close(websocket,1000,"timeout and not ready from the client")
            # receiver_task.cancel()
            # with contextlib.suppress(asyncio.CancelledError): await receiver_task
            # return
            await safe_send(websocket, {"type":"not ready"}, lock = send_lock)
        
        # original check block
        
        while websocket.application_state is WebSocketState.CONNECTED and not env.done and not stop_flag():
            try:
                #(debug)
                print("[DEBUG] env.done =", env.done)
                # print("[DEBUG] progress =", session.get("progress_log"))
                
                # Update environment state and render image
                progress=session["progress_log"]
                
                #get the parameters for render function inside environment class
                total_time=progress["time"]
                
                # current_delivered=session["progress_log"]["delivered"]
                current_delivered=env.count_delivered()

                # We COMMENTED this out to ensure that we only send from the queue(COMMENTED):
                # if not await safe_send(websocket,{"type":"render","stopped":stop_flag(),**active_sessions[session_id]["progress_log"],"number_delivered":current_delivered},lock=send_lock):
                #     break
                
                if(session["subs"].get("2d_metrics_subs",0)>0):
                    await put_drop(progress_q,{"type":"render","stopped":stop_flag(),**active_sessions[session_id]["progress_log"],"number_delivered":current_delivered})
                    
                    # Yield control to allow WebSocket tasks to process the queue
                    await asyncio.sleep(0)
                else:
                    await asyncio.sleep(0.2)
            except WebSocketDisconnect:
                print("Client disconnected (image update)")
                break
            except Exception as e:
                print(f"Render/WebSocket error: {e}")
                try:
                    await safe_send(websocket,{"error": str(e)},lock=send_lock)
                    # await websocket.send_text(json.dumps({"error": str(e)}))
                except:
                    print("Client disconnected during error handling")
                    break
            
            # WebSocket tasks are already running concurrently
                    
            # 0.5 means half a second
            await asyncio.sleep(0.5)
        # await websocket.close(code=1008, reason="should close but not closed")
        await try_close(websocket,1008,"should close but not closed")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        try:
            await safe_send(websocket,{"error": str(e)},lock=send_lock)
        except:
            pass
        # await websocket.close(code=1008, reason="2D plot but exception")
        await try_close(websocket,1008,"2D plot but exception")
        return
    finally:
        # hb_task.cancel()
        try:
            sender_task.cancel()
            receiver_task.cancel()
            with contextlib.suppress(asyncio.CancelledError): await receiver_task
        except Exception:
            pass
        # if(websocket.application_state==WebSocketState.CONNECTED):
        #     await websocket.close(code=1000, reason="normal close for 2D plot visuallization")
        await try_close(websocket,1000,"normal close for 2D plot visuallization")
        return

"""The new version related with using Websocket
"""
@app.websocket("/ws/vis2d/{session_id}")
async def websocket_plot_stream(websocket: WebSocket, session_id: str):
    print(f"[WebSocket] Connected for session {session_id}")
    # await websocket.accept()

    # hb_task = None
    if session_id not in active_sessions:
        # await websocket.accept()
        # await safe_send(websocket,{"type": "error","error": "Session not found"},lock=asyncio.Lock())
        # await websocket.close(code=1008, reason="2D plot but no valid session_id")
        await try_close(websocket,1008,"2D plot but no valid session_id")
        return

    origin = websocket.headers.get("origin")
    if origin not in origins:
        # await websocket.accept()
        # await safe_send(websocket,{"type": "error", "error": "Origin not allowed", "origin": origin},lock=asyncio.Lock())
        # await websocket.close(code=1008, reason="2D plot but no origin")
        await try_close(websocket,1008,"2D plot but no origin")
        return
    
    # [NECESSARY]
    await websocket.accept()
    
    session = active_sessions.get(session_id)
    # send_lock = get_ws_lock(session,"vis2d")
    send_lock = asyncio.Lock()
    
    for _ in range(60):  # up to ~6s with 0.1s sleeps
        session = active_sessions.get(session_id)
        if not session:
            break
        environment = session.get("env")
        stop_flag = session.get("stop_flag")
        if environment is not None and stop_flag is not None and "progress_log" in session:
            break
        await safe_send(websocket, {"type":"not ready"}, lock=send_lock)
        await asyncio.sleep(0.1)
    
    session = active_sessions.get(session_id)
    if not session:
        await safe_send(websocket,{"type":"error","error":"not ready"},lock=send_lock)
        # await websocket.close(code=1011, reason="not ready for 2d visualization")
        await try_close(websocket,1011,"not ready for 2d visualization")
        return

    # currently we move this block to here
    env = session.get("env")
    stop_flag=session.get("stop_flag")
    if env is None or stop_flag is None:
        # this helps to keep use alive but not open the socket
        await safe_send(websocket,{"type":"not ready"},lock=send_lock)
        return
    # and we also add this ready sent thing here
    await safe_send(websocket, {"type": "ready"}, lock=send_lock)
    
    ready_event=asyncio.Event()
    receiver_task = asyncio.create_task(ws_receiver(websocket, send_lock, ready_event, session, "2d_subs"))
    # hb_task = asyncio.create_task(heartbeat(websocket, interval=15.0))
    try:
        # previously use "queue.Queue"
        frame_q: asyncio.Queue = session.get("frame_q")
        if frame_q is None:
            await try_close(websocket,1011,"[WS] 2d visualization queue is missing")
            return
        sender_task = asyncio.create_task(drain_queue_to_ws(websocket, send_lock,frame_q))
        
        # WE REAPLACED the ABOVE VERSION for a better one below:
        # try:
        #     done, pending = await asyncio.wait(
        #         {asyncio.create_task(websocket.receive_json())},
        #         timeout=8.0,
        #         return_when=asyncio.FIRST_COMPLETED,
        #     )
        #     for t in pending:
        #         t.cancel()
        #     if done:
        #         msg = list(done)[0].result()
        #         if msg.get("type") != "ready":
        #             # Not a fatal error; just log and continue
        #             print(f"[vis2d] Unexpected first message: {msg}")
        # except Exception as e:
        #     # Don't close here; proceed to readiness wait
        #     print(f"[vis2d] Did not receive 'ready' in time: {e}")
        
        try:
            await asyncio.wait_for(ready_event.wait(),timeout=8.0)
        except asyncio.TimeoutError:
            print("[WS] 2d plot didn't receive anything")
            # await try_close(websocket,1000,"timeout and not ready from the client")
            # receiver_task.cancel()
            # with contextlib.suppress(asyncio.CancelledError): await receiver_task
            # return
            await safe_send(websocket, {"type":"not ready"}, lock = send_lock)
        
        # original check block
        
        while websocket.application_state is WebSocketState.CONNECTED and not env.done and not stop_flag():
            try:
                #(debug)
                print("[DEBUG] env.done =", env.done)
                # print("[DEBUG] progress =", session.get("progress_log"))
                
                # Update environment state and render image
                # progress=session["progress_log"]
                progress = session.get("progress_log") or {}
                
                #get the parameters for render function inside environment class
                episode=progress["episode"]
                current_progress=progress["progress"] #this is the progress counted in percentage(the number beside '%')
                total_reward=progress["reward"]
                total_time=progress["time"]
                
                # current_delivered=session["progress_log"]["delivered"]
                current_delivered=env.count_delivered()
                return4web=True
                
                if(session["subs"].get("2d_subs",0)>0):
                    img_base64 = env.render(
                        mode="human",
                        close=False,
                        reward=total_reward,
                        count_time=total_time,
                        delivered=current_delivered,
                        ThreeD_vis=False,
                        return64web=return4web,
                        web_plt=plt
                    )
                    
                    if img_base64:
                        # currently tries to make the images load as fast as possible:
                        # sent=await safe_send(websocket,{"image_base64": img_base64},lock=send_lock)
                        await put_drop(frame_q, {"type":"render","image_base64":img_base64,"ts":time.time()})
                        print(f"[WS plot] sent image len={len(img_base64) if img_base64 else 0}", flush=True)
                        
                        # Yield control to allow WebSocket tasks to process the queue
                        await asyncio.sleep(0)
                else:
                    await asyncio.sleep(0.2)
                
            except WebSocketDisconnect:
                print("Client disconnected (image update)")
                break
            except Exception as e:
                print(f"Render/WebSocket error: {e}")
                try:
                    await safe_send(websocket,{"error": str(e)},lock=send_lock)
                    # await websocket.send_text(json.dumps({"error": str(e)}))
                except:
                    print("Client disconnected during error handling")
                    break
            
            # WebSocket tasks are already running concurrently
                    
            # Send new frame every half second
            await asyncio.sleep(0.5)
        # await websocket.close(code=1000, reason="plot completed for now")
        await try_close(websocket,1000,"plot completed for now")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        try:
            await safe_send(websocket,{"error": str(e)},lock=send_lock)
            # await websocket.send_text(json.dumps({"error": str(e)}))
        except:
            pass
        await websocket.close(code=1008, reason="2D plot but exception")
    finally:
        # hb_task.cancel()
        try:
            sender_task.cancel()
            receiver_task.cancel()
            with contextlib.suppress(asyncio.CancelledError): await receiver_task
        except Exception:
            pass
        # if(websocket.application_state==WebSocketState.CONNECTED):
        #     await websocket.close(code=1000, reason="normal close for 2D plot visuallization")
        await try_close(websocket,1000,"normal close for 2D plot visuallization")
        return

# WebSocket for real-time updates
@app.websocket("/ws/training/{session_id}") #note that in the routing, paths like "../training/" are just custom routing
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    print(f"WebSocket connection attempt for session_id={session_id}")
    
    # Accept the WebSocket connection first
    # await websocket.accept()
    print(f"WebSocket connection accepted for session_id={session_id}")

    # hb_task = None
    if session_id not in active_sessions:
        await try_close(websocket,1008,"websocket training endpoint but no valid session_id")
        return
    
    origin = websocket.headers.get("origin")
    if origin not in origins:
        await try_close(websocket, 1008, "training WS: origin not allowed")
        return
    
    # [NECESSARY]
    await websocket.accept()
    session = active_sessions.get(session_id)
    # send_lock = get_ws_lock(session,"training")
    # await safe_send(websocket, {"type": "hello", "session_id": session_id},lock=send_lock)
    send_lock = asyncio.Lock()
    ready_event = asyncio.Event()
    log.info("[WS TRAINING] Ready event created, about to create tasks")
    
    # heartbeat
    # previously use "queue.Queue"
    progress_q: asyncio.Queue = session.get("progress_q")
    log.info(f"[WS TRAINING] progress_q reference: {progress_q}")
    log.info(f"[WS TRAINING] progress_q id: {id(progress_q)}")
    
    sender_task = asyncio.create_task(drain_queue_to_ws(websocket, send_lock, progress_q))
    log.info("[WS TRAINING] Sender task (drain_queue_to_ws) created and started")
    receiver_task = asyncio.create_task(ws_receiver(websocket, send_lock, ready_event, session,"training"))
    log.info("[WS TRAINING] ws_receiver task created")
    log.info(f"[WS TRAINING] Both tasks created - sender_task: {sender_task}, receiver_task: {receiver_task}")
    try:
        try:
            log.info("[WS TRAINING] Waiting for ready event")
            await asyncio.wait_for(ready_event.wait(),timeout=8.0)
            log.info("[WS TRAINING] Ready event received")
        except asyncio.TimeoutError:
            print("[WS] the websocket endpoint did not receive anything")
            log.info("[WS TRAINING] Timeout error when waiting for ready event")
            # await try_close(websocket,1000,"timeout and not ready from the client")
            # receiver_task.cancel()
            # with contextlib.suppress(asyncio.CancelledError): await receiver_task
            # return
            await safe_send(websocket, {"type":"not ready"}, lock = send_lock)
        # except WebSocketDisconnect as e: #we would consider the disconnect case as well
        #     print(f"[TRAINING-WS] client disconnected before READY (code={e.code})")
        #     return
        
        event_loop=asyncio.get_running_loop()
        # I add this loop to the session as well
        active_sessions[session_id]["loop"]=event_loop
        
        # Start training in a separate thread
        def run_training():
            try:
                agent = session["agent"]
                env = session["env"]
                config = session["config"]
                session["status"] = "training"
                
                def progress_callback(update):#progress is the dictionary passed into the parameter field
                    log.info("[Training success step] progress_callback called")
                    print("[Training success step] progress_callback called")
                    #save this in the "session" so that we can use it in matplotlib
                    session["progress_log"]=update.copy()
                    
                    # Send progress update
                    current_status = "render" if "image" in update else "training_progress"
                    # asyncio.run_coroutine_threadsafe(
                    #     safe_send(
                    #         websocket,
                    #         {
                    #         "type": current_status,
                    #         **update
                    #     },lock=send_lock),
                    #     event_loop
                    # )
                    try:
                        # await put_drop(progress_q,{"type":current_status, **update})
                        asyncio.run_coroutine_threadsafe(
                            put_drop(progress_q,{"type":current_status, **update}),
                            event_loop
                        )
                        log.info("[Training success step] progress_q enqueued")
                        log.info(f"[progress_callback] progress_q reference: {progress_q}")
                        log.info(f"[progress_callback] progress_q id: {id(progress_q)}")
                        
                        print(f"[Training success step] progress_q enqueued {current_status} with {update}")
                    except Exception as e:
                        log.warning("[Training Start] progress_q enqueue failed: %s",e)
                
                if(config.episodes):
                    # Modified training loop to send updates
                    max_episodes = config.episodes

                # Currently we move these below stop flag to the top(before try block):
                # if(session.get("stop_flag") is None):
                #     stop_flag=StopFlag()
                #     active_sessions[session_id]["stop_flag"]=stop_flag
                # stop_flag=active_sessions[session_id].get("stop_flag")
                stop_flag = active_sessions[session_id]["stop_flag"]
                
                print("[DEBUG] Entered run_training() and about to call agent.train")
                
                print("[DEBUG] run_training starting with episodes:", max_episodes)
                print("[DEBUG] stop_flag initially set:", stop_flag)
                
                # Memory monitoring is already done above during agent creation

                try:
                    log.info("[TRAINING START] About to start agent.train()...")
                    agent.train( max_episodes, progress_callback, stop_flag)
                    log.info("[TRAINING START] agent.train() completed successfully")
                except Exception as training_error:
                    log.error(f"[TRAINING START] Error during agent.train(): {training_error}")
                    raise training_error
                
                session["status"] = "completed"
                asyncio.run_coroutine_threadsafe(
                    safe_send(
                        websocket,
                        {
                        "type": "training_complete",
                        "message": "Training completed successfully"
                    },lock=send_lock),
                    event_loop
                )
                
            except Exception as e:
                session["status"] = "error"
                asyncio.run_coroutine_threadsafe(
                    safe_send(websocket,{
                        "type": "error",
                        "message": f"Training error: {str(e)}"
                    },lock=send_lock),
                    event_loop
                )
                traceback.print_exc()
            finally:
                print("[TRAINING] thread existing; status of this session is ",session.get("status"))
        
        # Start training thread
        if session.get("training_thread") is None or not session["training_thread"].is_alive():
            log.info("[TRAINING START] Creating and starting training thread...")
            training_thread = threading.Thread(target=run_training, daemon=True)
            session["training_thread"] = training_thread
            training_thread.start()
            log.info(f"[TRAINING START] Training thread started: {training_thread.name}")
            
            # Wait a moment to ensure thread started
            await asyncio.sleep(0.1)
            if training_thread.is_alive():
                log.info("[TRAINING START] Training thread confirmed alive")
            else:
                log.error("[TRAINING START] Training thread failed to start")
        else:
            log.warning("[TRAINING START] Training thread already exists and is alive")
        
        # hb_task = asyncio.create_task(heartbeat(websocket, interval=15.0))
        
        # Keep WebSocket alive and handle messages
        # while True:
        #     try:
        #         data = await websocket.receive_text()
        #         message = json.loads(data)
                
        #         if message.get("type") == "stop":
        #             session["stop_requested"] = True
        #             break
                    
        #     except WebSocketDisconnect:
        #         session["stop_requested"] = True
        #         break
        
        try:
            await asyncio.wait_for(ready_event.wait(),timeout=8.0)
        except asyncio.TimeoutError:
            print("[WS] the websocket endpoint did not receive anything")
            await safe_send(websocket, {"type":"not ready"}, lock = send_lock)
        except WebSocketDisconnect as e: #we would consider the disconnect case as well
            print(f"[TRAINING-WS] client disconnected before READY (code={e.code})")
            # return
                
        while (
                websocket.application_state is WebSocketState.CONNECTED
                and session.get("training_thread") is not None
                and session["training_thread"].is_alive()
                and not session.get("stop_requested")
            ):
                await asyncio.sleep(0.25)
        # try:
        #     await asyncio.gather(
        #         check_loop(websocket, session, sender_task, receiver_task, send_lock),
        #         sender_task,
        #         receiver_task,
        #         return_exceptions=True
        #     )
        # except Exception as e:
        #     log.info(f"[TRAINING-WS] error in check_loop: {e}")
            # await websocket.send_text(json.dumps({"error": str(e)}))
    except Exception as e:
        await safe_send(websocket,{"error": str(e)},lock=send_lock)
        # await websocket.send_text(json.dumps({"error": str(e)}))
    finally:
        # if hb_task:
        #     hb_task.cancel()
        try:
            sender_task.cancel()
            receiver_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receiver_task
                await sender_task
        except Exception:
            pass
        
        # if session_id in active_sessions:
        #     active_sessions[session_id]["stop_requested"] = True
            
        # Log exactly why we're exiting
        exit_snapshot = {
            "where": "training_ws_exit",
            "connected": getattr(websocket, "application_state", None),
            "has_thread": session.get("training_thread") is not None,
            "thread_alive": (session.get("training_thread").is_alive() if session.get("training_thread") else None),
            "stop_requested": session.get("stop_requested"),
            "status": session.get("status"),
            "sender_task_done": sender_task.done() if 'sender_task' in locals() else "not_created",
            "receiver_task_done": receiver_task.done() if 'receiver_task' in locals() else "not_created",
        }
        log.info(f"[WS TRAINING] Exit snapshot: {exit_snapshot}")
        print(exit_snapshot)
        
        # Clean up session when WebSocket closes
        if session_id in active_sessions:
            log.info(f"[CLEANUP] WebSocket closing - removing session {session_id} from active_sessions")
            del active_sessions[session_id]
            log.info(f"[CLEANUP] Remaining active sessions: {len(active_sessions)}")
            
        # if(websocket.application_state==WebSocketState.CONNECTED):
        #     await websocket.close(code=1000, reason="normal close for websocket endpoint(the training process)")
        await try_close(websocket,1000,"normal close for websocket endpoint(the training process)")
        return


# @app.get("/favicon.ico", include_in_schema=False)
# async def favicon():
#     return FileResponse("static/favicon.ico")

# app.mount("/", StaticFiles(directory="Web_App/FrontEnd/out", html=True), name="frontend")

if __name__ == "__main__":
    print("Starting RL Project API server...")
    port = int(os.environ.get("PORT", 8080))
    print("Your existing RL modules have been integrated!")
    print("Frontend should be available at: http://localhost:3000")
    print("API server will be available at: http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=port)