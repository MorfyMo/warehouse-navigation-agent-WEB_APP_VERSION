# Web_App/api_server.py
import sys
import os
import asyncio
import base64
import io
from typing import Dict, Any, Optional
import threading
import time
import ast
from pydantic import BaseModel
from datetime import datetime


# Add your project root to Python path so we can import your modules
project_root=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Now we can import your existing modules
# TODO: import statements
#import the previous main module
from main_module import *

# from DQN_test.your_dqn_file import DQN related Class
from DQN_test.DQN_module import Agent as DQNAgent, DQN_args_parser, writer as dqn_writer
dqn_args,dqn_parser=DQN_args_parser()

# from PPO_test.your_ppo_file import PPO related Class
from PPO_test.PPO_module import Agent as PPOAgent, PPO_args_parser, writer as ppo_writer
ppo_args,ppo_parser=PPO_args_parser()

# from env.your_env_file import Environment related Class
from Env.env_module import Environment
import matplotlib
matplotlib.use('Agg') #non_interactive backend for web
import matplotlib.pyplot as plt

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, APIRouter, Request
from fastapi import Body, Query
from fastapi.middleware.cors import CORSMiddleware

from starlette.websockets import WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import json
import copy

app = FastAPI(title="Warehouse Navigation API")

# Enable CORS for the frontend
#this creates the FastAPI application(application initialization)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



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
    
class LayoutRequest(BaseModel):
    # session_id: str
    layout: str

class Comment(BaseModel):
    # session_id: str
    message: str
    username: str
    time: datetime = datetime.utcnow()

# Basic endpoints
@app.get("/")
async def root():
    return {"message": "RL Project API is running!", "status": "connected"}

@app.get("/api/test")
async def test_connection():
    return {"status": "connected", "message": "Frontend can reach backend"}

@app.post("/api/training/start")
#config: the input argument that holdes training settings(i.e. algorithms, episodes, etc.)
async def start_training(config: TrainingConfig) -> TrainingResponse:
    try:
        # Generate session ID
        session_id = f"session_{int(time.time())}"
        
        # Create environment based on config
        if config.environment == "warehouse":
            total_packages = config.warehouse_config.get("total_packages") if config.warehouse_config else 146
            if(layoutModified["status"]=="layout_modified"):
                updated_layout=layoutModified["layout"]
                print("Modifed Layout in Start Training!")
                env = Environment(layout=updated_layout, done_standard=total_packages)
            else:#if the layout has not been modified
                env = Environment(done_standard=total_packages)
        else:#navigation environment(just in case if there is another option)
            # For navigation environment, use default settings
            env = Environment(done_standard=146)  # we don't alter the goal right now
        
        # Create agent based on algorithm
        if config.algorithm == "dqn":
            # Update DQN args with user config(only when there are really input from the users)
            if config.learning_rate:
                dqn_args.learning_rate = config.learning_rate
            #if not, we just use the default value
            agent = DQNAgent(dqn_args, env, dqn_writer, False, False,True)  # ThreeD=False for web
        elif config.algorithm == "ppo":
            # Update PPO args with user config(if there really exist user input)
            if config.learning_rate:
                ppo_args.actor_lr = config.learning_rate
                ppo_args.critic_lr = config.learning_rate * 2
            #if not, we just use the default value from PPO module
            agent = PPOAgent(ppo_args, env, ppo_writer, False,True)  # ThreeD=False for web
        else:
            raise ValueError(f"Unknown algorithm: {config.algorithm}")
        
        # Store session info
        active_sessions[session_id] = {
            "config": config,
            "env": env,
            "agent": agent,
            # "total_episodes": 1000,
            "status": "ready",
            "progress": 0,
            "current_episode": 0,
            "training_thread": None,
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
        
        return TrainingResponse(
            success=True,
            session_id=session_id,
            message=f"Training session created with {config.algorithm.upper()} algorithm"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

# this function is intended to show whether the environment has been initiallzed(whether we have decided "start_training")
@app.get("/api/env_init/{session_id}")
async def env_init(session_id: str):
    session=active_sessions.get(session_id)
    return {"is_ready":session is not None and session.get("env") is not None}

@app.post("/api/training/stop/{session_id}")
async def stop_training(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    session["status"] = "stopped"
    
    stop_flag=active_sessions[session_id].get("stop_flag")
    if(stop_flag):
        stop_flag.stop()
    
    # Stop training thread if running
    if session.get("training_thread") and session["training_thread"].is_alive():
        session["stop_requested"] = True
    
    return {"success": True, "message": "Training stopped"}

#this is intended for getting stop_status
def stop_status():
    session=active_sessions["active_session"]
    if(session["status"]=="stopped"):
        return True
    return False

@app.get("/api/training/status/{session_id}")
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

# #this function is for storing episode from frontend to backend
# @app.post("/api/session/{session_id}/set-episode")
# async def set_render_episode(session_id: str, data: dict):
#     if session_id not in active_sessions:
#         raise HTTPException(status_code=404, detail="Session not found")

#     max_episode = data.get("selected_episode")
#     if max_episode is None:
#         raise HTTPException(status_code=400, detail="Missing episode")

#     # Set the episode in your environment/session
#     active_sessions[session_id]["total_episodes"] = max_episode
#     return {"status": "ok"}

#this function is used to received the altered layout
@app.post("/api/layout_modification")
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

@app.websocket("/ws/layout/{session_id}")
async def websocket_layout(websocket: WebSocket, session_id:str):
    print("[WS ROUTE] Entered layout websocket route")
    await websocket.accept()
    
    session = active_sessions.get(session_id)
    if not session:
        await websocket.close(code=1008)
        return

    env = session.get("env")
    stop_flag = session.get("stop_flag")

    if env is None or stop_flag is None:
        await websocket.close(code=1008)
        return

    try:
        # Optional: Wait for frontend to say "ready"
        try:
            msg = await websocket.receive_json()
            if msg.get("type") != "ready":
                await websocket.close(code=1003)
                return
        except Exception as e:
            print(f"[Layout WS] Did not receive 'ready': {e}")
            await websocket.close()
            return

        # Main loop: stream layout until training is done
        while not env.done and not stop_flag():
            print("[Layout WS] Sending layout...")
            await env.stream_layout(websocket)
            print("[Layout WS] Layout sent.")
            await asyncio.sleep(0.2)  # Adjust speed if needed
    except WebSocketDisconnect:
        print("[Layout WS] Disconnected.")
    finally:
        await websocket.close()


@app.websocket("/ws/plot3d/{session_id}")
async def metrics3d(websocket: WebSocket, session_id:str):
    print(f"[BACKEND] WebSocket layout route hit: session_id = {session_id}")
    await websocket.accept()
    await asyncio.sleep(0.01)
    # this line is intended to check whether the sending thing works here
    print("[BACKEND] WebSocket accepted")

    session = active_sessions.get(session_id)
    env = session.get("env") if session else None

    if env is None:
        await asyncio.sleep(0.01)
        await websocket.send_text("ERROR: Env not initialized.")
        await websocket.close()
        return
    
    env=session["env"]
    # stop_flag=active_sessions[session_id].get("stop_flag")
    stop_flag = session.get("stop_flag")
    
    try:
        # await env.stream_layout(websocket)
        while not env.done and not stop_flag():
            try:
                await asyncio.sleep(0.01)
                # await env.stream_layout(websocket)
                progress=session["progress_log"]
                # delivered_package=env.count_delivered()
                # await env.stream_layout(websocket)
                delivered_package=env.count_delivered()
                
                await asyncio.sleep(0.01)
                #let's assume we have successfully did the visulization
                await websocket.send_text(json.dumps({"type":"render","stopped":stop_flag(),**active_sessions[session_id]["progress_log"],"number_delivered":delivered_package}))
            
            except WebSocketDisconnect:
                print("Client disconnected")
                break
            except Exception as e:
                print(f"3D Render/WEbSocket error:{e}")
                try:
                    await asyncio.sleep(0.01)
                    await websocket.send_text(json.dumps({"error": str(e)}))
                except:
                    print("Client disconnected during error handling")
                    break
            await asyncio.sleep(0.2)  # 5 updates/sec, adjustable
        # if(not env.done and stop_flag()):
        #     await websocket.send_text(json.dumps({"stopped":stop_flag(),**active_sessions[session_id]["progress_log"],"number_delivered":delivered_package}))
    except Exception as e:
        print(f"Unexpected error: {e}")
        try:
            await asyncio.sleep(0.01)
            await websocket.send_text(json.dumps({"error": str(e)}))
        except:
            pass
        await websocket.close()
    

#this is intended for creating a comment area
@app.post("/api/add_msg/")
async def AddComment(comment: Comment):
    # if comment.username not in messages:
    #     messages[comment.username]=[] #this means to create a section to store all the messages related with this session inside the message
    messages.append(comment)
    return {"status":"success"}

# this is intended to get all the messages from a specific session
@app.get("/api/msg/")
async def GetComment():
    # try:
    #     return messages.get(user,[])
    # except:
    #     raise Exception(f"404:Comments invalid!")
    return messages

#currently we don't need this so much - this is just for returning a static snapshot(not for continuous update)
# @app.get("/api/matplotlib/{session_id}")
@app.get("/api/plots/{plot_type}/{session_id}")
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

"""The new version related with using Websocket
"""
@app.websocket("/ws/plot/{session_id}")
async def websocket_plot_stream(websocket: WebSocket, session_id: str):
    # await websocket.accept()
    print(f"[WebSocket] Connected for session {session_id}")

    if session_id not in active_sessions:
        await websocket.accept()
        await websocket.send_text(json.dumps({"error": "Session not found"}))
        await websocket.close()
        return

    await websocket.accept()
    session = active_sessions[session_id]
    env = session["env"]
    
    stop_flag=active_sessions[session_id].get("stop_flag")
            
    try:
        while not env.done and not stop_flag():
            try:
                #(debug)
                print("[DEBUG] env.done =", env.done)
                # print("[DEBUG] progress =", session.get("progress_log"))
                
                # Update environment state and render image
                progress=session["progress_log"]
                
                #get the parameters for render function inside environment class
                episode=progress["episode"]
                current_progress=progress["progress"] #this is the progress counted in percentage(the number beside '%')
                total_reward=progress["reward"]
                total_time=progress["time"]
                
                # current_delivered=session["progress_log"]["delivered"]
                current_delivered=env.count_delivered()
                return4web=True
                
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
                    await websocket.send_text(json.dumps({"type":"render","stopped":stop_flag(),**active_sessions[session_id]["progress_log"]}))
                    await websocket.send_text(json.dumps({"image_base64": img_base64}))
                
            except WebSocketDisconnect:
                print("Client disconnected (image update)")
                break
            except Exception as e:
                print(f"Render/WebSocket error: {e}")
                try:
                    await websocket.send_text(json.dumps({"error": str(e)}))
                except:
                    print("Client disconnected during error handling")
                    break
                    
            await asyncio.sleep(0.5)  # Send new frame every half second
        if(not env.done and stop_flag()):
            print("🔥🔥🔥 Did not enter the loop: env not done but stopped")
            await websocket.send_text(json.dumps({"stopped":stop_flag(),**active_sessions[session_id]["progress_log"]}))
    except Exception as e:
        print(f"Unexpected error: {e}")
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except:
            pass
        await websocket.close()

# #this is for 3D version render plot
# @app.websocket("/ws/layout")
# async def layout_ws(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         while True:
#             if layoutRaw.get("layout"):  # assume this dict holds the latest layout
#                 await websocket.send_json({"layout": layoutRaw["layout"]})
#             await asyncio.sleep(1)  # adjust based on how often layout changes
#     except Exception as e:
#         print("WebSocket layout error:", str(e))

# WebSocket for real-time updates
@app.websocket("/ws/training/{session_id}") #note that in the routing, paths like "../training/" are just custom routing
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    print(f"WebSocket connection attempt for session_id={session_id}")
    await websocket.accept()
    
    if session_id not in active_sessions:
        await websocket.send_text(json.dumps({"error": "Session not found"}))
        return
    
    session = active_sessions[session_id]
    
    try:
        event_loop=asyncio.get_running_loop()
        
        # Start training in a separate thread
        def run_training():
            try:
                agent = session["agent"]
                env = session["env"]
                config = session["config"]
                session["status"] = "training"

                # stop_flag=StopFlag()
                # active_sessions[session_id]["stop_flag"]=stop_flag
                
                def progress_callback(update):#progress is the dictionary passed into the parameter field
                    #save this in the "session" so that we can use it in matplotlib
                    session["progress_log"]=update.copy()
                    # print("[Callback] update =", update)
                    
                    # Send progress update
                    current_status = "render" if "image" in update else "training_progress"
                    asyncio.run_coroutine_threadsafe(
                        websocket.send_text(json.dumps({
                            "type": current_status,
                            # "stopped":stop_flag(),
                            **update  #the "**update" here means unpack the dictionary passed in the parameter field
                        })),
                        event_loop
                    )
                
                if(config.episodes):
                    # Modified training loop to send updates
                    max_episodes = config.episodes
                    
                stop_flag=StopFlag()
                active_sessions[session_id]["stop_flag"]=stop_flag
                
                print("[DEBUG] Entered run_training() and about to call agent.train")
                # total_episodes = session.get("total_episodes", 1000) #just for safety purpose, we add the default value in this get function as well
                # #since we allow the user to select the episode they want, we need to get and store the episode
                agent.train(max_episodes,progress_callback,stop_flag)
                #since inside this function, we have operation like "progress_callback({...})",
                #this means that we have pass the dictionary "{...}" into the parameter "progress"
                
                session["status"] = "completed"
                asyncio.run_coroutine_threadsafe(
                    websocket.send_text(json.dumps({
                        "type": "training_complete",
                        "message": "Training completed successfully"
                    })),
                    event_loop
                )
                
            except Exception as e:
                session["status"] = "error"
                asyncio.run_coroutine_threadsafe(
                    websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Training error: {str(e)}"
                    })),
                    event_loop
                )
        
        # Start training thread
        training_thread = threading.Thread(target=run_training)
        session["training_thread"] = training_thread
        training_thread.start()
        
        # Keep WebSocket alive and handle messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "stop":
                    session["stop_requested"] = True
                    break
                    
            except WebSocketDisconnect:
                session["stop_requested"] = True
                break
                
    except Exception as e:
        await websocket.send_text(json.dumps({"error": str(e)}))
    finally:
        if session_id in active_sessions:
            active_sessions[session_id]["stop_requested"] = True

if __name__ == "__main__":
    print("Starting RL Project API server...")
    print("Your existing RL modules have been integrated!")
    print("Frontend should be available at: http://localhost:3000")
    print("API server will be available at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)