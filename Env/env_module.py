""""
Necessary classes:
1. Environment: define a distinct environment for the agent to interact with
2. __main__: the main function that (1)creates environment, (2)creates Agents that interact with the envrionment, (3)TRAIN the agent
"""
from fastapi import WebSocket, WebSocketDisconnect
import tensorflow as tf
import sys
import os
# from tensorflow.keras import keras # type: ignore
from tensorflow.keras.layers import Input, Dense, InputLayer # type: ignore
from Env import BFS_map,env_get # type: ignore
from Env.env_get import Get
from Env.BFS_map import BFS # type: ignore
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gymnasium as gym # type: ignore
import copy
import time
import math
from Env import Visualization
# import Visualization # type: ignore
from Env.Visualization import Visualize # type: ignore
# from mpl_toolkits.mplot3d import Axes3D # type: ignore

#Now we first initialize the color code for visualization
empty = gray = 0
wall = black = 1 #this also includes shelves and all other obstacles
agent = red = 2
package = brown = 3
receiving = yellow = 4
control = 5
transfer = sky = 6
updated_receiving = 7
trans_agent = blue = 8
recei_agent = sea = 9
# transfer means the area outside of the receiving region that would allow other characters to come and take the packages from receiving region

#use RGB intensity values
color_map = {
    "gray":[187,187,187],
    "black":[28,28,28],
    "red":[253,37,31],
    "brown":[174,145,61],
    "yellow":[229,217,155],
    "sky":[155,170,191],
    "blue":[4,156,216],
    "sea":[25,25,166]
}

int_color = {
    0:"gray",
    1:"black",
    2:"red",
    3:"brown",
    4:"yellow",
    5:"black", #yes we want the control room settling points to be wall as well
    6:"sky",
    7:"yellow",
    8:"blue",
    9:"sea"
}

#action
noop=0
up=1
down=2
left=3
right=4
p_left=5
p_right=6

#Block 1: the envrionment we defined by ourselves
class Environment(gym.Env):
    #This function intends to:
    # 1) initialize the layout string
    # 2)(1.2) alter the layout to array of number[the initial state] that can be transformed to pixal(and create copy)
    # 3) create the Action Space(action space of numbers & action space of actions(names) & action space of position change) and Observation Space
    # 4) define the image attribute(width, height, RGB channels[like how many colors involved in constructing a color element])
    # 5) define the render mode(internally Gym would check things like "if "human" in env.metadata.det("render.modes",[])")
    # 5) get the state(for the initial state)
    #done_standard: the number of packages we expect to pick and move to the receiving region
    #* the total number of packages in warehouse is 146, so default is 146;
    #* however, since it is always difficult to achieve that with only single number of agents, we can alter that
    def __init__(self,layout=None,done_standard=146):
        if(layout is None):
            self.layout=(
            "1111111111111111111111111111111111111111111111111111111111111111111111\n"
            "1000002000000000000000000000008000000000000000000000000000000000000001\n"
            "1000000000000000000000000000000000000000000000000000000000000000000001\n"
            "1000311300033333333333333333333333333333333000051111111111111111150001\n"
            "1000311300011111111111111111111111111111111000011000000000000000110001\n"
            "1000311300011111111111111111111111111111111000011000000000000000110001\n"
            "1000311300033333333333333333333333333333333000011000000000000011110001\n"
            "1000311300000008000000000020000000000000000000011000000000000111110001\n"
            "1000311300000000000000000000000000000000000000011000000000000111110001\n"
            "1000311300033333333333333333333333333333333000011000000000000111110001\n"
            "1000311300011111111111111111111111111111111000011000000000000011110001\n"
            "1000311300011111111111111111111111111111111000011000000000000000110001\n"
            "1000311300033333333333333333333333333333333000051100001111111111150001\n"
            "1000000000000000000000000000020000000000000000000000000000000000000001\n"
            "1000000000000000000000000000000000000000000000000000000000000000000001\n"
            "1000444444444444444444444444444400000000000000000000000000000000000001\n"
            "1000444444444444444444444444444444444444444444444444444444444440000001\n"
            "1000444444444444444444444444444444444444444444444444444444444440000001\n"
            "1111666666666666666666966666666666666666666666666666666666666661111111\n"
            )
        else:
            self.layout=layout
            print("the modified layout in our Environment:")
            print(self.layout)
        
        self.initial_layout=(
        "1111111111111111111111111111111111111111111111111111111111111111111111\n"
        "1000000000000000000000000000000000000000000000000000000000000000000001\n"
        "1000000000000000000000000000000000000000000000000000000000000000000001\n"
        "1000311300033333333333333333333333333333333000051111111111111111150001\n"
        "1000311300011111111111111111111111111111111000011000000000000000110001\n"
        "1000311300011111111111111111111111111111111000011000000000000000110001\n"
        "1000311300033333333333333333333333333333333000011000000000000011110001\n"
        "1000311300000000000000000000000000000000000000011000000000000111110001\n"
        "1000311300000000000000000000000000000000000000011000000000000111110001\n"
        "1000311300033333333333333333333333333333333000011000000000000111110001\n"
        "1000311300011111111111111111111111111111111000011000000000000011110001\n"
        "1000311300011111111111111111111111111111111000011000000000000000110001\n"
        "1000311300033333333333333333333333333333333000051100001111111111150001\n"
        "1000000000000000000000000000000000000000000000000000000000000000000001\n"
        "1000000000000000000000000000000000000000000000000000000000000000000001\n"
        "1000444444444444444444444444444400000000000000000000000000000000000001\n"
        "1000444444444444444444444444444444444444444444444444444444444440000001\n"
        "1000444444444444444444444444444444444444444444444444444444444440000001\n"
        "1111666666666666666666666666666666666666666666666666666666666661111111\n"
        )
        
        #color map & actions
        self.color_map=color_map
        self.int_color=int_color
        
        #flatten the layout before converting it
        lines=self.layout.strip().split("\n")
        #flatten the completely empty warehouse
        initial_lines=self.initial_layout.strip().split("\n")
        #[debug]check the number of columns if needed:
        # for i,line in enumerate(lines):
        #     print(f"Line {i}:{len(line)} characters")
        
        cols=len(lines[0])
        rows=len(lines)
        self.layout="".join(lines)
        #the completely empty version with no agents
        init_cols=len(initial_lines[0])
        init_rows=len(initial_lines)
        self.initial_layout="".join(initial_lines)
        
        #check if we have mismatch in layout size
        length=len(self.layout)
        assert length==rows*cols, "Mismatch in layout size!"
        #the completely empty version
        init_length=len(self.layout)
        assert init_length==init_rows*init_cols, "Mismatch in layout size!"
        
        #[debug]check if the agent exist in this layout
        # assert "2" in self.layout, "agent not in the layout!"
        # for i in self.layout:
        #     if i=="2":
        #         print(f"agent exist!")
        #         break
        
        #alter the layout to an array that can used as image pixal
        self.initial_state=np.array([int(i) for i in self.layout],dtype=np.int8)
        self.initial_state=self.initial_state.reshape(rows,cols)
        #for this we also have the completely empty warehouse one
        self.initial_empty_state=np.array([int(i) for i in self.initial_layout],dtype=np.int8)
        self.initial_empty_state=self.initial_empty_state.reshape(init_rows,init_cols)
        #copy the state for the state in the current envrionment(since we haven't started at the initiation)
        self.grid_state = copy.deepcopy(self.initial_state)
        
        #[debug]check if the layout was converted as we expect
        # print(f"layout after conversion:{self.initial_state}")
        
        #[debug]check if the agent exist after conversion
        assert agent in self.initial_state, "agent not in layout!"
        
        #construct observation space
        self.observation_space=gym.spaces.Box(low=0,high=9,shape=self.grid_state.shape,dtype=int)
        
        #automatically infer the image grid size and define image attribute
        cell_height=3
        cell_width=3
        height, width=self.grid_state.shape
        self.img_shape=[height*cell_height,width*cell_width,3]
        self.cell_height=cell_height
        self.cell_width=cell_width
        
        #define the render mode
        self.metadata = {
            "render.modes":["human"]
        }
        
        #define action space and actions
        self.action_space=gym.spaces.Discrete(7)
        self.actions=["noop","up","down","left","right","p_left","p_right"]
        #for python dictionary, always remember to add "" for strings
        #p_left & p_right have to be adjusted with specific cases(like when packages are in different condiitons relative to agent)
        self.action_pos_dict={
            "noop":[[0,0],[0,0]],
            "up":[[-1,0],[0,0]],
            "down":[[1,0],[0,0]],
            "left":[[0,-1],[0,0]],
            "right":[[0,1],[0,0]],
            "p_left":[[0,0],[1,1]],
            "p_right":[[0,0],[1,1]]
        }
        #the expectation here is that if there is no package in progress: just use the first 5;
        #if there is package in progress:
        # we just the whole 6 actions(first 5 in this case is carrying the package with the robot;
        # last 4 is just changing direction of the packages)
        
        #initialize the start states(and we also includes the package state-even if we don't have it yet)
        self.agent_state=list(self.get_state())
        self.track=Get(self.grid_state,self.initial_state,self.agent_state,self.initial_empty_state)
        self.num_agents=self.track.get_num_agents()
        #now we also have to initialize the package state of each agent
        self.package_state=list()
        """we need to prevent the case when we have several agents intending to take the package[so that's why we need lock package list]
        The format of each lock_package location should be tuple, and the large iterable that include each tuple can be a set or list.
        Note: this lock packages list is to store all the packages that has already been transferred to anther agent inside a single time step;
        (if it has already been transferred, we stop the other agents from tranferring it)
        *By preventing the other agents from tranferring the package again, we prevent the case when we transfer the same package twice in one step
        (if in one step we do two things, this causes duplication)
        """
        self.lock_packages=list()
        
        #the time of taking up packages
        #as can be seen from this below code, we are assigning the pickup distinctively for each agent(each agent only count for itself - not the "collctive intelligence")
        self.count_pickup=list()
        for i in range(self.num_agents):
            self.package_state.append(None)
            #the time of taking up packages
            self.count_pickup.append(0)
        
        #initialize the 4 elements(the ones necessary: which includes just info and done)
        #action is selected by the agent itself, while reward is determined by the action alone
        self.info=[]
        #each agent's information includes "info"
        for i in range(self.num_agents):
            Info = {
                "success":False,
                "in_progress":False,
                "just_drop":False,
                "bounce":False,
                "count_bounce":0,
                "time":0
            }
            
            #add the agent's information to the list of Agents
            self.info.append(Info)
        self.done=False
        
        #count_done is the actual delivered packages(both those that are currently on the receiving region & those already transferred away by the collector)
        self.count_done=0
        #these are the pacakges that have been transferred away from the warehouse(no longer exist anymore on the map)
        self.received=0
        
        self.level_achieved=0
        self.time=0 #in seconds
        if(done_standard>146):
            raise ValueError("Goal to large: there are only a total of 146 packages in warehouse")
        self.transferring_packages_goal=done_standard
        #this is the total number of packages expected to be carried to the receiving region
        # *(regardless of how many agents there are in the warehosue)
        #visualization
        self.plt=plt
        self.view_initialization=False
        #this stores the Visualization thing
        self.Visual_View=Visualize(self.grid_state,self.int_color,self.color_map,self.img_shape,self.plt)

    #This function (1) checks whether the start state and goal state are invalid
    #If the two states are valid, (2) we get the shape of the two states and return the two shapes
    def get_state(self):
        #first we retrieve start state and goal state
        condition= (self.initial_state==agent)|(self.initial_state==trans_agent)|(self.initial_state==recei_agent)
        start_state=np.where(condition)
        
        #then we check whether this is invalid retrieve action
        not_found=(len(start_state[0])==0)
        if not_found:
            sys.exit(
                "Start state is not present in this warehouse."
                "Check the layout."
            )
        
        return list(zip(*start_state))
    
    #set observation arrays for agent and potentially its package when updating
    #return: direct_return, agent_first, turn, next_obs, a_obs, reward
    def set_obs(self,ith_agent,agent_first,turn, agent_row_change, agent_column_change,action,reward):
        direct_return=False
        agent_first=agent_first
        
        if self.info[ith_agent]["in_progress"]==False:
            next_obs = (
                #this changes up and down
                self.agent_state[ith_agent][0]+agent_row_change,
                #this changes left and right
                self.agent_state[ith_agent][1]+agent_column_change
            )
            turn=False
            a_obs=None
        else:
            #the case when the two things can only move left & right if not turn
            if(self.agent_state[ith_agent][0]==self.package_state[ith_agent][0]): #this means on the same row
                #package on the right side(Case 4)
                if(self.package_state[ith_agent][1]>self.agent_state[ith_agent][1]):
                    #moving to the right, which means package is the first thing that directs the move
                    if(action==4):
                        a_obs=(
                            self.package_state[ith_agent][0]+agent_row_change,
                            self.package_state[ith_agent][1]+agent_column_change
                        )
                        #of course agent state should move with the package at the same magnitude
                        next_obs=(
                            self.agent_state[ith_agent][0]+agent_row_change,
                            self.agent_state[ith_agent][1]+agent_column_change
                        )
                        turn=False
                    #moving to the left, which means agent is the first thing that directs the move
                    elif(action==3):
                        agent_first=True
                        next_obs=(
                            self.agent_state[ith_agent][0]+agent_row_change,
                            self.agent_state[ith_agent][1]+agent_column_change
                        )
                        a_obs=(
                            self.package_state[ith_agent][0]+agent_row_change,
                            self.package_state[ith_agent][1]+agent_column_change
                        )
                        turn=False
                    #turning to the left, which means the package gets to the upside of the agent
                    elif(action==5):
                        a_obs=(
                            self.package_state[ith_agent][0]-1,
                            self.package_state[ith_agent][1]-1
                        )
                        next_obs=(
                            self.agent_state[ith_agent][0],
                            self.agent_state[ith_agent][1]
                        )
                        turn=True
                    #turning to the right, which means the package gets to downside of the agent
                    elif(action==6):
                        a_obs=(
                            self.package_state[ith_agent][0]+1,
                            self.package_state[ith_agent][1]-1
                        )
                        next_obs=(
                            self.agent_state[ith_agent][0],
                            self.agent_state[ith_agent][1]
                        )
                        turn=True
                    else:
                        direct_return =True
                        next_obs=a_obs=None
                        return direct_return, agent_first, turn, next_obs, a_obs, reward
                #package on the left side(Case 3)
                else:
                    #moving to the left, which means package is the one that directs the move
                    if(action==3):
                        a_obs=(
                            self.package_state[ith_agent][0]+agent_row_change,
                            self.package_state[ith_agent][1]+agent_column_change
                        )
                        next_obs=(
                            self.agent_state[ith_agent][0]+agent_row_change,
                            self.agent_state[ith_agent][1]+agent_column_change
                        )
                        turn=False
                    #moving to the right, which means agent the one that directs the move
                    elif(action==4):
                        agent_first=True
                        next_obs=(
                            self.agent_state[ith_agent][0]+agent_row_change,
                            self.agent_state[ith_agent][1]+agent_column_change
                        )
                        a_obs=(
                            self.package_state[ith_agent][0]+agent_row_change,
                            self.package_state[ith_agent][1]+agent_column_change
                        )
                        turn=False
                    #turing to the left, which means package gets to the downside
                    elif(action==5):
                        a_obs=(
                            self.package_state[ith_agent][0]+1,
                            self.package_state[ith_agent][1]+1
                        )
                        next_obs=(
                            self.agent_state[ith_agent][0],
                            self.agent_state[ith_agent][1]
                        )
                        turn=True
                    #turning to the right, which means package gets to the upside
                    elif(action==6):
                        a_obs=(
                            self.package_state[ith_agent][0]-1,
                            self.package_state[ith_agent][1]+1
                        )
                        next_obs=(
                            self.agent_state[ith_agent][0],
                            self.agent_state[ith_agent][1]
                        )
                        turn=True
                    else:
                        direct_return =True
                        next_obs=a_obs=None
                        return direct_return, agent_first, turn, next_obs, a_obs, reward#the rest actions are invliad here
            #the case when the two things can only move up & down if not turn
            elif(self.agent_state[ith_agent][1]==self.package_state[ith_agent][1]):
                #package on the upside(Case 1)
                if(self.package_state[ith_agent][0]<self.agent_state[ith_agent][0]):
                    #going up with packaging leading the move
                    if(action==1):
                        a_obs=(
                            self.package_state[ith_agent][0]+agent_row_change,
                            self.package_state[ith_agent][1]+agent_column_change
                        )
                        next_obs=(
                            self.agent_state[ith_agent][0]+agent_row_change,
                            self.agent_state[ith_agent][1]+agent_column_change
                        )
                        turn=False
                    #going down with agent leading the move
                    elif(action==2):
                        agent_first=True
                        next_obs=(
                            self.agent_state[ith_agent][0]+agent_row_change,
                            self.agent_state[ith_agent][1]+agent_column_change
                        )
                        a_obs=(
                            self.package_state[ith_agent][0]+agent_row_change,
                            self.package_state[ith_agent][1]+agent_column_change
                        )
                        turn=False
                    #turning left, which means package gets to the left side of the agent
                    elif(action==5):
                        a_obs=(
                            self.package_state[ith_agent][0]+1,
                            self.package_state[ith_agent][1]-1
                        )
                        next_obs=(
                            self.agent_state[ith_agent][0],
                            self.agent_state[ith_agent][1]
                        )
                        turn=True
                    #turning right, which means packages gets to the right side of the agent
                    elif(action==6):
                        a_obs=(
                            self.package_state[ith_agent][0]+1,
                            self.package_state[ith_agent][1]+1
                        )
                        next_obs=(
                            self.agent_state[ith_agent][0],
                            self.agent_state[ith_agent][1]
                        )
                        turn=True
                    else:
                        direct_return =True
                        next_obs=a_obs=None
                        return direct_return, agent_first, turn, next_obs, a_obs, reward
                #package on the downside(Case 2)
                else:
                    #going down, which means package leads the move
                    if(action==2):
                        a_obs=(
                            self.package_state[ith_agent][0]+agent_row_change,
                            self.package_state[ith_agent][1]+agent_column_change
                        )
                        next_obs=(
                            self.agent_state[ith_agent][0]+agent_row_change,
                            self.agent_state[ith_agent][1]+agent_column_change
                        )
                        turn=False
                    #going up, which means agent leads the move
                    elif(action==1):
                        agent_first=True
                        next_obs=(
                            self.agent_state[ith_agent][0]+agent_row_change,
                            self.agent_state[ith_agent][1]+agent_column_change
                        )
                        a_obs=(
                            self.package_state[ith_agent][0]+agent_row_change,
                            self.package_state[ith_agent][1]+agent_column_change
                        )
                        turn=False
                    #turning left, which means package goes from downside to the right
                    elif(action==5):
                        a_obs=(
                            self.package_state[ith_agent][0]-1,
                            self.package_state[ith_agent][1]+1
                        )
                        next_obs=(
                            self.agent_state[ith_agent][0],
                            self.agent_state[ith_agent][1]
                        )
                        turn=True
                    #turning right, which means package goes from downside to the left
                    elif(action==6):
                        a_obs=(
                            self.package_state[ith_agent][0]-1,
                            self.package_state[ith_agent][1]-1
                        )
                        next_obs=(
                            self.agent_state[ith_agent][0],
                            self.agent_state[ith_agent][1]
                        )
                        turn=True
                    else:
                        direct_return =True
                        next_obs=a_obs=None
                        return direct_return, agent_first, turn, next_obs, a_obs, reward
            #the diagonal or L-shaped case(which we need to avoid)
            else:
                direct_return =True
                next_obs=a_obs=None
                return direct_return, agent_first, turn, next_obs, a_obs, reward
        direct_return = False
        return direct_return, agent_first, turn, next_obs, a_obs, reward
    
    #compare the distance to the receiving region between the original agent state and the updated agent state
    def comp_dist(self,objective,current_state,next_obs,custom_agent=False,correspond_agent_state=None,is_receiving=False):
        num=self.track.comp_dist(self.grid_state,objective,current_state,next_obs,custom_agent=custom_agent,correspond_agent_state=correspond_agent_state,is_receiving=is_receiving)
        return num

    #count how many packages are delivered right now
    def count_delivered(self):
        delivered_number=0
        receiving_region=self.track.get_receiving_region()
        for area in receiving_region:
            state=self.grid_state[area[0],area[1]]
            if(state==package):
                delivered_number+=1
        delivered_number=delivered_number+self.received
        return delivered_number
    
    #NOTE THAT the selection of action here should consider the difference in the first 5 and the last 4
    #perform the moving action of the agent
    #the same as the Gym envrionment, step(action) returns 4 things: observation, reward, done, info
    #now this step function is specific for a certain agent's action:
    # ith_agent is the position of an agent - an element in self.info
    def step(self,ith_agent,action):
        """This check referenced from GPT: we are trying to know whether & why there is index problem"""
        if ith_agent >= self.num_agents:
            print(f"🚨 Error: ith_agent={ith_agent}, which is {self.grid_state[self.agent_state[ith_agent][0],self.agent_state[ith_agent][1]]} type agent, exceeds num_agents={self.num_agents}")
            raise IndexError("Agent index out of range")
        
        #first we initialize the current 4 elements
        action = int(action)
        self.info[ith_agent]["success"]=False
        self.info[ith_agent]["bounce"]=False
        reward = 0.0
        #to make sure our states are in list form
        self.agent_state[ith_agent]=list(self.agent_state[ith_agent])
        if self.package_state[ith_agent] is not None:
            self.package_state[ith_agent]=list(self.package_state[ith_agent])
        
        #now we want to identify the type of the agent(normal, helper, or collector)
        #first we identify whether this is helper or not
        if(self.grid_state[self.agent_state[ith_agent][0],self.agent_state[ith_agent][1]]==trans_agent):
            Trans=True
        else:
            Trans=False
        #next we identify whether this is collector or not
        if(self.grid_state[self.agent_state[ith_agent][0],self.agent_state[ith_agent][1]]==recei_agent):
            Recei=True
        else:
            Recei=False
        
        #we first get rid of the case when there is no action
        if(action==noop):
            return self.grid_state, reward, self.done, self.info[ith_agent]
        #get the action in string form key
        action_key=self.actions[action]
        
        #to make sure we don't get wrong if the agent carries package,
        #we need to specify which postion changed in specific
        agent_row_change=self.action_pos_dict[action_key][0][0]
        agent_column_change=self.action_pos_dict[action_key][0][1]
        
        agent_first=False
        turn=False
        #now we make the move and get the next observation: which is using the orignal agent state plus the action position change in the dictionary
        #when we say next_obs of agent or agent, we mean the next_obs of the whole moving object: which includes agent and the package
        direct_return, agent_first, turn, next_obs, a_obs, reward=self.set_obs(ith_agent,agent_first,turn, agent_row_change, agent_column_change,action,reward)
        if direct_return:
            return self.grid_state,reward,self.done,self.info[ith_agent]
        
        #THIS CASE DISCUSSES THE INFO(the flag)
        #Get rid of the invalid case
        #1. NEXT STATE INVALID: First check whether the position after the move is out of the range(which is invalid)
        next_state_invalid = (next_obs[0]<0 or next_obs[0]>=self.grid_state.shape[0])\
            or (next_obs[1]<0 or next_obs[1]>=self.grid_state.shape[1])
            
        if a_obs is not None:
            a_state_invalid = (a_obs[0]<0 or a_obs[0]>=self.grid_state.shape[0])\
            or (a_obs[1]<0 or a_obs[1]>=self.grid_state.shape[1])
            
            next_state_invalid = next_state_invalid or a_state_invalid
            
        #2. NEXT STATE WALL: second check whether the position after the move is getting into wall
        if a_obs is not None:
            next_wall = (tuple(next_obs) in self.track.get_wall()) or (tuple(a_obs) in self.track.get_wall())
        else:
            next_wall = (tuple(next_obs) in self.track.get_wall())
            
        #3. NEXT STATE TRASNFER: eventually check whether the position after the move is the transfer region(not in the working area)
        #Note: since we now have collector, this only works for the cases when the agent is normal agent or helpers
        if a_obs is not None:
            next_transfer_region = ((tuple(next_obs) in self.track.get_transfer_region()) or (tuple(a_obs) in self.track.get_transfer_region()) and not Recei)
        else:
            next_transfer_region = (tuple(next_obs) in self.track.get_transfer_region() and not Recei)
        
        #Note: the first line checks whether the row is out of range; the second line checks whether the column is out of range
        if next_state_invalid or (next_transfer_region and not Trans and not Recei):
            reward-=0.1
            self.info[ith_agent]["bounce"]=True
            self.info[ith_agent]["count_bounce"]+=1
            return self.grid_state, reward, self.done, self.info[ith_agent]
        elif next_wall:
            reward-=3
            self.info[ith_agent]["bounce"]=True
            if(self.info[ith_agent]["count_bounce"]<=5):
                self.info[ith_agent]["count_bounce"]+=1
            else:
                reward-=10
                self.info[ith_agent]["count_bounce"]=0
            return self.grid_state, reward, self.done, self.info[ith_agent]
        
        #initialize for the transferred package(if we have passed the package through the transfer region)
        #this is specific useful for the collector case
        is_transferred=False
        
        #THIS CASE DISCUSSES THE COMPLETETION STATE(done) & THE REWARD(reward)
        #now we discuss the reward for each special cases
        #this step prepare for the later compare distance action after all operations
        current_state=self.agent_state[ith_agent]
        #first we locate the next_state(check whether we are carrying the package)
        next_state_agent=True
        if(self.info[ith_agent]["in_progress"]==False):
            next_state = self.grid_state[next_obs[0],next_obs[1]]
        else:#if we are carrying the package
            if(agent_first): #the case when next_state would directly be the agent(package is not the leading one)
                next_state = self.grid_state[next_obs[0],next_obs[1]]
            else: #the case when next_state would be the package(not the agent)
                next_state = self.grid_state[a_obs[0],a_obs[1]]
                current_state=self.package_state[ith_agent]
                next_state_agent=False
        
        #the case when the agent's next place is empty space
        if next_state == empty:
            #Ensure the agent is NOT collector
            if(Recei):
                reward-=1
                self.info[ith_agent]["bounce"]=True
                if(self.info[ith_agent]["count_bounce"]<=5):
                    self.info[ith_agent]["count_bounce"]+=1
                else:
                    reward-=10
                    self.info[ith_agent]["count_bounce"]=0
                return self.grid_state,reward,self.done,self.info[ith_agent]
            #now we can safely discuss the other cases!
            if(self.info[ith_agent]["in_progress"]==False):
                #turn the color of the space left behind to its original color
                if(tuple(self.agent_state[ith_agent]) in self.track.get_receiving_region()):
                    self.grid_state[self.agent_state[ith_agent][0],self.agent_state[ith_agent][1]]=receiving
                else:
                    self.grid_state[self.agent_state[ith_agent][0],self.agent_state[ith_agent][1]]=empty
                
                if not Trans:
                    self.grid_state[next_obs[0],next_obs[1]]=agent
                else:
                    self.grid_state[next_obs[0],next_obs[1]]=trans_agent
                #put the agent to the next position
                self.agent_state[ith_agent]=copy.deepcopy(next_obs)
                
            #Carrying packages but we are still on the way to the receiving area
            else: #info["in_progress"]==True
                #moving in the direct way
                if(agent_first and not turn):
                    #turn the color of the space left behind to its original color
                    if(tuple(self.agent_state[ith_agent]) in self.track.get_receiving_region()):
                        self.grid_state[self.agent_state[ith_agent][0],self.agent_state[ith_agent][1]]=receiving
                    else:
                        self.grid_state[self.agent_state[ith_agent][0],self.agent_state[ith_agent][1]]=empty

                    if(tuple(self.package_state[ith_agent]) in self.track.get_receiving_region()):
                        self.grid_state[self.package_state[ith_agent][0],self.package_state[ith_agent][1]]=receiving
                    else:
                        self.grid_state[self.package_state[ith_agent][0],self.package_state[ith_agent][1]]=empty

                    if not Trans:
                        self.grid_state[next_obs[0],next_obs[1]]=agent
                    else:
                        self.grid_state[next_obs[0],next_obs[1]]=trans_agent
                    #we need to check which position changed
                    self.grid_state[a_obs[0],a_obs[1]]=package
                    
                    self.agent_state[ith_agent]=copy.deepcopy(next_obs)
                    self.package_state[ith_agent]=copy.deepcopy(a_obs)
                #the case when moving in the direct way and if the package if the leading one
                elif(not agent_first and not turn): #the sequence of setting the color of package state and agent state matters(so differ from first case)
                    #turn the color of the space left behind to its original color
                    if(tuple(self.package_state[ith_agent]) in self.track.get_receiving_region()):
                        self.grid_state[self.package_state[ith_agent][0],self.package_state[ith_agent][1]]=receiving
                    else:
                        self.grid_state[self.package_state[ith_agent][0],self.package_state[ith_agent][1]]=empty
                        
                    if(tuple(self.agent_state[ith_agent]) in self.track.get_receiving_region()):
                        self.grid_state[self.agent_state[ith_agent][0],self.agent_state[ith_agent][1]]=receiving
                    else:
                        self.grid_state[self.agent_state[ith_agent][0],self.agent_state[ith_agent][1]]=empty

                    if not Trans:
                        self.grid_state[next_obs[0],next_obs[1]]=agent
                    else:
                        self.grid_state[next_obs[0],next_obs[1]]=trans_agent
                    #we need to check which position changed
                    self.grid_state[a_obs[0],a_obs[1]]=package
                    
                    self.agent_state[ith_agent]=copy.deepcopy(next_obs)
                    self.package_state[ith_agent]=copy.deepcopy(a_obs)
                #check if this is the case when we only turn the package
                elif(turn):
                    #check where the package is on
                    if(tuple(self.package_state[ith_agent]) in self.track.get_receiving_region()):
                        self.grid_state[self.package_state[ith_agent][0],self.package_state[ith_agent][1]]=receiving
                    else:
                        self.grid_state[self.package_state[ith_agent][0],self.package_state[ith_agent][1]]=empty
                    
                    self.grid_state[a_obs[0],a_obs[1]]=package
                    self.package_state[ith_agent]=copy.deepcopy(a_obs)
                    self.agent_state[ith_agent]=copy.deepcopy(next_obs)
                    
        #just to get rid of the case when we mistakenly forget to consider the wall case before(double safecase)
        #Note: since we now have the collector, the bounce case at transfer region only works for normal agents and helpers
        elif next_state==wall or next_state==control or next_state==agent or next_state==trans_agent or next_state==recei_agent:
            if((not next_state==trans_agent) or (not Recei and next_state==recei_agent)):
            #this condition (1)prevent us from moving away or repelling the helper agents;
            # (2)also prevent the collector from avoiding the other agents
                reward -= 1
            self.info[ith_agent]["bounce"]=True
            if(self.info[ith_agent]["count_bounce"]<=5):
                self.info[ith_agent]["count_bounce"]+=1
            else:
                reward-=10
                self.info[ith_agent]["count_bounce"]=0
            return self.grid_state,reward,self.done,self.info[ith_agent]
        #the case when we have the next state would be the package
        #Note: this actually does two steps in one: first takes the package, then step out from the space that originally has the package
        elif next_state==package:
            #Now this is the case when we need package to move with the agent(which mean in the state like "[robot][package]",
            # not having package stacking above robot)
            if(self.info[ith_agent]["in_progress"]):
                reward-=0.1
                self.info[ith_agent]["bounce"]=True
                self.info[ith_agent]["count_bounce"]+=1
                return self.grid_state,reward,self.done,self.info[ith_agent]
            #if the next state is the package-while we only have the agent alone
            else: #not carrying the package in progress(not self.info[ith_agent]["in_progress"])
                #we need to make sure that we are not trying to take package already carrying by the other agents(if not making the situation any better)
                for ith in range(self.num_agents):
                    #Note: "ith" represent the agent from the iteration(not the current one), "ith_agent" is the current agent we are talking about
                    if((ith!=ith_agent) and (self.package_state[ith]==tuple(next_obs)) and self.info[ith]["in_progress"] and (self.package_state[ith] is not None) and (self.package_state[ith_agent] is None)):
                        #if this current agent is closer to the receiving area, we can transfer the original package from the other agent to this one
                        #if this is collector, we don't allow transfer to happen; and also, if the one transferring to is collector, this is not allowed as well
                        to_collector=(self.grid_state[self.agent_state[ith][0],self.agent_state[ith][1]]==recei_agent)
                        if(not Recei and not to_collector and self.comp_dist(receiving,self.agent_state[ith],self.agent_state[ith_agent])>0):#the first 2 conditions make sure we are Not & Not Transferring packages to collector
                            #we need to check if this is operating as we want(ensure the package state isn't already claimed)
                            if(self.package_state[ith] is None and (tuple(self.package_state[ith]) in self.lock_packages)):
                                continue
                            
                            #alter the next_obs in grid_state
                            if(tuple(next_obs) in self.track.get_receiving_region()):
                                self.grid_state[next_obs[0],next_obs[1]]=receiving
                            else:
                                self.grid_state[next_obs[0],next_obs[1]]=empty
                            
                            if(tuple(self.package_state[ith]) in self.track.get_receiving_region()): #now this is for altering the old package_state
                                self.grid_state[self.package_state[ith][0],self.package_state[ith][1]]=receiving
                            else:
                                self.grid_state[self.package_state[ith][0],self.package_state[ith][1]]=empty
                            
                            self.count_pickup[ith_agent]+=1
                            reward+=25*self.count_pickup[ith_agent]
                            #encourage normal agents to deem helpers as friends
                            if(not Trans and not Recei and self.grid_state[self.agent_state[ith][0],self.agent_state[ith][1]]==trans_agent):
                                self.count_pickup[ith]+=1
                                reward+=85*self.count_pickup[ith]
                            else:
                                self.count_pickup[ith]+=1
                                reward+=80*self.count_pickup[ith]
                                
                            self.package_state[ith_agent]=copy.deepcopy(self.package_state[ith])
                            self.lock_packages.append(tuple(self.package_state[ith]))
                            self.package_state[ith]=None
                            self.info[ith]["in_progress"]=False
                            self.info[ith_agent]["in_progress"]=True
                            break
                        else:#the case when the original agent(the one already carrying package is closer to the deliver area) should keep the package - this logic works fine
                            reward-=0.1
                            self.info[ith_agent]["bounce"]=True
                            self.info[ith_agent]["count_bounce"]+=1
                        return self.grid_state,reward,self.done,self.info[ith_agent]
                if(not self.info[ith_agent]["just_drop"] and (tuple(self.agent_state[ith_agent]) not in self.track.get_receiving_region()) and not Trans):
                    status=self.grid_state[self.agent_state[ith_agent][0]-agent_row_change,self.agent_state[ith_agent][1]-agent_column_change]
                    if(status==empty or status==receiving):
                        self.grid_state[self.agent_state[ith_agent][0],self.agent_state[ith_agent][1]]=package
                        self.grid_state[self.agent_state[ith_agent][0]-agent_row_change,self.agent_state[ith_agent][1]-agent_column_change]=agent
                        #the case when simply getting the package from the shelf
                        if(tuple(next_obs) not in self.track.get_receiving_region()):
                            self.grid_state[next_obs[0],next_obs[1]]=empty
                        else:
                            #Getting the package in receiving region again after dropping with agent outside receiving region:
                            #currently we do consider this case
                            self.count_pickup[ith_agent]-=1
                            reward-=110*self.count_pickup[ith_agent]
                            self.grid_state[next_obs[0],next_obs[1]]=receiving
                        self.info[ith_agent]["in_progress"]=True
                        
                        self.package_state[ith_agent]=copy.deepcopy(self.agent_state[ith_agent])
                        self.agent_state[ith_agent][0]=self.agent_state[ith_agent][0]-agent_row_change
                        self.agent_state[ith_agent][1]=self.agent_state[ith_agent][1]-agent_column_change
                        reward+=25*self.count_pickup[ith_agent]
                    else:
                        #this goes back to the first case(with agent in progress)
                        reward-=0.1
                        self.info[ith_agent]["bounce"]=True
                        self.info[ith_agent]["count_bounce"]+=1
                        return self.grid_state,reward,self.done,self.info[ith_agent]
                elif(not Recei and not self.info[ith_agent]["just_drop"] and (tuple(self.agent_state[ith_agent]) in self.track.get_receiving_region())):
                    reward-=0.1
                    self.info[ith_agent]["bounce"]=True
                    self.info[ith_agent]["count_bounce"]+=1
                    return self.grid_state,reward,self.done,self.info[ith_agent]
                #the case when we have just dropped the package and not leaving the receiving area
                #prevent the case when we would take the package away from the receiving region
                elif(not Recei and self.info[ith_agent]["just_drop"] and tuple(self.agent_state[ith_agent]) in self.track.get_receiving_region()):
                    self.info[ith_agent]["just_drop"]=False
                    reward-=0.1
                    self.info[ith_agent]["bounce"]=True
                    self.info[ith_agent]["count_bounce"]+=1
                    return self.grid_state,reward,self.done,self.info[ith_agent]
                #this case is specifically for collector(no matter it has just dropped the package or not)
                elif(Recei and self.info[ith_agent]["in_progress"]==False and tuple(self.agent_state[ith_agent]) in self.track.get_receiving_region()):
                    status=self.grid_state[self.agent_state[ith_agent][0]-agent_row_change,self.agent_state[ith_agent][1]-agent_column_change]
                    #we need to ensure that our agent is still on the receiving region; if it gets to the empty region, we need to stop this
                    if(status==receiving or status==transfer):
                        self.info[ith_agent]["just_drop"]=False
                        #first we update the status of the current package and agent
                        self.grid_state[self.agent_state[ith_agent][0],self.agent_state[ith_agent][1]]=package
                        self.grid_state[self.agent_state[ith_agent][0]-agent_row_change,self.agent_state[ith_agent][1]-agent_column_change]=recei_agent
                    else:#this is the case when we cannot accept(i.e. going into wall or the other control region)
                        #our collector cannot be on anywhere else, so we also need to penalize that
                        reward-=1
                        self.info[ith_agent]["bounce"]=True
                        if(self.info[ith_agent]["count_bounce"]<=5):
                            self.info[ith_agent]["count_bounce"]+=1
                        else:
                            reward-=10
                            self.info[ith_agent]["count_bounce"]=0
                        return self.grid_state,reward,self.done,self.info[ith_agent]
                        
                    #now we turn the position left behind to their original colors
                    if(tuple(next_obs) in self.track.get_receiving_region()):
                        self.grid_state[next_obs[0],next_obs[1]]=receiving
                    else:#this is the case we need to concern a little bit(just keep it here and see if there is anything going wrong)
                        self.grid_state[next_obs[0],next_obs[1]]=transfer
                    self.info[ith_agent]["in_progress"]=True
                    #currently we just don't consider the case when the agent re-pickup the package from the receiving region and fool for more points of rewards
                    self.package_state[ith_agent]=copy.deepcopy(self.agent_state[ith_agent])
                    self.agent_state[ith_agent][0]=self.agent_state[ith_agent][0]-agent_row_change
                    self.agent_state[ith_agent][1]=self.agent_state[ith_agent][1]-agent_column_change
                    self.count_pickup[ith_agent]+=1
                    #the same logic as having normal agent picking up packages from the shelves
                    reward+=25*self.count_pickup[ith_agent]
                else: #just drop and the agent is outside
                    self.info[ith_agent]["just_drop"]=False
                    reward-=0.1
                    self.info[ith_agent]["bounce"]=True
                    self.info[ith_agent]["count_bounce"]+=1
                    return self.grid_state,reward,self.done,self.info[ith_agent]
            
        elif next_state==receiving and not Recei:
            #be sure the the next state is the case when the packages falls on the receiving area(2 possibilities):
            #P1: it is not about the agent falling on receiving area;instead, we have only the package falls on the receiving area
            #P2: the agent also falls on the receiving area
            if(self.info[ith_agent]["in_progress"]==True and not next_state_agent):
                #first we check whether we are arriving at the last row available inside the receiving region
                self.info[ith_agent]["success"]=True
                self.info[ith_agent]["in_progress"]=False
                self.info[ith_agent]["just_drop"]=True
                edge=False
                
                #turn the color of the space left behind to its original color
                if(tuple(self.package_state[ith_agent]) in self.track.get_receiving_region()):
                    self.grid_state[self.package_state[ith_agent][0],self.package_state[ith_agent][1]]=receiving
                else:
                    edge=True
                    self.grid_state[self.package_state[ith_agent][0],self.package_state[ith_agent][1]]=empty
    
                #turn the color of the space left behind to original color
                if(tuple(self.agent_state[ith_agent]) in self.track.get_receiving_region()):
                    self.grid_state[self.agent_state[ith_agent][0],self.agent_state[ith_agent][1]]=receiving
                    reward+=10
                else:
                    self.grid_state[self.agent_state[ith_agent][0],self.agent_state[ith_agent][1]]=empty
                
                #put the line at this step:
                # to avoid overwriting the original agent position with yellow or empty if noop action
                self.grid_state[a_obs[0],a_obs[1]]=package
                if not Trans:
                    self.grid_state[next_obs[0],next_obs[1]]=agent
                else:
                    self.grid_state[next_obs[0],next_obs[1]]=trans_agent
                self.agent_state[ith_agent]=copy.deepcopy(next_obs)
                self.package_state[ith_agent]=None
                reward+=80*self.count_pickup[ith_agent]
            #the case when the agent falls on the receiving area(agent only OR carrying and agent first)
            else:
                #turn the color of the space left behind to its original color
                if(tuple(self.agent_state[ith_agent]) in self.track.get_receiving_region()):
                    self.grid_state[self.agent_state[ith_agent][0],self.agent_state[ith_agent][1]]=receiving
                else:
                    self.grid_state[self.agent_state[ith_agent][0],self.agent_state[ith_agent][1]]=empty
                
                #put the line at this step:
                # to avoid overwriting the original agent position with yellow or empty if noop action
                if not Trans:
                    self.grid_state[next_obs[0],next_obs[1]]=agent
                else:
                    self.grid_state[next_obs[0],next_obs[1]]=trans_agent
                #put the agent to the next position
                self.agent_state[ith_agent]=copy.deepcopy(next_obs)
                
                #the case when we carry the package(maybe package on the yellow area[change the position of the package])
                if(self.info[ith_agent]["in_progress"]==True):
                    #turn the color of the space left behind to its original color
                    if(tuple(self.package_state[ith_agent]) in self.track.get_receiving_region()):
                        self.grid_state[self.package_state[ith_agent][0],self.package_state[ith_agent][1]]=receiving
                        self.grid_state[a_obs[0],a_obs[1]]=package
                        self.info[ith_agent]["success"]=True
                        self.info[ith_agent]["in_progress"]=False
                        self.info[ith_agent]["just_drop"]=True
                        self.package_state[ith_agent]=None
                        reward+=80*self.count_pickup[ith_agent]**2
                        reward+=10*self.count_pickup[ith_agent]**2
                    else: #package is not on receiving region
                        self.grid_state[self.package_state[ith_agent][0],self.package_state[ith_agent][1]]=empty
                        self.grid_state[a_obs[0],a_obs[1]]=package
                        self.package_state[ith_agent]=copy.deepcopy(a_obs)
        elif next_state==receiving and Recei:
            if(self.info[ith_agent]["in_progress"]==True):#this is the case when we are simply carrying the package & moving on the receiving region(towards transfer region)
                #the case when collectors are carring package; for collectors, we expect this to be treated the same way as empty region
                if(agent_first):
                    #turn the color of the space left behind to its original color; since the collector HAVE TO stay on receiving region, we only have this line
                    if(tuple(self.agent_state[ith_agent]) in self.track.get_receiving_region()):
                        self.grid_state[self.agent_state[ith_agent][0],self.agent_state[ith_agent][1]]=receiving
                    else:
                        self.grid_state[self.agent_state[ith_agent][0],self.agent_state[ith_agent][1]]=transfer

                    if(tuple(self.package_state[ith_agent]) in self.track.get_receiving_region()):
                        self.grid_state[self.package_state[ith_agent][0],self.package_state[ith_agent][1]]=receiving
                    elif(tuple(self.package_state[ith_agent]) in self.track.get_transfer_region()):
                        self.grid_state[self.package_state[ith_agent][0],self.package_state[ith_agent][1]]=transfer
                    else:
                        self.grid_state[self.package_state[ith_agent][0],self.package_state[ith_agent][1]]=empty

                    self.grid_state[next_obs[0],next_obs[1]]=recei_agent
                    #we need to check which position changed
                    self.grid_state[a_obs[0],a_obs[1]]=package
                    
                    self.agent_state[ith_agent]=copy.deepcopy(next_obs)
                    self.package_state[ith_agent]=copy.deepcopy(a_obs)
                #the case when moving in the direct way and if the package if the leading one
                else: #the sequence of setting the color of package state and agent state matters(so differ from first case)
                    #turn the color of the space left behind to its original color
                    if(tuple(self.package_state[ith_agent]) in self.track.get_receiving_region()):
                        self.grid_state[self.package_state[ith_agent][0],self.package_state[ith_agent][1]]=receiving
                    elif(tuple(self.package_state[ith_agent]) in self.track.get_transfer_region()):
                        self.grid_state[self.package_state[ith_agent][0],self.package_state[ith_agent][1]]=transfer
                    else:
                        self.grid_state[self.package_state[ith_agent][0],self.package_state[ith_agent][1]]=empty
                    
                    if(tuple(self.agent_state[ith_agent]) in self.track.get_receiving_region()):
                        self.grid_state[self.agent_state[ith_agent][0],self.agent_state[ith_agent][1]]=receiving
                    else:
                        self.grid_state[self.agent_state[ith_agent][0],self.agent_state[ith_agent][1]]=transfer

                    #we need to check which position changed
                    self.grid_state[a_obs[0],a_obs[1]]=package#since this case has the package being the leading one, we first assign the package state
                    self.grid_state[next_obs[0],next_obs[1]]=recei_agent
                    
                    self.package_state[ith_agent]=copy.deepcopy(a_obs)
                    self.agent_state[ith_agent]=copy.deepcopy(next_obs)
            else:#this is the case when we just simply step on the receiving region(the case for picking up package should be on the "package" condition)
                #turn the color of the space left behind to its original color
                if(tuple(self.agent_state[ith_agent]) in self.track.get_receiving_region()):
                    self.grid_state[self.agent_state[ith_agent][0],self.agent_state[ith_agent][1]]=receiving
                elif(tuple(self.agent_state[ith_agent]) in self.track.get_transfer_region()):
                    self.grid_state[self.agent_state[ith_agent][0],self.agent_state[ith_agent][1]]=transfer
                else:#this is the case when the agent is going to step on the empty region(just add to make sure the edge case is not going to happen)
                    #our collector cannot be on anywhere else, so we also need to penalize that
                    reward-=1
                    self.info[ith_agent]["bounce"]=True
                    if(self.info[ith_agent]["count_bounce"]<=5):
                        self.info[ith_agent]["count_bounce"]+=1
                    else:
                        reward-=10
                        self.info[ith_agent]["count_bounce"]=0
                    return self.grid_state,reward,self.done,self.info[ith_agent]
                
                self.grid_state[next_obs[0],next_obs[1]]=recei_agent
                #put the agent to the next position
                self.agent_state[ith_agent]=copy.deepcopy(next_obs)
        elif next_state==transfer and Recei:
            #if currently we are carrying package & the next state is the package instead of agent
            if(self.info[ith_agent]["in_progress"]==True and not next_state_agent):
                #turn area left behind to the original color
                if(tuple(self.package_state[ith_agent]) in self.track.get_transfer_region()):
                    self.grid_state[self.package_state[ith_agent][0],self.package_state[ith_agent][1]]=transfer
                elif(tuple(self.package_state[ith_agent]) in self.track.get_receiving_region()):
                    self.grid_state[self.package_state[ith_agent][0],self.package_state[ith_agent][1]]=receiving
                else:
                    self.grid_state[self.package_state[ith_agent][0],self.package_state[ith_agent][1]]=empty
                
                if(tuple(self.agent_state[ith_agent]) in self.track.get_transfer_region()):
                    self.grid_state[self.agent_state[ith_agent][0],self.agent_state[ith_agent][1]]=transfer
                else:
                    self.grid_state[self.agent_state[ith_agent][0],self.agent_state[ith_agent][1]]=receiving
                    
                #convert the new state to the color we want
                self.grid_state[a_obs[0],a_obs[1]]=package
                self.grid_state[next_obs[0],next_obs[1]]=recei_agent
                self.package_state[ith_agent]=None
                self.agent_state[ith_agent]=copy.deepcopy(next_obs)
                self.info[ith_agent]["in_progress"]=False
                #this is count towards the same way as the other agents putting things onto the receiving region
                reward+=80*self.count_pickup[ith_agent]
                
                is_transferred=True
                self.received+=1
            else:#if agent is the first one arrives on the transfer region:
                #1) this includes the case when we only have the agent arrives alone on the transfer region
                #2) and also the case when we have the package with us as well but the agent is the first one
                #turn the color of the space left behind to its original color
                if(tuple(self.agent_state[ith_agent]) in self.track.get_transfer_region()):
                    self.grid_state[self.agent_state[ith_agent][0],self.agent_state[ith_agent][1]]=transfer
                else:
                    self.grid_state[self.agent_state[ith_agent][0],self.agent_state[ith_agent][1]]=receiving
                
                #put the line at this step:
                # to avoid overwriting the original agent position with yellow or empty if noop action
                self.grid_state[next_obs[0],next_obs[1]]=recei_agent
                #put the agent to the next position
                self.agent_state[ith_agent]=copy.deepcopy(next_obs)
                
                #the case when we carry the package(maybe package on the yellow area[change the position of the package])
                if(self.info[ith_agent]["in_progress"]==True):
                    #turn the color of the space left behind to its original color
                    if(tuple(self.package_state[ith_agent]) in self.track.get_transfer_region()):
                        self.grid_state[self.package_state[ith_agent][0],self.package_state[ith_agent][1]]=transfer
                        self.grid_state[a_obs[0],a_obs[1]]=package
                        self.info[ith_agent]["in_progress"]=False
                        self.info[ith_agent]["just_drop"]=True
                        self.package_state[ith_agent]=None
                        reward+=80*self.count_pickup[ith_agent]**2
                        reward+=10*self.count_pickup[ith_agent]**2
                        is_transferred=True
                        self.received+=1
                    else: #package is not on transfer region
                        self.grid_state[self.package_state[ith_agent][0],self.package_state[ith_agent][1]]=receiving
                        self.grid_state[a_obs[0],a_obs[1]]=package
                        self.package_state[ith_agent]=copy.deepcopy(a_obs)
        
        #count the time passed in seconds
        self.time+=0.2 #100ms if not having time.sleep
        self.info[ith_agent]["time"]=self.time
        #as the time pass, we automatically reduce reward(increase anxiety of the agent if possible :))
        reward-=10
        
        #our aim of using the collector is to make sure the region of receiving not too crowded
        self.count_done=self.count_delivered()
        
        if self.track.in_control_room(self.agent_state,ith_agent):
            reward-=100
        
        #this is to update the image for the package delivered onto the transfer region
        """delivered package onto the transfer region
        1) update the region to the transfer region(means we have delivered the package out from the warehouse)
        2) if this is the case, we have delivered the package on the transfer region
        """
        if(is_transferred):
            self.grid_state[a_obs[0],a_obs[1]]=transfer
        
        #reward for approaching the shelves
        if(self.info[ith_agent]["in_progress"] and not agent_first):
            updated_state=self.package_state[ith_agent]
        else:
            updated_state=self.agent_state[ith_agent]
            
        #Compare the Distance and then offer Rewards
        if (self.info[ith_agent]["in_progress"]==False):
            if(not Trans):#normal agent & collector
                if(not Recei):#normal agent(want the package not on receiving region)
                    dist=self.comp_dist(package,current_state,updated_state,is_receiving=False)
                    dist2collector=-self.comp_dist(recei_agent,current_state,updated_state)
                    reward+=dist2collector
                else:#collector(we want the package on receiving region)
                    dist=self.comp_dist(package,current_state,updated_state,is_receiving=True)
                reward+=4*dist
            #Helper case
            else:#The helper want to get closer to the agent who is carrying the package that is far from the receiving region
                max_dist2agent=self.track.cmp_dist2agents(self.grid_state,receiving,self.info,self.agent_state,current_state,next_obs,max=True,to_with_package=True,package_state=self.package_state,is_receiving=False)
                #This case only happens when there are agents that are currently carrying packages exist in this warehouse
                reward+=8*max_dist2agent
        #reward for taking package to receiving region - actually approaching the transfer region NOW
        elif (self.info[ith_agent]["in_progress"]==True):
            """This case we are:
            1) carrying packages currently(carrying the packages by ourselves)
            2) looking for agents with no packages that are not collectors
            On this way, we want to prioritize the action of approaching the receiving region
            """
            dist=self.comp_dist(transfer,current_state,updated_state,is_receiving=True)
            reward+=10*dist
            if(not Trans and not Recei):#Normal agents
                min_agent,min_dist=self.track.optimal_agent2objective(self.info,self.grid_state,self.agent_state,objective=agent,max=False,to_with_package=False,package_state=None,is_receiving=True)
                help_dist=self.track.cmp_dist2agents(self.grid_state,receiving,self.info,self.agent_state,current_state,next_obs,max=False,to_with_package=False,package_state=None,is_receiving=True)
            elif(Trans and not Recei):#Helper agents
                reward+=10*dist
                min_agent,min_dist=self.track.optimal_agent2objective(self.info,self.grid_state,self.agent_state,objective=trans_agent,max=False,to_with_package=False,package_state=None,is_receiving=True)
                help_dist=self.track.cmp_dist2agents(self.grid_state,receiving,self.info,self.agent_state,current_state,next_obs,max=False,to_with_package=False,package_state=None,is_receiving=True)
            #and also, to ensure that we are not tranferring to any agent that is far from the receiving region compare to our current one, we need to ensure that the help distance is smaller
            if(not Recei):
                if(dist>0 and min_dist<abs(dist)):#we have absolute value function because the distance is already "rewardlized"
                    """the purpose of this operation is to ensure the agents are happy to work together given the condition that we are heading towards completing the task
                    Problem: we should only encourage actions approaching AGENTS CLOSER TO THE RECEIVING region & WITH NO PACKAGES(not all agents of opposite kind)
                    *To solve this problem, we don't really need the target to be specifically helper or agent(normal), we just need our target to be agents with no package carrying at this moment
                    """
                    reward+=8*help_dist
                    if(not Trans):#this is the reward for normal agent approaching to the helper if we are approaching the receiving region
                        reward+=2*help_dist
        #useless actions
        else:
            reward-=10
        
        if(self.count_done==self.transferring_packages_goal):
            self.done=True
            self.count_done=0
        elif(self.level_achieved==0 and self.count_done==10):
            reward+=500
            self.level_achieved=1
        elif(self.level_achieved==1 and self.count_done==50):
            reward+=1000
            self.level_achieved=2
        elif(self.level_achieved==2 and self.count_done==100):
            reward+=1500
            self.level_achieved=3
        
        return self.grid_state,reward,self.done,self.info[ith_agent]
    
    def last_row_receiving(self,package_state):
        #first we get the next row of the package
        next_row=package_state[0]+1
        next_col=package_state[1]
        transfer_row=self.track.get_transfer_region()[0]
        
        #first case, the next row is directly the transfer row
        if(self.grid_state[next_row,next_col]==transfer):
            return True
        #second case(a large one), the next one is a package(this can be based on the case when the rows before are packages as well or not)
        elif(self.grid_state[next_row,next_col]==package):
            for index in range(next_row,transfer_row): #whenever the packge of the next state possible is not solid
                if(self.grid_state[index,next_col]==receiving):
                    return False
            return True
        return False
    
    #this function reset all the states to the intial state
    def reset(self):
        #reset the environment state to the initial one
        self.grid_state = copy.deepcopy(self.initial_state)
        #update the other states(the agent state to the current one)
        self.agent_state=self.get_state()
        for i in range(self.num_agents):
            self.package_state[i]=None
        self.level_achieved=0
        self.time=0
        self.count_done=0
        self.received=0
        return self.grid_state
    
    #Note that this function is currently static - we need to alter it if we want to show it with agent moving after adding the network
    def render(self,mode="human",close=False,reward=None,count_time=0,delivered=0,ThreeD_vis=False,return64web=False,web_plt=None):
        if(not return64web):
            self.Visual_View.render(self.plt,self.grid_state,mode,close,reward,count_time,delivered,ThreeD_vis,return64web,web_plt)
        else:#the case when we are visualizing for the webapp
            current_state_base64=self.Visual_View.render(self.plt,self.grid_state,mode,close,reward,count_time,delivered,ThreeD_vis,return64web,web_plt)
            return current_state_base64
        
    async def stream_layout(self, websocket: WebSocket):
        # await websocket.accept()
        try:
            # since we previously sent a message from the front end to backend - we should receive a message here
            # msg = await websocket.receive_json()
            # if msg.get("type") != "ready":
            #     print("Invalid init message. Closing WebSocket.")
            #     await websocket.close()
            #     return
            
            layout = self.grid_state.astype(int).tolist()  # Make sure this returns 2D list
            # print("layout sent for 3D version:",layout)
            
            # CURRENTLY we just don't send from this end(COMMENTED it out):
            # await websocket.send_json({"layout": layout})
            
            # this below is to generate the plausible payload for 3d frame
            payload = {"type":"render","layout":layout,"ts":time.time()}
            return payload
            # await asyncio.sleep(0.5)
        except TypeError as e:
            print(f"Layout stream error: {e}")
        except WebSocketDisconnect:
            print("Disconnected from layout_sream")
        except Exception as e:
            print(f"Layout stream error: {e}")