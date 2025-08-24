"""This class is specifically for the get methods from the environment.
Note: since the compare distance function is returning in a opposite logic, every distance return from this cmp function should be reversed
1)cmp_dist_to_agents is mainly used to compare the distance from different agents(that are not helpers) that are currently carrying packages
to the target region(which is usually receiving region in this case).
This function is used for helper agents on searching to help agents that are away from the receiving region to carry packages.
"""
# from . import BFS_map
from Env import BFS_map
from Env.BFS_map import BFS
import numpy as np
import math
empty = gray = 0
wall = black = 1
agent = red = 2
package = brown = 3
receiving = yellow = 4
control = 5
transfer = sky = 6
updated_receiving=7
trans_agent = blue = 8
recei_agent = sea = 9

class Get:
    def __init__(self,grid_state,initial_state,agent_state,initial_empty_state):
        self.grid_state=grid_state
        self.initial_state=initial_state
        self.initial_empty_state=initial_empty_state
        self.agent_state=agent_state
        self.all_num_agents=np.sum(self.grid_state==agent)+np.sum(self.grid_state==trans_agent)+np.sum(self.grid_state==recei_agent)
        
    def get_helper_region(self,grid_state):
        self.grid_state=grid_state
        return list(zip(*np.where(self.grid_state==trans_agent)))
    
    def get_agent_region(self,grid_state):
        self.grid_state=grid_state
        return list(zip(*np.where(self.grid_state==agent)))
    
    def get_collector_region(self,grid_state):
        self.grid_state=grid_state
        return list(zip(*np.where(self.grid_state==recei_agent)))
    
    def get_package_region(self,grid_state):
        self.grid_state=grid_state
        package_available_region=(self.grid_state==package)&(self.initial_state!=receiving)
        return list(zip(*np.where(package_available_region)))
    
    def get_package_region_original(self,initial_state):
        self.initial_state_state=initial_state
        return list(zip(*np.where(self.initial_state==package)))
    
    def get_custom_agent(self,correspond_agent_state):
        return [tuple(correspond_agent_state)]
    
    #Returns: index of the agents that satisfies the requirement & the list of agents WITH packages
    def get_agent_with_package(self,info,agent_state):
        self.agent_state=agent_state
        agents=[]
        list_agents=[]
        for ith in range(len(self.agent_state)):
            if(info[ith]["in_progress"]==True):
                agents.append(ith)
                agent_added=self.agent_state[ith]
                list_agents.append(agent_added)
        return list(agents),list_agents
    
    #Returns:index of the agents that satisfies the requirement & the list of agents WITHOUT packages
    def get_agent_without_package(self,info,agent_state):
        self.agent_state=agent_state
        agents=[]
        list_agents=[]
        for ith in range(len(self.agent_state)):
            if(info[ith]["in_progress"]==False):
                agents.append(ith)
                agent_added=self.agent_state[ith]
                list_agents.append(agent_added)
        return list(agents),list_agents
    
    #the total number of agents(including all types of agents)
    def get_num_agents(self):
        return self.all_num_agents
    
    #get the receiving region
    def get_receiving_region(self):
        return list(zip(*np.where(self.initial_empty_state==receiving)))
    
    def get_updated_receiving_region(self,updated_grid_state):
        self.grid_state=updated_grid_state
        return list(zip(*np.where(self.grid_state==receiving)))
    
    #get current empty region
    def get_empty_region(self):
        return list(zip(*np.where(self.initial_empty_state==empty)))
    
    def get_wall(self):
        solid=(self.initial_empty_state==wall) | (self.initial_empty_state==control)
        return list(zip(*np.where(solid)))
    
    def get_transfer_region(self):
        return list(zip(*np.where(self.initial_empty_state==transfer)))
    
    #know whether it is in control room region
    def in_control_room(self,agent_state,ith_agent):
        self.agent_state=agent_state
        row=self.agent_state[ith_agent][0]
        col=self.agent_state[ith_agent][1]
        
        """edge points(row,col):
        2 points left side of control room but besides the wall of warehouse
        (beware the wall might change upward/downward) - (1,45), (13,45)
        
        4 points in control room - (3,49), (3,63), (10,49), (10,63)
        """
        control_room=list(zip(*np.where(self.initial_empty_state==control)))
        low_r=float("inf")
        low_c=float("inf")
        high_r=-float("inf")
        high_c=-float("inf")
        for p in control_room:
            #know the range of row
            if p[0]<low_r:
                low_r=p[0]
            elif p[0]>high_r:
                high_r=p[0]
            
            #know the range of column
            if p[1]<low_c:
                low_c=p[1]
            elif p[1]>high_c:
                high_c=p[1]
        
        #know whether the agent is in control room region
        if((row>=low_r and row<=high_r) and (col>=high_c and col<=high_c)):
            return True
        return False
            
    """get the closest distance of the state to any region in receiving area(or other specific areas)
    this is the compare functions to get the distances
    Note: this encourage the distance closer(positive rewarding), and underplay distance farther(negative rewrading)
    1)target_trans:for objective being agent without package and the objective is normal agent or just helper
    2)custom_agent: for objective being custom agent
    """
    def comp_dist(self,grid_state,objective,current_state,next_obs,custom_agent=False,correspond_agent_state=None,is_receiving=False):
        origin=float('inf')
        updated=float('inf')
        self.grid_state=grid_state
        
        #know what is the comp dist objective region
        if(objective==receiving):
            region=self.get_receiving_region()
        elif(objective==package):
            region=self.get_package_region(self.grid_state)
        elif(objective==updated_receiving):
            region=self.get_updated_receiving_region(self.grid_state)
        elif(objective==trans_agent):
            region=self.get_helper_region(self.grid_state)
        elif(objective==agent):
            region=self.get_agent_region(self.grid_state)
        elif(objective==recei_agent):
            region=self.get_collector_region(self.grid_state)
        elif(objective==transfer):
            region=self.get_transfer_region()
        elif(objective==None and custom_agent and correspond_agent_state is not None):
            region=self.get_custom_agent(correspond_agent_state)
        else:
            region=self.get_package_region(self.grid_state)
            
        #this gets the closest spot in the region to the original one(agent)
        for area in region:
            x,y=area[0],area[1]
            point=tuple((x,y))
            dist_origin=math.dist(current_state,point)
            dist_updated=math.dist(next_obs,point)
            if(objective==package):
                in_receiving=(point in self.get_receiving_region())
                #this deal with the case when the package we are looking for is not in our target
                if((is_receiving and not in_receiving)or(not is_receiving and in_receiving)):
                    continue
            
            if(dist_origin<origin):
                origin=dist_origin
            if(dist_updated<updated):
                updated=dist_updated
        
        if(objective==receiving):
            return_signal = self.map_BFS_cmp(receiving,current_state,next_obs,custom_agent=False,correspond_agent_state=None,is_receiving=is_receiving)
        elif(objective==updated_receiving):
            return_signal = self.map_BFS_cmp(updated_receiving,current_state,next_obs,custom_agent=False,correspond_agent_state=None,is_receiving=is_receiving)
        elif(objective==package):
            return_signal = self.map_BFS_cmp(package,current_state,next_obs,custom_agent=False,correspond_agent_state=None,is_receiving=is_receiving)
        elif(objective==trans_agent):
            return_signal = self.map_BFS_cmp(trans_agent,current_state,next_obs,custom_agent=False,correspond_agent_state=None,is_receiving=is_receiving)
        elif(objective==agent):
            return_signal = self.map_BFS_cmp(agent,current_state,next_obs,custom_agent=False,correspond_agent_state=None,is_receiving=is_receiving)
        elif(objective==recei_agent):
            return_signal = self.map_BFS_cmp(recei_agent,current_state,next_obs,custom_agent=False,correspond_agent_state=None,is_receiving=is_receiving)        
        elif(objective==transfer):
            return_signal = self.map_BFS_cmp(transfer,current_state,next_obs,custom_agent=False,correspond_agent_state=None,is_receiving=is_receiving)
        elif(objective==None and custom_agent):
            return_signal = self.map_BFS_cmp(None,current_state,next_obs,True,correspond_agent_state,is_receiving)
        else:
            #the updated action makes the situation worse
            if(origin<updated):
                return -1*(updated**4)
            #the updated action makes the situation better
            elif(origin>updated):
                return 1*(origin**4)
        return return_signal

    #get the map distance with Breadth First Search(Data Structures)
    def map_BFS_cmp(self,objective,origin_state,updated_state,custom_agent=False,correspond_agent_state=None,is_receiving=False):
        if(not custom_agent and correspond_agent_state is None):
            original=BFS(objective,origin_state,self.grid_state,self.initial_empty_state,custom_agent=False,correspond_agent_state=None)
            updated=BFS(objective,updated_state,self.grid_state,self.initial_empty_state,custom_agent=False,correspond_agent_state=None)
        elif(not custom_agent and correspond_agent_state is not None):
            raise ValueError("Invalid: Can't have Custom Agent be False while Correspond Agent State Exist!!")
        elif(custom_agent and correspond_agent_state is None):
            raise ValueError("Invalid: Can't have Custom Agent be True while Correspond Agent State Not Exist!!")
        else:#this is the case when we are customizing the agent we are searching for
            original=BFS(objective,origin_state,self.grid_state,self.initial_empty_state,custom_agent,correspond_agent_state)
            updated=BFS(objective,updated_state,self.grid_state,self.initial_empty_state,custom_agent,correspond_agent_state)
            
        map_original_dist=original.map_distance(is_receiving)
        map_updated_dist=updated.map_distance(is_receiving)
    
        #problem: no path found
        if(map_original_dist == float("inf") or map_updated_dist==float("inf")):
            return 0
        
        if(map_original_dist<map_updated_dist):#penalize the case when the new route is even farther from the target
            return -1*(map_updated_dist**4)
        elif(map_original_dist>map_updated_dist):
            return 1*(map_original_dist**4)
        else:
            return 0
    
    """This function compares agents at different positions inside the warehouse;
    and we want to find the agent that is having the max/min distance from the objective.
    Return: (1)the agent's state, (2)the corresponding distance of the agent to the objective region
    Our thought is this:
    1. first we find the distance from different agents that satisfies the requirement to the objective(this can be done with to_with_package keyword)
    2. next we compare the distances and return the agent and the corresponding distance that satisfy the max/min target
    
    1)max: the keyword that defines whether we are looking for the agent with the maximum distance to the objective
    if max is False, it means we are looking for the minimum distance;
    if max is True, it means we are looking for the maximum distance
    2)to_with_package: this is the keyword that defines whether we are trying to approach the agent with package or not
    if this keyword is False, it means we are now CARRYING the package, but we are trying to find the agent without the package;
    if this keyword is True, it means we are NOT CARRYING the package, we are trying to find the agent with the package
    """
    def optimal_agent2objective(self,info,grid_state,agent_state,objective=receiving,max=False,to_with_package=False,package_state=None,is_receiving=False):
        self.agent_state=agent_state
        #we need to make sure either to_with_package and package_state both exist or both not exist
        if((to_with_package and package_state is None) or (not to_with_package and package_state is not None)):
            raise ValueError("Invalid: key word 'to_with_package' and 'target_package_state' should either both exist or both not exist!!")
        
        #first we try to find all the agents that satisfy the requirement of carrying package/not carrying package
        if(to_with_package):
            target_package_index=-1
            indices,list_agents=self.get_agent_with_package(info,self.agent_state)
        else:
            indices,list_agents=self.get_agent_without_package(info,self.agent_state)

        self.grid_state=grid_state
        #know what is the comp dist objective region
        if(objective==receiving):
            region=self.get_receiving_region()
        elif(objective==package):
            region=self.get_package_region(self.grid_state)
        elif(objective==updated_receiving):
            region=self.get_updated_receiving_region(self.grid_state)
        elif(objective==trans_agent):
            region=self.get_helper_region(self.grid_state)
        elif(objective==agent):
            region=self.get_agent_region(self.grid_state)
        elif(objective==transfer):
            region=self.get_transfer_region()
        else:
            region=self.get_package_region(self.grid_state)
            
        #this is the target distance from the best agent to the objective region(either max or min, depend on the condition)
        #this finds the target distance(from the best agent to the target region with the expected distance[max/min])
        #Now we want to initialize the target dist and the corresponding agent state
        if(not max):#find the minimum distance
            target_dist=float('inf')
        else:#we are looking for the maximum distance case
            target_dist=-float('inf')
        #initialize the target agent state
        target_agent=None
        
        #(Method 1)this step trys to find the agent(that satisfies the requirement of package) that has the max/min distance from the objective region
        for area in region:
            x,y=area[0],area[1]
            point=tuple((x,y))
            for index,ith_agent in zip(indices,list_agents):
                a,b=ith_agent[0],ith_agent[1]
                agent_pos=tuple((a,b))
                dist=math.dist(agent_pos,point)#this line compute the distance from this current agent to the objective region
                if(not max):#if to find the minimum distance
                    if(dist<target_dist):
                        target_dist=dist
                        target_agent=ith_agent
                        if(to_with_package):
                            #same logic as in BFS map
                            if(is_receiving and (tuple(package_state[index]) in self.get_receiving_region())):
                                target_agent=package_state[index]
                            elif(not is_receiving and (tuple(package_state[index]) not in self.get_receiving_region())):
                                target_agent=package_state[index]
                else:
                    if(dist>target_dist):
                        target_dist=dist
                        target_agent=ith_agent
                        if(to_with_package):
                            #same logic as in BFS map
                            if(is_receiving and (tuple(package_state[index]) in self.get_receiving_region())):
                                target_agent=package_state[index]
                            elif(not is_receiving and (tuple(package_state[index]) not in self.get_receiving_region())):
                                target_agent=package_state[index]
        
        #this returns the distance from the optimal agent to the target region
        if(target_dist!=float('inf') and target_dist!=-float('inf') ): #this checks whether the absolute value way of computing work or not
            target_dist=target_dist**4
            return target_agent, target_dist
        else:#(Method 2) when distance equation doesn't work much - now we want to use BFS searching method
            for ith_agent in list_agents:
                a,b=ith_agent[0],ith_agent[1]
                agent_pos=tuple((a,b))
                cmp_agent=BFS(objective,agent,self.grid_state)
                new_dist_cmp=cmp_agent.map_distance(is_receiving)
                if(not max):#to find the minimum distance from the agent to the target region
                    if(new_dist_cmp<target_dist):
                        target_dist=new_dist_cmp
                        target_agent=ith_agent
                else:#to find the maximum distance
                    if(new_dist_cmp>target_dist):
                        target_dist=new_dist_cmp
                        target_agent=ith_agent
            
            target_dist=target_dist**4
            return target_agent, target_dist

    """Since we already have the previous "comp_agents" function;
    this function is simply used to find the distance from one agent to another target agent and then penalize or reward based on this found distance.
    General Thought:
    1. first we find the distance from the agent (that satisfy the requirement to the objective region) & the agent's state
    2. next we penalize based on the current state and the next state(next_obs); whether we are moving towards or moving away from the targeted agent
    *Note: this involves compare the distance of current state to the target
    """
    def cmp_dist2agents(self,grid_state,objective,info,agent_state,current_state,next_obs,max=False,to_with_package=False,package_state=None,is_receiving=False): #if max is False, this means find the one with the min distance
        #first we need to make sure that package_state and "to_with_package" should either both exist or both not exist
        if((to_with_package and package_state is None) or (not to_with_package and package_state is not None)):
            raise ValueError("Invalid: key word 'to_with_package' and 'package_state' should either both exist or both not exist!!")
        
        self.grid_state=grid_state
        self.agent_state=agent_state
        #First, we try to find the optimal agent & its distance
        optimal_agent_state, optimal_agent_dist=self.optimal_agent2objective(info,self.grid_state,self.agent_state,objective,max,to_with_package,package_state,is_receiving)
        
        #Next, we compare our distance(whether moving away or near the target agent) to the optimal agent we are targeting towards
        #and we also reward/penalize based on the distance
        reward_correspond=self.comp_dist(self.grid_state,objective,current_state,next_obs,custom_agent=True,correspond_agent_state=optimal_agent_state,is_receiving=is_receiving)
        return reward_correspond