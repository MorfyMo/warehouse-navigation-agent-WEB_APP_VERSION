"""General Idea of BFS(Breadth First Search - Data Structures):
    store the start position to the queue(FIFO)[if don't store, it is okay as well];
    for each position popped from the queue, look at the neighbors enabled by the actions;
    for each neighbour, if it has not been visited before (and not an obstacle):
        we add the neighbour to the queue (and mark it as visited);
        and optionally store the parent cell so that we can reconstruct the path later
    """
from collections import deque
import numpy as np

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

class BFS:
    def __init__(self,objective,start_state,grid_map,initial_state,custom_agent=False,correspond_agent_state=None):
        self.queue=deque([(tuple(start_state),0)])#each elements is (state,distance_to_goal)
        #actions(in displacement form)
        self.actions=[
            (-1,0), #up
            (1,0), #down
            (0,-1), #left
            (0,1) #right
        ]
        self.custom_agent=custom_agent
        self.custom_agent_state=None
        #the goal(i.e. receiving, package) we want(and then we would try to find the distance to the goal)
        if(objective==receiving):
            self.object=receiving
        elif(objective==package):
            self.object=package
        elif(objective==updated_receiving):
            self.object=receiving #since we are using self.grid_state as grid_map in the original occasion, we just set the object as receiving for updated receiving
        elif(objective==trans_agent):
            self.object=trans_agent
        elif(objective==transfer):
            self.object=transfer
        elif(objective==agent):
            self.object=agent
        elif(objective==recei_agent):
            self.object=recei_agent
        elif(objective==None and custom_agent):#case for customr agent
            self.object=agent
            try:
                self.custom_agent_state=correspond_agent_state
            except:
                raise NameError("ith and agent_state should be valid for custom agent assignment!!")
        else:
            self.object=package
        self.grids=grid_map
        self.initial_state=initial_state
        self.visited=[]
        self.distance=0
    
    #get neighboring positions based on the available actions
    def get_neighbors(self,dist):
        available_actions=[]
        if(len(self.queue)==0):
            return None
        
        for i in self.actions:
            row_change=i[0]
            col_change=i[1]
            current,dist=self.queue.popleft()
            neigh_r=current[0]+row_change
            neigh_c=current[1]+col_change
            
            if(self.grids[neigh_r,neigh_c]==empty or self.grids[neigh_r,neigh_c]==receiving):
                neighbor=(neigh_r,neigh_c)
                self.queue.append((neighbor,dist+1))
                available_actions.append(i)
        
        if(len(self.queue)==0):
            return None
        return available_actions
    
    #get BFS map distance:
    #this is a recursive function
    #Note: this method uses the "get_neighbors" function above
    def map_distance(self,is_receiving=False):
        while(self.queue):
            #first pop out the top state and its distance to the goal
            top_state,top_dist=self.queue.popleft()
            dist=top_dist
            
            #then check if we have reached the receiving area
            row=top_state[0]
            col=top_state[1]
            if(not self.custom_agent):
                if(self.grids[row,col]==self.object):
                    #for the package case, we need to discuss it in specific(special case)
                    if(self.object==package):
                        #this specifies that we are discussing the case abt package
                        #(1. on receiving region 2. on shelves)
                        if(is_receiving and tuple((row,col)) in self.receiving_region()):#this deals with the packages on the receiving region
                            self.distance=dist
                        elif(not is_receiving and tuple((row,col)) in self.package_region()):#packages from the shelves
                            self.distance=dist
                    else:
                        self.distance=dist
                    return dist
            elif self.custom_agent and self.custom_agent_state is not None:#this is the case when we customize the agent region we want to search for
                if(row==self.custom_agent_state[0] and col==self.custom_agent_state[1]):
                    self.distance=dist
                    return dist
            
            #now if we haven't reached the final area, we check whether we have already went through this position
            if tuple(top_state) in self.visited:
                continue
            self.visited.append(tuple(top_state))
            
            #on the way-get the neighbors(the next possible step)
            available_actions=self.get_neighbors(dist)
            if(self.queue==None):
                return float("inf")
            
        return float("inf")
    
    def package_region(self):
        package_available_region=(self.grids==package)&(self.initial_state!=receiving)
        return list(zip(*np.where(package_available_region)))
    
    def receiving_region(self):
        return list(zip(*np.where(self.initial_state==receiving)))