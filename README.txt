This RL uses DQN Agent to interact and learn from the environment.

In this example we would try construct a warehouse Robot Navigation.
1. There are a total of 146 number of packages expected to be carried to the receiving region from the shelves,
but since it is hard to achieve, user can alter the number of expected package to be carried to the receiving region as goal.
2. In this context, we assume the robot can only take one pakage at a time;
in addition, it can move/change position of the package already put onto the yellow area.
3. However, even though the transfer region(where other characters would come in this place and take the packages put on the receiving region)
is flat in theory, the agents should not step onto the transfer region(this is not their working environment) - This is treated in the same way as invalid.
4. Types of agents:
1)normal one
2)helper(trans_agent): The only difference between helper and normal agent is helper cannot pickup package from anywhere(shelves, receiving region) except from the other agents
3)collector(recei_agent): the only area collector can move onto is the transfer region and the receiving region;
and also, collector is the only agent that does not allow operation of tranferring packages between agents to happen
*using collector is to reduce the burden of package collection o the receiving region
Note:[it can only pickup packages on the receiving region & drop packages on transfer region]

Goal: learn a policy that reaches targets in a partialy known warehouse layout

State(Observation sapce): the robot's position + the layout of the map
Action Space: {no_action, up, down, left, right, turn_left, turn_right}
Reward Rules:
1) +25*time_of_pickup reach a package(also reward the case when successfully transfer a package from a agent to another)
*another +85 reward if normal agent transfer package to helper(or helper to normal agent)
2) +80*count_pickup transmitting the package to the yellow area
3) +10*count_pickup**2 more if the agent is carrying package onto the receiving region(encourage the agents to proceed on receiving region instead of staying on the edge)
*note that in this case the previous 80 would also becomes 80*count_pickup**2
4) -3 if crash into obstacles(shelves/walls)
5) +4*dist_to_package^4 if the agent approachs the shelves if it does not carry packages
*-4*dist_to_package^4 if goes far
6) +10*dist_to_receiving^4 if the agent with package get closer to the receiving region
*-4*dist_to_receiving^4 if get far
7)-dist2collector for the normal agent when it is not carrying packages
8) -10 if the action does not make the situation better
9) -0.1 if crash into package/next state invalid
10) -100 continuously if stay in the control room
11) +8*dist_trans_agent if approach helper(or helper appraoch agent)when helper is not carrying packages(encourage normal agents to deem helpers as real assistants instead of competitor)
*
1. +8*dist_to_max_dist_agent if approach agent that is carrying packages but is having the largest distance from the receiving region(when we are the helper agent)
2. +8*dist_to_trans_agent more if we are having this current agent(normal/helper) getting close to the helper/agent when delivering package
*currently we just allow helper approaching add reward - because we find out thtat cooperation from normal agent with this helper doesn't help much
3. +2*dist_to_agent_transfer_to if we are having our current agent trying to transfer package to someone closer to the receiving region(in general)
NOTE: one alternative is if the agent pass package to helper, they get similar reward as dropping onto receiving region, but do not add anything if approaching helper
4. +10*dist_to_receiving for the helper(to make the helper have more incentive to deliver package)
12) -110*count_pickup if agent tries to fool for reward by taking and carrying packages from yellow space again
13) -0.1 as the agent stay more in the process, we automatically reduce reward(to increase anxiety of the agent)
14) Extra Bonue:
1. +500 if reach a 10 package delivered result
2. +1000 if reach a 50 package delivered result
3. +1500 if reach a 100 package delivered result
15) 0 otherwise
*to be noted as a result of formulation, this problem does not need goal state
- as our target is to move all the packages to the yellow area;
this means that if we have "done" being True, we have reached the goal of our RL problem

4 elements of envrionment:
1) next_state
2) reward
3) done: when all the packages have been put on the yellow area - with no more and no less
4) info: let's say info has two keys("success": when the robot successfully put package on the yellow area
, "in_progress": when the robot has got the package but have not put the package on yellow area)

#Version:
fastapi==0.104.1
uvicorn[standard]==0.24.0
websockets==12.0