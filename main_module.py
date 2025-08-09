"""High Dimension Risk:
having 5 agents in the envrionment

Currently if we want to modify number of agents, we need to:
1) modify the layout
2) modify the parameters of Envrionment and Agent in this main function
"""
import os,sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__),'Env'))
import env_module # type: ignore
from env_module import Environment # type: ignore
sys.path.append(os.path.join(os.path.dirname(__file__),'DQN_test'))
import DQN_module # type: ignore
from DQN_module import Replay_buffer,DQN,Agent,writer,DQN_args_parser # type: ignore
sys.path.append(os.path.join(os.path.dirname(__file__),'PPO_test'))
#[debug] If can't be sure about whether we have correctly import the folder, we print this path
# print("sys.path:", sys.path)
import PPO_module # type: ignore #Note also that if we have imported the folder, we can directly import the module file
from PPO_module import Agent,writer,PPO_args_parser # type: ignore

#Block 5: run the main function
if __name__=="__main__":
    #first we create the envrionment
    #default done_standard is 146
    env=Environment(done_standard=146)
    n_agents=env.num_agents
    nn_choice=input("Select the neural network choice for the Agent:")
    visualize_choice=input("Select the way we render the warehouse(3D or 2D):")
    web_choice=input("whether intend for web(yes or no):")
    
    #This checks whether we want to use 3D as the way we render the environment - for more realistic experience
    if(visualize_choice=="3D"):
        ThreeD=True
    elif(visualize_choice=="2D"):
        ThreeD=False
    else:
        raise ValueError("Input Error: Your have to enter either '3D' or '2D'! No other options")
    
    #this checks whether return for web(add in addition,actually have no use here)
    if(web_choice=="yes"):
        return4web=True
    elif(web_choice=="no"):
        return4web=False
    else:
        raise ValueError("Input Error: you have to enter 'yes' or 'no' for web option!")
    
    #then create the agent
    if(nn_choice=="DQN"):
        DQN_args,DQN_parser=DQN_args_parser()
        agent=DQN_module.Agent(DQN_args,env,DQN_module.writer,False,ThreeD,return4web)
    elif(nn_choice=="PPO"):
        PPO_args,PPO_parser=PPO_args_parser()
        agent=PPO_module.Agent(PPO_args,env,PPO_module.writer,ThreeD,return4web)
    else:
        raise ValueError("Input Error: You have to enter either 'DQN' or 'PPO'!")
    
    #after this we train the agent running in the envrionment
    agent.train(1)
    print("Training Complete")
    