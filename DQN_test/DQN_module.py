#DQN specific Blocks:
"""
DQN specific Classes__ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
1. Replay_Buffer: the buffer that stores the past experiences
2. DQN: the model an agent would be ; this part involves the policy
3. Agent: the agent itself - it uses "replay_buffer" and "DQN" we defined before - has the "train method"

This DQN uses epsilon-greedy policy.
1) The strategy expect to use is:
if we try the same action over 2 times and its does not work for the current condition, we switch to another.

2) The strategy currently use is:
when acting randomly(in epsilon probability),
2.1) we avoid using the same action(to wall and package) if the agent is crashing into solid as before;
2.2) meanwhile, if the agent is crashing and the agent is carrying package, we try the turn actions if it is not trying before;
2.3) if the agent continueously crash into solid for 5 times, as the reward -10(in environment module), we also freeze the action for 10 future steps.
2.4*) Not Random Action: if the agent is not bouncing to hard things, we just continue the action
"""
import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__),'..','Env')) #we do not need Test_RL here because this folder is already in the largest environment of Test_RL folder
import env_module # type: ignore
from env_module import Environment,up,down,left,right,p_left,p_right # type: ignore
import argparse
from datetime import datetime
import random
import numpy as np
import time
from collections import deque
import tensorflow as tf
from tensorflow import keras # type: ignore
from tensorflow.keras.layers import Input,InputLayer,Dense,Lambda  # type: ignore

#now we get parser (we have the function because we want to make it easy for import in other files)
def DQN_args_parser():
    parser=argparse.ArgumentParser(prog="Test_DQN")
    parser.add_argument("--env",type=str,default="Environment")
    parser.add_argument("--learning_rate",type=float,default=0.005)
    parser.add_argument("--batch_size",type=int,default=256)
    parser.add_argument("--gamma",type=float,default=0.95)
    parser.add_argument("--eps",type=float,default=1.0)
    parser.add_argument("--eps_decay",type=float,default=0.99999995)
    parser.add_argument("--eps_min",type=float,default=0.01)
    parser.add_argument("--logdirectory",type=str,default="logs")
    # return parser.parse_args(), parser
    return parser

# args,parser = DQN_args_parser()
parser = DQN_args_parser()
args = parser.parse_args([]) #added

#prepare for the tensorboard launch
logdir=os.path.join(args.logdirectory,parser.prog,args.env,datetime.now().strftime("%Y%m%d-%H%M%S"))
print(f"Saving training logs to:{logdir}")
writer=tf.summary.create_file_writer(logdir)

#Block 2: the replay_buffer that stores the past experiences
# (each experience is of 5 elements [state,action,reward,next_state,done])
class Replay_buffer:
    """
    1)initialization
    2)store(when adding a new experience to the replay_buffer)
    3)sample(sample an experience with the 5 elements)
    4)size(get the size of the replay buffer)
    """
    
    def __init__(self,args,capacity=10000):
        self.args=args
        self.buffer = deque(maxlen=capacity)
    
    #store an experience with 5 elements [state,action,reward,next_state,done]
    def store(self,state,action,reward,next_state,done):
        self.buffer.append([state,action,reward,next_state,done])
    
    #get a sample of a past experience(also of the 5 elements above)
    #since state is the major things that might vary for different samples, we only need to reshape states & next_states
    def sample(self):
        #random sample from the replay-buffer samples of batch size
        sample = random.sample(self.buffer,self.args.batch_size)
        #get the unzipped samples for each of the 5 elements
        states, actions, rewards, next_states, done=map(np.asarray,zip(*sample))
        
        #now we reshape the array of states and next_states
        states = np.array(states).reshape(self.args.batch_size,-1)
        next_states= np.array(next_states).reshape(self.args.batch_size,-1)
        return states,actions,rewards,next_states,done

    #get the size of the buffer
    def size(self):
        return len(self.buffer)

#Block 3: DQN model template
#note that this is defined as a template for the agent,
# so as the agent, we need to think about possible things an agent would do
class DQN:
    """
    as can see from this class, the neural network part - learns to predict Q-values for all actions given a state - which is separate from the policy search part;
    This is naturally different from the action-selection method - the policy;
    Thus the "get_action" part is independent from the neural network(also the other functions that are related with the NN)
    => this means, from "get_action", we have the action_selection policy;
    from the NN, we learns to estimate the Q value depend on actions given different states.
    
    1)initialization
    2)nn_model: build the DQN neural network model
    3)train: train the neural network model
    4)predict: use the model to predict with the state
    5)get_action(where the policy comes in): when the agent perform action(make decision with the brain policy)
    """
    
    #state & action dimensions(& also epsilon from argument)
    def __init__(self,args,state_dim,action_dim,model=None):
        self.state_dim=state_dim
        self.action_dim=action_dim
        #this is the only one that we need command line to define aside
        self.args=args
        self.epsilon=args.eps
        if (model is  not None):
            self.nnmodel=model
        else:
            self.nnmodel = None
            # directly built one model
            self.nnmodel=self.nn_model()
        
        self._predict_fn = tf.function(
            lambda x: self.nnmodel(x,training=False),
            input_signature=[tf.TensorSpec(shape=[None,self.state_dim],dtype=tf.float32)],
            reduce_retracing=True,
        )
        
        # self.nnmodel=model
        self.prev_action=None
        self.avoid_action=None
        self.avoid_action_extend=None
        self.count_avoid=0
    """
    DQN predicts: one Q value for each action given the state;
    thus for each pass of the model, we should look at the Q value of the action(output) given the state(input)
    """
    #based on this construction, we have:
    # the input shape being: states.shape=(batch_size, state_dim)
    # the output shape being: (batch_size, action_dim)
    #each row is the predicted Q-values for all actions in that state
    def nn_model(self):
        if (self.nnmodel is None):
            model=keras.models.Sequential([
                Input((self.state_dim,)),
                Dense(50,activation="relu"),
                Dense(156,activation="relu"),
                Dense(256,activation="relu"),
                Dense(256,activation="relu"),
                Dense(128,activation="relu"),
                Dense(64,activation="relu"),
                Dense(self.action_dim)
            ])
            model.compile(loss="mse",optimizer=keras.optimizers.SGD((self.args).learning_rate))
            # the following tries to run the model once with a fake input(so that we can initialize the weights)
            import tensorflow as tf
            dummy = tf.zeros((1,self.state_dim),dtype=tf.float32)
            _ = model(dummy,training=False)
            self.nnmodel = model
        return self.nnmodel
        #     return model
        # else:
        #     return self.nnmodel
    
    def train(self,states,targets):
        self.nn_model().fit(states,targets,epochs=1)
    
    # def predict(self,state):
    #     return self.nn_model().predict(state)
    
    # @tf.function(reduce_retracing=True, input_signature=[tf.TensorSpec(shape=[None,...],dtype=tf.float32)])
    # def _predict_fn(self,state):
    #     # return self.nn_model()(state,training=False)
    #     return self.nnmodel(state,training=False)
    
    def predict(self,state):
        state = np.asarray(state,dtype=np.float32)
        if state.ndim == 1:
            state = state.reshape(1,-1)
        # return self._predict_fn(state).numpy()
        
        # self.nn_model()
        x = tf.convert_to_tensor(state, dtype=tf.float32)
        return self._predict_fn(x).numpy()
    
    #now this part implement the policy: e.g. \epsilon-greedy
    def get_action(self,state,bounce,carry,count_bounce):
        #reshape the state: put it into a 1D array
        state = state.reshape(-1,self.state_dim)
        
        #get epsilon: and avoid the epislon get to below a certain baseline
        self.epsilon*=(self.args).eps_decay
        self.epsilon=max(self.epsilon,(self.args).eps_min)
        
        #get Q value
        Q_value=self.predict(state)[0]
        
        #start to implement the epsilon-greedy policy
        #this is when we can explore explored+unexplored actions RANDOMLY
        if np.random.random()<self.epsilon:
            rand_act=np.random.randint(0,self.action_dim)
            if(self.prev_action is None):#initialization
                self.prev_action=rand_act
            else: #cases after initialization
                if(bounce):
                    #if the agent has crashed into solids, avoid getting the same action as before
                    if(self.avoid_action is not None and self.count_avoid<=10):
                        while(self.prev_action == rand_act or rand_act==self.avoid_action or rand_act==self.avoid_action_extend):
                            rand_act=np.random.randint(0,self.action_dim)
                        self.count_avoid+=1
                    else:
                        self.avoid_action=None
                        self.avoid_action_extend=None
                        self.count_avoid=0
                
                    #if continueously bounce for 5 times(corresponds to the reward-10):
                    #start the "freeze mode": freeze the action for 5 steps afterwards
                    if(count_bounce==5):
                        self.avoid_action=self.prev_action
                        if(self.avoid_action==up):
                            self.avoid_action_extend=down
                        elif(self.avoid_action==down):
                            self.avoid_action_extend=up
                        elif(self.avoid_action==left):
                            self.avoid_action_extend=right
                        elif(self.avoid_action==right):
                            self.avoid_action_extend=left
                            
                    while(self.prev_action == rand_act):
                        if(carry and (self.prev_action != p_left or self.prev_action!=p_right)):
                            rand_act=np.random.randint(p_left,p_right+1)
                        else:
                            rand_act=np.random.randint(0,self.action_dim)
                # else: #this is not possible because if this is the case the agent would just go in the direct way
                #     #now this is not random at all...:
                #     # Indeed this is encouraging the agents to go forward if not bouncing
                #     rand_act=self.prev_action
                #[not realized]if the agent is going smoothly, we try to go in the same direction
                self.prev_action=rand_act
            return rand_act
        return np.argmax(Q_value)

#Block 4: Agent that uses Block 2 and Block 3
class Agent:
    """
    1)initialization
    2)update_target: independent modular function from "replay_experience"
    3)replay_experience: independent modular function from "update_target"
    4)train: train with writer while using "update_target" and "replay_experience"
    """
    #may have self identified number of agents as parameter
    def __init__(self,args,env,writer,continue_train=False,ThreeD=False,return4web=False):
        """Initialize
        1)envrionment
        2)state_dim,action_dim
        3)model, target_model: and make sure that both models start identiacally(which is why we would use update_target)
        4)create the replay_buffer
        """
        self.env=env
        self.writer=writer
        self.agents=[]
        self.replay_buffer=[]
        self.num_agents=self.env.num_agents
        self.ThreeD=ThreeD
        self.return4web=return4web
        
        #state is the observations,action is from the action_space
        self.state_dim=self.env.observation_space.shape[0]
        self.action_dim=self.env.action_space.n
        
        #now we initialize the models(model & the target model)
        self.args=args
        
        #the saved version for multiple agent case is not complete yet
        if(continue_train):
            try:
                for ith in range(self.num_agents):
                    model_name=f"current_Agent{ith}_model_v1.0.keras"
                    target_model_name=f"target_Agent{ith}_model_v1.0.keras"
                    
                    get_model=keras.models.load_model(model_name)
                    get_target=keras.models.load_model(target_model_name)
                    
                    self.model=DQN(args,self.state_dim,self.action_dim,get_model)
                    self.target_model=DQN(args,self.state_dim,self.action_dim,get_target)
                    
                    #store agents to the whole list of agents
                    self.agents.append((self.model,self.target_model))
        
                    #and we create the replay_buffer
                    self.replay_buffer.append(Replay_buffer(args))
            except Exception as e:
                raise RuntimeError("Fail to load previous Models: The Number of agents in envrionment does not match!!!")
        else:
            for i in range(self.num_agents):
                self.model=DQN(args,self.state_dim,self.action_dim)
                self.target_model=DQN(args,self.state_dim,self.action_dim)
                #and we what to make the two models identical(make sure the structures of the two models the same)
                self.update_target(init=True)
                
                #store agents to the whole list of agents
                self.agents.append((self.model,self.target_model))
        
                #and we create the replay_buffer
                self.replay_buffer.append(Replay_buffer(args))
        
    
    #this step update the target_model(Q_target generator) with the model(the one that generates common Q-value )
    def update_target(self,init=False):
        #first we get neural network of the two model
        # model=(self.model).nn_model()
        # target_model=(self.target_model).nn_model()
        for ith_agent,agent in zip(range(len(self.agents)),self.agents):
            model=agent[0].nn_model()
            target_model=agent[1].nn_model()
        
            #this is incompatible for the saving version with multiple agents
            if not init:
                #and we want to save the models so that later we can build our result on the trained models
                model_name=f"current_Agent{ith_agent}_model_v1.0.keras"
                target_model_name=f"target_Agent{ith_agent}_model_v1.0.keras"
                
                model.save(model_name)
                target_model.save(target_model_name)
            
            #then we update the weight of the target model to be the same the model
            weights=model.get_weights()
            target_model.set_weights(weights)
    
    #this step train the model
    def replay_experience(self,ith_agent):
        #train for 10 times(for each time, we first sample the experience and then train)
        for i in range(10):
            #first we get all the elements for sampling
            states,actions,rewards,next_states,done=self.replay_buffer[i].sample()
                
            #get the two Q-values needed for DQN iteration: Q-value & target Q-value
            target_model=self.agents[ith_agent][1]
            model=self.agents[ith_agent][0]
                    
            target_Q=target_model.predict(states)
            next_Q_values=model.predict(next_states).max(axis=1)
            
            #for the web purpose, we compute the loss
            loss=tf.reduce_mean(tf.squared(next_Q_values-target_Q))
                
            #now update the iteration using the formula
            target_Q[range((self.args).batch_size),actions]=rewards+(1-done)*(self.args).gamma*next_Q_values

            model.train(states,target_Q)
            return loss
    
    #"web_progress_callback" is a function:
    # this means if a web_progress_callback function is provided,
    # we call it and passing a dictionary with current training process
    def train(self,max_episodes=1000,web_progress_callback=None,stopped=None):
        #[debug] check if we get into agent's train method(whether we actually start to train the agent)
        print("[DEBUG] Inside Agent.train()")
        
        with self.writer.as_default():
            #in this writer, we would run the whole process max_episodes times
            for episode in range(max_episodes):
                #first initialize everything about this environmnet,including:
                # done, accumulated reward, count_done, observation
                done=False
                count_done=0
                episode_reward=0
                episode_time=0
                episode_loss=[]
                observations=self.env.reset()
                bounce_count=0
                bounce=False
                carry=False
                #this one is for decay time and corresponding epsilon
                decay_time=1
                updated_epsilon=1
                
                print("DEBUG: Starting training loop, done =", done)
                #first we want to run each episode until it is done
                while not done:
                    print("DEBUG: Inside training loop")
                    #first we check whether we are required to stop the training process
                    try:
                        print("DEBUG: stopped() returns", stopped())
                        if stopped():
                            print("[Train] Training stopped early by signal")
                            observations = self.env.reset()
                            self.writer.flush()
                            break
                    except Exception as e:
                        print("!!! Exception in stopped():", str(e))
                        break
                    
                    print("[Train] Entered training loop")
                    #now for each step we need to turn the lock package list to nothing(because this is just meaningful for each whole time step)
                    self.env.lock_packages=list()
                    current_model=None
                    #get the actions for the observations
                    # actions=int(self.model.get_action(observations,bounce,carry,bounce_count))
                    for i,replay_buffer_i,agent in zip(range(len(self.agents)),self.replay_buffer,self.agents):
                        model=agent[0]
                        actions=int(model.get_action(observations,bounce,carry,bounce_count))
                    
                        #then with the actions got from the states(observations), we step forward in this envrionment
                        next_observation, reward, done, info=self.env.step(i,actions)
                        bounce=info["bounce"]
                        bounce_count=info["count_bounce"]
                        carry=info["in_progress"]
                        
                        # Debug: Log step information
                        print(f"[TRAINING DEBUG] Agent {i} step - action: {actions}, reward: {reward}, done: {done}")
                        print(f"[TRAINING DEBUG] Agent {i} position: {self.env.agent_state[i] if i < len(self.env.agent_state) else 'N/A'}")
                        print(f"[TRAINING DEBUG] Environment count_done: {self.env.count_done}, time: {self.env.time}")
                        
                        #store the experience into the replay_buffer
                        replay_buffer_i.store(observations,actions,reward,next_observation,done)
                        
                        #update(for the next iteration in this while loop)
                        episode_reward+=reward
                        # episode_time=info["time"]
                        episode_time=self.env.time
                        average_reward=episode_reward/self.env.num_agents
                        if(not self.return4web):
                            self.env.render(reward=average_reward,count_time=episode_time,delivered=self.env.count_done,ThreeD_vis=self.ThreeD,return64web=self.return4web)
                        else:
                            img_base64=self.env.render(reward=episode_reward,count_time=episode_time,delivered=self.env.count_done,ThreeD_vis=self.ThreeD,return64web=self.return4web)
                        observations=next_observation #note that originally this line is after the render line
                    
                    updated_epsilon=self.args.eps*(self.args.eps_decay**decay_time)
                    decay_time+=1
                    
                    assert callable(web_progress_callback), "Callback is not callable"
                    print("[DEBUG] web_progress_callback =", web_progress_callback)
                        
                    #this is for the web progress callback(if we have the web thing)
                    try:
                        print("[DEBUG] Calling web_progress_callback")
                        if(web_progress_callback):
                            delivered=self.env.count_delivered()
                            current_epsilon=updated_epsilon
                            #the following is for time
                            time_text="0.0s"
                            if(episode_time<60):
                                time_text=f"{episode_time:.1f}s"
                            elif(episode_time>=60 and episode_time<3600):
                                min_time=int(episode_time//60)
                                remain_time=episode_time-min_time*60
                                if(remain_time==0):
                                    time_text=f"{min_time:.1f}m"
                                else:
                                    time_text=f"{min_time:.1f}m {remain_time}s"
                            else:
                                h_time=int(episode_time//3600)
                                remain_time=episode_time-h_time*3600
                                if(remain_time==0):
                                    time_text=f"{h_time:.1f}h"
                                else:
                                    min_time=int(remain_time//60)
                                    psecond_time=remain_time-min_time*60
                                    if(psecond_time==0):
                                        time_text=f"{h_time:.1f}h {min_time:.1f}m"
                                    else:
                                        time_text=f"{h_time:.1f}h {min_time:.1f}m {psecond_time:.1f}s"
                            reformat_reward=round(average_reward,2)
                            print(f"time_text: {time_text}")
                            
                            web_progress_callback({
                                "episode":episode+1,
                                "progress":((episode+1)/max_episodes)*100,
                                "reward":reformat_reward, #yes we use the average reward here in DQN
                                "delivered":delivered,
                                "image":img_base64,
                                "time": episode_time,
                                "loss": 0.0,
                                "epsilon": current_epsilon,
                                "actor_loss":0.0,
                                "critic_loss":0.0,
                                "time_text":time_text,
                                "current_algo":"dqn"
                            })
                    except Exception as e:
                        print("[Callback Error]", str(e))
                    
                    time.sleep(0.01)
                    #this is added specifically for the webapp, to prevent overwhelming
                    
                #after successfully running through the envrionment, we train the models
                for ith_agent,ith_replay_buffer in zip(range(self.replay_buffer),self.replay_buffer):
                    if(ith_replay_buffer.size()>=(self.args).batch_size):
                        loss=self.replay_experience(ith_agent)
                        episode_loss.append(loss.numpy() if hasattr(loss,"numpy") else float(0.0))
                        average_loss=(sum(episode_loss)/len(episode_loss) if episode_loss else 0.0)
                #in this step we get the target model weights
                self.update_target()
                
                try:
                    print("[DEBUG] Calling web_progress_callback")
                    if(web_progress_callback):
                        delivered=self.env.count_delivered()
                        current_epsilon=self.args.eps*(self.args.eps_decay**decay_time)
                        #the following is for time
                        time_text="0.0s"
                        if(episode_time<60):
                            time_text=f"{episode_time:.1f}s"
                        elif(episode_time>=60 and episode_time<3600):
                            min_time=int(episode_time//60)
                            remain_time=episode_time-min_time*60
                            if(remain_time==0):
                                time_text=f"{min_time:.1f}m"
                            else:
                                time_text=f"{min_time:.1f}m {remain_time}s"
                        else:
                            h_time=int(episode_time//3600)
                            remain_time=episode_time-h_time*3600
                            if(remain_time==0):
                                time_text=f"{h_time:.1f}h"
                            else:
                                min_time=int(remain_time//60)
                                psecond_time=remain_time-min_time*60
                                if(psecond_time==0):
                                    time_text=f"{h_time:.1f}h {min_time:.1f}m"
                                else:
                                    time_text=f"{h_time:.1f}h {min_time:.1f}m {psecond_time:.1f}s"
                        
                        reformat_reward=round(average_reward,2)
                        print(f"time_text: {time_text}")
                        
                        web_progress_callback({
                            "episode":episode+1,
                            "progress":((episode+1)/max_episodes)*100,
                            "reward":reformat_reward, #yes we use the average reward here in DQN
                            "delivered":delivered,
                            "image":img_base64,
                            "time": episode_time,
                            "loss": episode_loss,
                            "epsilon": current_epsilon,
                            "actor_loss":0.0,
                            "critic_loss":0.0,
                            "time_text":time_text,
                            "current_algo":"dqn"
                        })
                except Exception as e:
                    print("[Callback Error]", str(e))
                
                
                tf.summary.scalar("accumulated_episode_rewards",episode_reward,step=i)
                
                self.writer.flush()
                