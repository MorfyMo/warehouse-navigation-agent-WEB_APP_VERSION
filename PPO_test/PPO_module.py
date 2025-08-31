"""
This module is the PPO version of agents' searching.
"""
import tensorflow as tf
import time
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..','Env'))
import env_module # type: ignore
from env_module import Environment # type: ignore
from tensorflow import keras # type: ignore
from tensorflow.keras.layers import Input,InputLayer,Dense,Lambda # type: ignore
from datetime import datetime
import argparse
import gymnasium as gym # type: ignore
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from PPO_test.Running_Stats import Running_Stats # type: ignore

#add parser for future inputs(we have the function to make it easier for importion in other files)
def PPO_args_parser():
    parser = argparse.ArgumentParser(prog="PPO-program")
    parser.add_argument("--env",default="Environment")
    parser.add_argument("--update_freq",type=int,default=5)
    parser.add_argument("--epochs",type=int,default=3)
    parser.add_argument("--actor_lr",type=float,default=0.0005)
    parser.add_argument("--critic_lr",type=float,default=0.001)
    parser.add_argument("--clip_ratio",type=float,default=0.1)
    parser.add_argument("--gae_lambda",type=float,default=0.95)
    parser.add_argument("--gamma",type=float,default=0.99)
    parser.add_argument("--logdir",default="logs")
    # return parser.parse_args(), parser
    return parser
    
# args, parser=PPO_args_parser()
parser=PPO_args_parser()
args = parser.parse_args([]) #added

#initialize tensorboard
log_directory=os.path.join(args.logdir,parser.prog,args.env,datetime.now().strftime("%Y/%m/%d-%H:%M:%S"))
writer=tf.summary.create_file_writer(log_directory)

#now we have the three main classes for PPO: Actor, Critic, Agent(this is the actual class that we have the agent)
""" Three Blocks:
1. Actor: this class updates the policy(the one that deal with Policy)
    *[with the value from TD and advantage]
2. Critic: this class mainly deal with the Value Function(mainly with the value from TD learning)
    *Note: we do not use Q value here because value function is more stable in comparison & Q value is hard to get
    (since we had DQN class before, later we may consider merging DQN's Q value into these classes)
3. Agent: this class implements THE ACTUAL PPO process by using the the two classes before as modules(tools in other words)

The two tool classes Actor-Critic:
both have 4 major functions as backbone:
1. init
2. neural network models
3. compute loss
4. train
*For Actor, we have 2 more other functions AS FOR THE POLICY DEFINING ONE:
1). get_action
2). log_pdf: compute the probability density function for policies
"""
class Actor:
    """Initialize all the required things:
    1. action related: as required by all the policy related ones
        1) action dimension
        2) action bound
        3) standard deviation bound: as required for getting action
        *we get action based on distribution N~(mu,std)
    2. state dimension
    3. model: neural network model(as always required by those classes)
    4. optimizer
    """
    _logPDF_cache = {}
    
    def __init__(self,args,state_dim,action_dim,action_bound,std_bound):
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.upper_action_bound=action_bound
        self.model=self.nn_model()
        self.std_bound=std_bound
        self.args=args
        self.opt=tf.keras.optimizers.Adam(self.args.actor_lr)
        
        # @tf.function(reduce_retracing=True,input_signature=[tf.TensorSpec([None,self.action_dim],tf.float32),tf.TensorSpec([None],tf.int32)])
        # def _log_pdf_fn(logits,action):
        #     logits = tf.reshape(logits,[-1,self.action_dim])
        #     action = tf.reshape(action,[-1])
            
        #     log_prob=tf.nn.log_softmax(logits,axis=-1)
        #     log_policy_pdf = tf.squeeze(log_prob)
        #     log_policy_pdf = log_policy_pdf[action]
        #     return log_policy_pdf
        #     # return tf.gather(log_prob, action, batch_dims=1)
        
        # @staticmethod
        # @tf.function(reduce_retracing=True)
        
        fn = self._logPDF_cache.get(self.action_dim)
        if fn is None:
            def _log_pdf_fn(logits, action):
                return -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=action,logits=logits)

            fn = tf.function(
                _log_pdf_fn,
                input_signature=[
                    tf.TensorSpec(shape=[None, self.action_dim],dtype=tf.float32),
                    tf.TensorSpec(shape=[None],dtype=tf.int32),
                ],
            )
            self._logPDF_cache[self.action_dim]=fn
        self._log_pdf_fn = fn
    
    #this is the neural network:
    # through which we derive the mu and std for selecting action
    # from the given bunch of states
    def nn_model(self):
        state_input=Input((self.state_dim,))
        dense1=Dense(50,activation="relu")(state_input)
        dense2=Dense(100,activation="relu")(dense1)
        dense3=Dense(256,activation="relu")(dense2)
        dense4=Dense(256,activation="relu")(dense3)
        dense5=Dense(128,activation="relu")(dense4)
        dense6=Dense(64,activation="relu")(dense5)
        #output layers - mu output have to be transformed based on the bounds of action selection
        #This is obsoleted because this is just for continuous
        # out_mu=Dense(self.action_dim,activation="tanh")(dense6)
        # mu_output=Lambda(lambda x:x*self.action_bound)(out_mu)
        # std_output=Dense(self.action_dim,activation="softplus")(dense6)
        #For discrete we have direct one output
        # action_prob=Dense(self.action_dim,activation="softmax")(dense6)
        action_prob=Dense(self.action_dim,activation=None)(dense6)
        return keras.Model(inputs=[state_input],outputs=[action_prob])
    
    #we get the action from the pdf of the distribution
    # that is formed by mu and std returned from the NN model
    def get_action(self,state,training: bool=False):
        #first reshape the states to the shape allowed by our state dimension
        state = tf.convert_to_tensor(state,dtype=tf.float32)
        state=tf.reshape(state,(-1,self.state_dim))
        #now we get the mu and std required to form our distribution by predicting based on the states
        # action_prob=self.model.predict(state)
        # action_prob = self.model(state)
        
        #with the distribution derived, we sample the action from the distribution
        # logits=tf.math.log(action_prob)
        logits = self.model(state,training=training)
        action = tf.random.categorical(logits,num_samples=1)[:,0]
        # action=tf.random.categorical(logits,1)
        # action = int(action.numpy()[0][0])
        action = tf.cast(action,tf.int32)
        
        #since we don't need the continuous action space, we don't need the log_policy function
        log_policy = self.log_pdf(logits,action) #and thus with this action derived, we can compute the log pdf
        
        #ensure it return a scalar when batch is 1
        if tf.shape(action)[0]==1:
            log_policy = tf.squeeze(log_policy,axis=0)
            action = tf.squeeze(action,axis=0)
        
        return log_policy, action
    
    #this is the general function that computes the log PDF(probability density function): in specific, this is the function that computes the main formula
    #note that since our action space is discrete instead of continuous, we want to replace mu and std with the transformation of logits
    # def log_pdf(self,logits,action):
    #     #Note: these below are for continuous
    #     # std = tf.clip_by_value(std,self.std_bound[0],self.std_bound[1])
    #     # var = std**2
    #     # log_policy_pdf=-0.5*((action-mu)**2/var)-0.5*tf.math.log(2*np.pi*var)
        
    #     #Discrete version for our Environment:
    #     log_prob=tf.nn.log_softmax(logits)
    #     log_policy_pdf = tf.squeeze(log_prob)
    #     log_policy_pdf = log_policy_pdf[action]
        
    #     return log_policy_pdf
    
    # @tf.function(reduce_retracing=True)
    def log_pdf(self,logits,action):
        logits = tf.convert_to_tensor(logits,tf.float32)
        action = tf.convert_to_tensor(action, tf.int32)
        
        logits = tf.ensure_shape(logits,(None,self.action_dim))
        action = tf.ensure_shape(action,(None,))
        
        if logits.shape.rank == 1:
            logits = tf.reshape(logits,(1,-1))
        elif logits.shape.rank == 2 and logits.shape[-1] != self.action_dim:
            logits = tf.reshape(logits,(-1,self.action_dim))
            
        if action.shape.rank == 0:
            action = tf.reshape(action,(1,))
            
        return self._log_pdf_fn(logits,action)
        
    #in specific, this is the function that computes the main formula
    def compute_loss(self,log_old_policy,log_new_policy,actions,gaes): #this is the actual one that computes GAE and the reward after clip
        #first compute the ratio with the policies
        ratio = tf.exp(log_new_policy-log_old_policy)
        gaes=tf.stop_gradient(gaes) #to prevent tensorflow backpropagate through the whole graph and go through Critic's part(which was used to compute GAE as one step)
        
        clipped_ratio=tf.clip_by_value(ratio,1.0-self.args.clip_ratio,1.0+self.args.clip_ratio)
        surrogate=-tf.minimum(ratio*gaes,clipped_ratio*gaes) #in tensorflow minimum gets the lowest value, but we wants to maximize the min value, so we addes "-"
        return tf.reduce_mean(surrogate) #this averages all the values - which is the last step in Policy Gradient
    
    def train(self,log_old_policy,states,actions,gaes):
        with tf.GradientTape() as tape:
            action_prob=self.model(states,training=True)
            logits,actions=self.get_action(states)
            log_new_policy=self.log_pdf(logits,actions)
            loss=self.compute_loss(log_old_policy,log_new_policy,actions,gaes)
        grads=tape.gradient(loss,self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads,self.model.trainable_variables))
        return loss
    
#note that since PPO mainly uses TD 1 step - the value function only - we mostly just need V(s)
class Critic:
    def __init__(self,args,state_dim):
        self.state_dim=state_dim
        #note that this is only for the not-saved version;
        #*we need to add the other lines for the saving version later
        self.model=self.nn_model()
        self.args=args
        self.opt=tf.keras.optimizers.Adam(self.args.critic_lr)
    
    def nn_model(self): #note that this model is mainly based on V(s), which means we would return one thing based on one state(1-1 relation)
        model=keras.models.Sequential([
            Input((self.state_dim,)),
            Dense(50,activation="relu"),
            Dense(156,activation="relu"),
            Dense(256,activation="relu"),
            Dense(256,activation="relu"),
            Dense(128,activation="relu"),
            Dense(64,activation="relu"),
            Dense(1,activation="linear")
        ])
        return model
    
    def compute_loss(self,td_target,v_pred):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_target,v_pred)
    
    def train(self,states,td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states,training=True)
            assert v_pred.shape==td_targets.shape #this step ensures that the condition is valid
            stop_gradient =tf.stop_gradient(td_targets) #prevent unwanted results from flowing back through the td_targets(we don't want the target to change just because this certain part is being trained)
            loss = self.compute_loss(td_targets,stop_gradient)
        grads=tape.gradient(loss,self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads,self.model.trainable_variables))
        return loss

#now this class actually uses the Actor and Critic models and implements the whole process to updating policies and going through the whole environment
"""3 functions:
1. init: initialize everythings
2. gae_target: find the target through GAE(generalized advantage estimate)
3. train
"""
class Agent:
    """initialize all the things - Mainly Actor and Critic of the agent:
    1) get and store all the necessary elements
    2) define the optimizers of Actor and Critic(prepare for Actor and Critic)
    3) define and store the Actor and Critic object for this Agent
    """
    def __init__(self,args,env,writer,ThreeD=False,return4web=False):
        self.env=env
        self.state_dim=(self.env).observation_space.shape[0]
        #since our action space is discrete, we can't use the shape function as in Gym library
        self.action_dim=7
        #this below is the case for Box or others like Gym - the continuous case
        #self.action_dim=(self.env).action_space.shape[0]
        self.upper_action_bound=6 #let's try this, even though our envrironment is defined by ourselves
        #since this is discrete action space, we can just have action bound be 1(only continuous action space needs action_bound)
        self.std_bound=[1e-2,1.0]
        self.args=args
        self.writer=writer
        self.ThreeD=ThreeD
        self.return4web=return4web
        self.num_agents=self.env.num_agents
        self.Agents=[] #the agents are stored with the ith agent in the form "(actor_model,critic_model)"
        
        #optimizers of actor and critic(These two are saved for emergency - actually they are not used so much in this class as we can see)
        self.actor_opt=tf.keras.optimizers.Adam(self.args.actor_lr)
        self.critic_opt=tf.keras.optimizers.Adam(self.args.critic_lr)
        
        #actor and critic of agent
        for ith_agent in range(self.num_agents):
            actor=Actor(self.args,self.state_dim,self.action_dim,self.upper_action_bound,self.std_bound)
            critic=Critic(self.args,self.state_dim)
            #now we store the ith_agent's actor and critic models
            self.Agents.append((actor,critic))
    
    """This function computes GAE with formula:
    since the formula is recursive, we need to work backward
    1) first we get to the end of the training(go forward to assign forward value with the next_v_value)
    2) Then we work backward to actually compute with the GAE FORMULA
    RETURN:
    1) gae
    2) gae_cumulative
    """
    def gae_target(self,rewards,v_values,next_v_value,done):
        #first prepare everything we need - to initialize all the elements
        #note that we make the shape the same as reward because:
        # we need to compute the advantage per timestep, and we have exactly one reward per time step
        n_step_targets=np.zero_like(rewards)
        gae = np.zero_like(rewards)
        #now define the cumulative gae value and the forward value
        gae_cumulative=0 #GAE cumulative: Sigma_{k+1}
        forward_val=0 #Torward value: V_{k+1}
        
        #first we get to the end of the training to get the forward value at the end of the training:
        # (just in the same way as getting to the end of a linked list)
        if not done: #since the end forward value is the end of the training, we have the loop terminate by "done"
            forward_val=next_v_value
        
        #then we backpropagate to compute GAE formula(since this formula is a recursive function)
        for k in reversed(range(0,len(rewards))):
            #first we compute sigma
            sigma = rewards[k]+self.args.gamma*forward_val-v_values[k]
            
            #Then we get gae cumulative at this step(this single step)& Store it
            gae_cumulative=sigma+self.args.gamma*self.args.gae_lambda*gae_cumulative
            gae[k]=gae_cumulative #yes we store the gae_cumulative at k position(since it is running backward)
            
            #eventually we compute the target return: assign(set) the potential next value in the GAE computation
            forward_val=v_values[k]
            n_step_targets[k]=gae[k]+v_values[k]
        return gae, n_step_targets

    def stop_request(self,status):
        if(status=="stopped"):
            return True
        else:
            return False

    #now train the Agent
    """Utilize Actor and Critic classes defined before on training process:
    1) use the Actor class:
            this is used to fill the state,action,reward,old_policy batches
    2) use the Critic class
            this one applies the Critic model to each element inside the filled batches
            GOAL: compute GAE & td_target
    3) train with actor_loss & critic_loss
    *4 main elements: states, actions, rewards, old_policies
    """
    #"web_progress_callback" is function:
    # this means that if a "web_progress_callback" is provided
    # we call it and passing a dictionary with the current training process
    def train(self,max_episodes=1000,web_progress_callback=None,stopped=None): #stopped here is a function
        print("[DEBUG] Inside Agent.train()")

        with self.writer.as_default():
            reward_stats=Running_Stats()
            for episode in range(max_episodes):
                #first initialize all the batches & key elements
                #initialize key elements
                episode_reward,done=0,False
                episode_time=0
                state=self.env.reset()
                
                #now we initialize all the batches
                #For multi-agents, we can have 2D lists with each batch be in the form of "[[agent1_v1,agent2_v1,...],[agent1_v2,agent2_v2,...]]"
                state_batch=[]
                action_batch=[]
                reward_batch=[]
                old_policy_batch=[]
                agents_gaes=[]
                agents_td_targets=[]
                
                #Step 1. We use the Actor class: fill all the necessary batches with Actor's get action
                while not done:
                    #first we check whether we are required to stop the training process
                    if(stopped is not None):
                        if(stopped()):
                            observations=self.env.reset()
                            self.writer.flush()
                            break
                    
                    print("[Train] Entered training loop")
                    #For multi-agents, we have first,second,etc states for multiple agents stored with multiple agents' same state in one 1D array
                    state_for_agents=[]
                    action_for_agents=[]
                    next_state_agents=[]
                    reward_agents=[]
                    log_old_policy_agents=[]
                    
                    #we add these lists here to make sure we can compute the current loss here(optional_web)
                    actor_loss=[]
                    critic_loss=[]
                    
                    #now for each step we need to turn the lock package list to nothing(because this is just meaningful for each whole time step)
                    self.env.lock_packages=list()
                    
                    for ith_agent in range(self.num_agents):
                        #get the 4+1(the other one is the next_state => for backward purpose as known before)
                        actor_model=self.Agents[ith_agent][0]
                        log_old_policy,action=actor_model.get_action(state)
                        
                        # this below block is added to specifically address the action scalar problem(for tensorflow problem in webapp)
                        try:
                            action_scalar = int(action)
                        except Exception:
                            import numpy as np
                            try:
                                import tensorflow as tf
                                if tf.is_tensor(action):
                                    action_scalar = int(action.numpy().reshape(-1)[0])
                                else:
                                    action_scalar = int(np.asarray(action).reshape(-1)[0])
                            except Exception:
                                action_scalar = int(np.asarray(action).reshape(-1)[0])
                        
                        # next_state,reward,done,info=(self.env).step(ith_agent,action)
                        next_state,reward,done,info=(self.env).step(ith_agent,action_scalar)
                        
                        #temporarily we put the render here
                        episode_time=self.env.time
                        if(not self.return4web):
                            self.env.render(reward=episode_reward,count_time=episode_time,delivered=self.env.count_done,ThreeD_vis=self.ThreeD,return64web=self.return4web)
                        else:
                            img_base64=self.env.render(reward=episode_reward,count_time=episode_time,delivered=self.env.count_done,ThreeD_vis=self.ThreeD,return64web=self.return4web)
                        
                        #now we reshape the 5 elements to acceptable shapes
                        state = np.reshape(state,[-1,self.state_dim])
                        # action = np.reshape(action,[-1,self.action_dim])
                        action=tf.one_hot(action, depth=self.action_dim)
                        next_state = np.reshape(next_state,[-1,self.state_dim])
                        reward = np.reshape(reward,[1,1])
                        log_old_policy=np.reshape(log_old_policy,[-1,self.action_dim])

                        #In order to normalize the reward, we use the running stats created before to update
                        reward_stats.update(reward)
                        normalized_reward = (reward-reward_stats.get_mean())/(reward_stats.get_std()+1e-8)
                        episode_reward+=float(normalized_reward)

                        #since we want to have multi-agents, we also need to add those things to the 1D array that stores the same state index for different agents
                        state_for_agents.append(state)
                        action_for_agents.append(action)
                        next_state_agents.append(next_state)
                        reward_agents.append(normalized_reward)
                        log_old_policy_agents.append(log_old_policy)
                        
                        # #compute the loss(optional_web)
                        # actorModel=self.Agents[ith_agent][0]
                        # criticModel=self.Agents[ith_agent][1]
                        # #compute in advance the correspond value
                        # v_value=critic_model.model.predict(state)
                        # next_v_value=critic_model.model.predict(next_state)
                        # gae, td_target=self.gae_target(normalized_reward,v_value,next_v_value,done)
                        # #train the Actor one
                        # actor_loss_i=actorModel.train(log_old_policy,state,action,gae)
                        # actor_loss.append(actor_loss_i)
                        # #train the Critic one
                        # critic_loss_i=criticModel.train(state,td_target)
                        # critic_loss.append(critic_loss_i)
                    
                    #Finally, we add those reshaped gotten elements to the batch
                    state_batch.append(state_for_agents)
                    action_batch.append(action_for_agents)
                    reward_batch.append(reward_agents) #note that here we normalize the reward(to prevent exploding/vanishing gradient)
                    old_policy_batch.append(log_old_policy_agents)
                    
                    #this is for the web progress callback(if we are having the web thing)
                    try:
                        print("[DEBUG] Calling web_progress_callback")
                        if web_progress_callback:
                            delivered=self.env.count_delivered()
                            #this following block is for time
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
                            reformat_reward=round(episode_reward,2)
                            print(f"time_text: {time_text}")
                            
                            # mean_actor_loss=(sum(actor_loss[0])/len(actor_loss)) if actor_loss else 0.0
                            # mean_critic_loss=(sum(critic_loss[0])/len(critic_loss)) if critic_loss else 0.0
                            web_progress_callback({
                                "episode":episode+1,
                                "progress":((episode+1)/max_episodes)*100,
                                "reward": reformat_reward,
                                "delivered": delivered,
                                "image": img_base64,
                                "time": episode_time,
                                "loss": 0.0,
                                "epsilon": 0.0,
                                "actor_loss": 0.0,
                                "critic_loss": 0.0,
                                "time_text": time_text,
                                "current_algo":"ppo"
                            })
                    except Exception as e:
                        print("[Callback Error]", str(e))
                    
                    time.sleep(0.01)
                    #this is added specifically for the webapp, to prevent overwhelming
                
                #Step 2. We use the Critic class: apply Critic's model to compute the Value for each element inside the batches & GAE
                #GOAL: gae, td_target
                if len(state_batch)>=self.args.update_freq or done:
                    #this checks whether the previous done block ends right OR it stops because of real reason like the batch size cannot satisfy the requirement
                    
                    #Now, since we want to alter this to multi-agents, we also have to change this in some ways.
                    #1. First we want to group all the states for the same agent together - where zip function comes in
                    state_batch=zip(state_batch)
                    action_batch=zip(action_batch)
                    reward_batch=zip(reward_batch)
                    old_policy_batch=zip(old_policy_batch)
                    #yes since in the previous case we don't have next_state in batches(we only have one next_state), we don't add next_state to anywhere
                    
                    #1. First we DR(降维打击) all the batches to 1D(not sure if we need to alter this for multi-agents, we just put it in this way)
                    states=np.array([state.squeeze() for state in state_batch])
                    actions=np.array([action.squeeze() for action in action_batch])
                    # next_states=np.array([next_state.squeeze() for next_state in next_state])
                    rewards=np.array([reward.squeeze() for reward in reward_batch])
                    old_policies=np.array([old_pi.squeeze() for old_pi in old_policy_batch])
                    
                    #2. Next we get the needed v_value & next_v_value based on the Critic model's predict function
                    #since we are changeing this to multi-agents, we append the results to lists
                    for ith,ith_agent in range(self.num_agents),self.Agents:
                        critic_model=ith_agent[1]
                        #multi-agents specific
                        v_values=critic_model.model.predict(states[ith])
                        next_v_values=critic_model.model.predict(next_state[ith])
                    
                        gaes, td_targets=self.gae_target(rewards,v_values,next_v_values,done)
                        agents_gaes.append(gaes)
                        agents_td_targets.append(td_targets)
                
                #Finally, we train the Actor-Critic models: compute the losses
                #since we are using multi-agents, we would just use these 2 lists as temporary lists
                actor_losses=[]
                critic_losses=[]
                agents_losses=[] #this is for multi-agents: we would store with each element in the form of "(mean_actor_losses,mean_critic_losses)"
                for ith,ith_agent in range(self.num_agents),self.Agents:
                    actor_model=ith_agent[0]
                    critic_model=ith_agent[1]
                    for epoch in range(self.args.epochs):
                        #train the Actor one
                        actor_loss=actor_model.train(old_policies[ith],states[ith],actions[ith],gaes[ith])
                        actor_losses.append(actor_loss)
                        
                        #train the Critic one
                        critic_loss=critic_model.train(states[ith],td_targets[ith])
                        critic_losses.append(critic_loss)
                    
                    #now we can compute the mean and store them
                    mean_actor_loss_ith=np.mean(actor_losses)
                    mean_critic_loss_ith=np.mean(critic_losses)
                    
                    #write the losses to the scaler; note that we have to take the mean of the losses
                    tf.summary.scalar(f"actor_losses of {ith+1}th agent",np.mean(actor_losses),step=epoch)
                    tf.summary.scalar(f"critic_losses of {ith+1}th agent",np.mean(critic_losses),step=epoch)

                    #eventually we add the two losses into the agent loss large list
                    agents_losses.append((mean_actor_loss_ith,mean_critic_loss_ith))
                    
                
                try:
                    print("[DEBUG] Calling web_progress_callback")
                    if web_progress_callback:
                        delivered=self.env.count_delivered()
                        # mean_actor_loss=(sum(agents_losses[0])/len(agents_losses)) if agents_losses else 0.0
                        # mean_critic_loss=(sum(agents_losses[1])/len(agents_losses)) if agents_losses else 0.0
                        mean_actor_loss=list(agents_losses[i][0] for i in len(agents_losses))
                        mean_critic_loss=list(agents_losses[i][1] for i in len(agents_losses))
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
                        reformat_reward=round(episode_reward,2)
                        print(f"time_text: {time_text}")
                        
                        web_progress_callback({
                            "episode":episode+1,
                            "progress":((episode+1)/max_episodes)*100,
                            "reward": reformat_reward,
                            "delivered": delivered,
                            "image": img_base64,
                            "time": episode_time,
                            "loss": 0.0,
                            "epsilon": 0.0,
                            "actor_loss": mean_actor_loss,
                            "critic_loss": mean_critic_loss,
                            "time_text":time_text,
                            "current_algo":"ppo"
                        })
                except Exception as e:
                    print("[Callback Error]", str(e))
                
                #reset all the batches to empty
                state_batch=[]
                action_batch=[]
                reward_batch=[]
                old_policy_batch=[]
                agents_gaes=[]
                agents_td_targets=[]
                #update reward and state
                for ith in range(self.num_agents):
                    episode_reward+=reward[ith][0]
                    state=next_state[ith]
                
                    print(f"Episode#{episode} of {ith}th Agent Reward:{episode_reward}")
                    tf.summary.scalar(f"episode_reward_{ith}th_agent",episode_reward,step=episode)
                    
                self.writer.flush()