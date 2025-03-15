# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 20:39:01 2022

@author: chong
"""
import numpy as np
import SWMM_ENV
import DQN
#import Rainfall_data as RD
from collections import deque
import datetime
from tensorflow.keras.optimizers import Adam
#import matplotlib.pyplot as plt
from joblib import Parallel, delayed

env_params={
        'orf':'chaohu',
        'advance_seconds':300
    }
env=SWMM_ENV.SWMM_ENV(env_params)


raindata = np.load('training_raindata.npy').tolist()

agent_params={
    'state_dim':len(env.config['states']),
    'action_dim':2**len(env.config['action_assets']),
    'evalnet_layer_A':3,
    'evalnet_A':[{'num':128},{'num':128},{'num':128}],
    'evalnet_layer_V':3,
    'evalnet_V':[{'num':128},{'num':128},{'num':128}],
    'targetnet_layer_A':3,
    'targetnet_A':[{'num':128},{'num':128},{'num':128}],
    'targetnet_layer_V':3,
    'targetnet_V':[{'num':128},{'num':128},{'num':128}],
    'num_rain':50,
     
    'training_step':500,
    'gamma':0.3,
    'epsilon':1,
    'ep_min':1e-50,
    'ep_decay':0.05
}


model = DQN.DQN(agent_params)
#model.model.save_weights('./model/DQNagent.h5')
model.load_model('./model/DQNagent.h5')


###############################################################################
# Train
###############################################################################
    
def interact(i,ep):   
    env=SWMM_ENV.SWMM_ENV(env_params)
    tem_model = DQN.DQN(agent_params)
    tem_model.load_model('./model/DQNagent.h5')
    tem_model.params['epsilon']=ep
    s,a,r,snext,last_value = [],[],[],[],[]
    observation, episode_return, episode_length = env.reset(raindata[i],i,True), 0, 0
    
    done = False
    while not done:
        # Get the action, and take one step in the environment
        observation = np.array(observation).reshape(1, -1)
        snext.append(observation)
        action = DQN.sample_action(observation,tem_model,True)
        at = tem_model.action_table[int(action)].tolist()
        observation_new, reward, flooding,CSO,done = env.step(at)
        episode_return += reward
        episode_length += 1

        # Store obs, act, rew, v_t, logp_pi_t
        s.append(observation)
        a.append(action)
        r.append(reward)
        
        last_value.append(done)
        
        # Update the observation
        observation = observation_new
    
    return s,a,r,snext,last_value,episode_return,episode_length


Train = False
pa = True
if Train:
    # main training process   
    history = {'episode': [], 'Batch_reward': [], 'Episode_reward': [], 'Loss': []}
    
    # Iterate over the number of epochs
    for epoch in range(model.params['training_step']):
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0
        
        # Initialize the buffer
        buffer = deque(maxlen= int(len(raindata[0])*model.params['num_rain']))
        #buffer = Buffer.Buffer(model.params['state_dim'], int(len(raindata[0])*model.params['num_rain']))
        
        # Iterate over the steps of each epoch
        if pa:
            # Parallel method in joblib
            res = Parallel(n_jobs=10)(delayed(interact)(i,model.params['epsilon']) for i in range(model.params['num_rain'])) 
            
            for i in range(model.params['num_rain']):
                #s, a, r, vt, lo, lastvalue in buffer
                #buffer = deque(maxlen= res[i][6])
                for o,a,r,no,done, in zip(res[i][0],res[i][1],res[i][2],res[i][3],res[i][4]):
                    buffer = DQN.remember(buffer, o, a, r, no, done)
                    #buffer.store(o,a,r,no)
                sum_return += res[i][5]
                sum_length += res[i][6]
                num_episodes += 1
            
            # Get values from the buffer and update q value function
            X, y = DQN.process_batch(buffer,model,sum_length)
            loss = model.model.fit(X, y, epochs=20, verbose=0)
        
    
        
        #model.model.save_weights('./model/DQNagent_'+str(epoch)+'.h5')
        model.model.save_weights('./model/DQNagent.h5')
        
        # log training results
        history['episode'].append(epoch)
        history['Episode_reward'].append(sum_return)
        np.save('./Results/Train.npy',history)
        # reduce the epsilon egreedy and save training log
        if model.params['epsilon'] >= model.params['ep_min'] and epoch % 10 == 0:
            model.params['epsilon'] -= model.params['ep_decay']
            
              
        
        # Print mean return and length for each epoch
        print(
            f" Epoch: {epoch + 1}. Return: {sum_return}. Mean Length: {sum_length / num_episodes}"
        )
        
        np.save('./Results/Train1.npy',history)
    
###############################################################################
# end Train
###############################################################################


# test DQN agent
def test(model,rain,i):
    # simulation on given rainfall
    test_history = {'time':[] ,'state': [], 'action': [], 'reward': [], 'F':[], 'C':[]}
    observation = env.reset(rain,i,False)
    done, t= False, 0
    test_history['time'].append(t)
    test_history['state'].append(observation)
    while not done:
        observation = np.array(observation).reshape(1, -1)
        action = DQN.sample_action(observation,model,False)
        at=model.action_table[int(action)].tolist()
        observation_new,reward,F,C,done = env.step(at)
        observation = observation_new
        t += 1
        
        test_history['time'].append(t)
        test_history['state'].append(observation)
        test_history['action'].append(action)
        test_history['reward'].append(reward)
        test_history['F'].append(F)
        test_history['C'].append(C)
    
    return test_history


raindata = np.load('test_raindata.npy').tolist()
model.load_model('./model/DQNagent.h5')

for i in range(len(raindata)):
    test_his = test(model,raindata[i],i)
    np.save('./Results/'+str(i)+'.npy',test_his)




