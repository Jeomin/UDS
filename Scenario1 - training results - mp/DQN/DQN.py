# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 22:10:33 2023

@author: chongtm
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 20:10:48 2022

@author: chong
"""
import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

class DQN:
    def __init__(self,params):
        tf.compat.v1.disable_eager_execution()
        self.params=params
        self.action_table=pd.read_csv('./DQN_action_table.csv').values[:,1:]
        
        self.model=self._build_net()
        self.model.compile(loss='mse', optimizer=Adam(1e-3))
        self.target_model=self._build_net()
        
        self.max_state=np.array([2.01026872e+00, 4.22421662e+00, 3.25000000e+00, 3.35000000e+00,
               2.00000000e+00, 2.21000000e+00, 1.96000000e+00, 2.05000000e+00,
               1.28939237e+03, 1.38030260e+03, 8.02888818e+02, 2.15337375e+03,
               6.83000000e+02, 6.83000000e+02, 1.00000000e+03, 1.00000000e+03,
               1.57454046e+03, 2.79013935e+02])
        self.min_state=np.array([ 4.98142986e-01,  9.87964095e-01,  1.37786022e-01,  0.00000000e+00,
                4.00174015e-02,  0.00000000e+00,  2.13572798e-04,  4.85072906e-03,
                1.08492137e+01,  0.00000000e+00, -5.03835243e+02,  9.27032212e-02,
                0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                1.11917992e+00,  0.00000000e+00])
        
    def _build_net(self):
        #DQN
        #eval net
        self.s = layers.Input(shape=self.params['state_dim'],name='s_input')
        V_prev = self.s
        for i in np.arange(self.params['evalnet_layer_V']):
            V_prev=layers.Dense(self.params['evalnet_V'][i]['num'], activation='relu', name='evalnet_V'+str(i))(V_prev)
        self.eval_out=layers.Dense(self.params['action_dim'], activation='linear', name='evalnet_out_V')(V_prev)
        
        model=models.Model(inputs=[self.s],outputs=self.eval_out)
        return model
    
    def load_model(self,file):
        self.model.load_weights(file)
    


def sample_action(state,model,train_log):
    #input state, output action
    # state归一化
    state=((np.array([state])-model.min_state)/(model.max_state-model.min_state)).reshape(1, -1)
    if train_log:
        #epsilon greedy
        pa = np.random.uniform()
        if pa > model.params['epsilon']:
            action_value = model.model.predict(state)
            action = np.argmax(action_value)
        else:
            action = np.random.randint(model.params['action_dim'])
    else:
        action_value = model.model.predict(state)
        action = np.argmax(action_value)
    return action

    
def process_batch(memory_buffer,model,batch):
     # 从经验池中随机采样一个batch
    #data = random.sample(self.memory_buffer, batch)
    data=[]
    for i in range(batch):
        data.append(memory_buffer[-i])
    # 生成Q_target
    states = np.array([d[0] for d in data]).squeeze()
    next_states = np.array([d[3] for d in data]).squeeze()
    y = model.model.predict(states)
    q = model.target_model.predict(next_states)

    for i, (_, action, reward, _, done) in enumerate(data):
        target = reward
        if not done:
            target += model.params['gamma'] * np.amax(q[i])
        y[i][action] = target
    return states, y 


def remember(memory_buffer, state, action, reward, next_state, done):
    item = (state, action, reward, next_state, done)
    memory_buffer.append(item)
    return memory_buffer    
        