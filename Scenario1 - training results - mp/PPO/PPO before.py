# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:03:30 2022

@author: chong
"""


import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.optimizers import Adam

from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import time


class PPO:
    def __init__(self,params,env):
        tf.compat.v1.disable_eager_execution()
        self.params=params
        self.memory_buffer = deque(maxlen=1000)
        self.env=env
        
        self.sc = layers.Input(shape=self.params['state_dim'],name='sc_input')
        self.sa = layers.Input(shape=self.params['state_dim'],name='sa_input')
        ## PG RL的action_dim是泵数量        
        
        self.critic = self._build_critic(self.sc)
        self.actor = self._build_actor(self.sa, True)
        
        self.policy_optimizer = Adam(learning_rate=self.params['policy_learning_rate'])
        self.value_optimizer = Adam(learning_rate=self.params['value_learning_rate'])
    
    
    #critic
    def _build_critic(self, s):
        V_prev = s
        for i in np.arange(self.params['evalnet_layer_V']):
            V_prev=layers.Dense(self.params['evalnet_V'][i]['num'], activation='relu', name='evalnet_V'+str(i))(V_prev)
        eval_out=layers.Dense(1, activation='linear', name='evalnet_out_V')(V_prev)
        model = models.Model(inputs=[s],outputs=tf.squeeze(eval_out))
        return model
    
    #actor
    def _build_actor(self, s,trainable):
        A_prev = s
        for i in np.arange(self.params['actornet_layer_A']):
            A_prev=layers.Dense(self.params['actornet_A'][i]['num'], activation='relu', trainable=trainable, name='actornet_A'+str(i))(A_prev)
        mu = layers.Dense(self.params['actornet_mu']['num'],activation='tanh', trainable=trainable)(A_prev)
        sigma = layers.Dense(self.params['actornet_sigma']['num'],activation='sigmoid', trainable=trainable)(A_prev)
        norm_dist = tf.distributions.Normal(loc=mu,scale=sigma)
        
        sample_op = tf.squeeze(pi.sample(1),axis=0)
        logist = layers.Dense(self.params['action_dim'], activation='sigmoid', name='actornet_A')(A_prev)
        model = models.Model(inputs=[s],outputs=logist)
        return model
        
    
    def choose_action(self,state,train_log):
        #input state, output action
        if train_log:
            #epsilon greedy
            pa = np.random.uniform()
            if pa < self.params['epsilon']:
                logits = np.random.random(self.params['action_dim']).reshape((1,-1)).astype(np.float32)
            else: 
                logits = self.actor.predict(np.array([state]))
        else:
            logits = self.actor.predict(np.array([state]))
        
        action = logits.copy()
        for i in range(action.shape[1]):
            if action[0][i]<0.5:
                action[0][i]=0
            else:
                action[0][i]=1
                
            #action[0][i]=0
        #return np.clip(action, -self.bound, self.bound)
        return logits,np.clip(action, self.params['bound_low'], self.params['bound_high'])
    
    def remember(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        self.memory_buffer.append(item)
        
    def process_batch(self, batch):
         # 从经验池中随机采样一个batch
        #data = random.sample(self.memory_buffer, batch)
        data=[]
        for i in range(batch):
            data.append(self.memory_buffer[-i])
        # 生成Q_target
        states = np.array([d[0] for d in data])
        next_states = np.array([d[3] for d in data])
        y = self.model.predict(states)
        q = self.target_model.predict(next_states)

        for i, (_, action, reward, _, done) in enumerate(data):
            target = reward
            if not done:
                target += self.params['gamma'] * np.amax(q[i])
            y[i][action] = target
        return states, y
    
    def discount_reward(self,states,rewards,snext):
        s = np.vstack([states,snext.reshape((-1,self.params['state_dim']))])
        q_values = self.critic.predict(s).flatten()
        target = rewards + self.params['gamma']*q_values[1:]
        target = target.reshape(-1,1)
        return target
            
        
    def update(self,states, logits, old_act, dr):
        
        print('start training')
        t1=time.time()
        ## Training
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
            adv = tf.cast(tf.convert_to_tensor(dr),dtype=tf.float32) - tf.reshape(self.critic(states),shape=(-1,1))
            log_all = tf.nn.log_softmax(self.actor(states))
            logprobability = tf.reduce_sum(logits * log_all, axis=1)
            
            log_all_old = tf.squeeze(tf.nn.log_softmax(old_act))
            logprobability_old = tf.reduce_sum(logits * log_all_old, axis=1)
              
            ratio = tf.exp(logprobability - logprobability_old)
            min_advantage = tf.cast(tf.where(
                    dr > 0,
                    (1 + self.params['clip_ratio']) * adv,
                    (1 - self.params['clip_ratio']) * adv,
                ),dtype=tf.float32)
            
            self.policy_loss = -tf.reduce_mean(tf.minimum(ratio * adv, min_advantage))
            policy_grads = tape.gradient(self.policy_loss, self.actor.trainable_variables)
            self.policy_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))
            
        with tf.GradientTape() as tape:
            adv = tf.cast(tf.convert_to_tensor(dr),dtype=tf.float32) - tf.reshape(self.critic(states),shape=(-1,1))
            self.critic_loss = tf.reduce_mean(tf.square(adv))
            critic_grads = tape.gradient(self.critic_loss, self.critic.trainable_variables)
            self.value_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        '''
        kl_log_all = tf.nn.log_softmax(self.actor.predict(states))
        logprobability = tf.reduce_sum(old_act * kl_log_all, axis=1)
        kl = tf.reduce_sum(tf.reduce_mean(logits - logprobability))
        return kl
        '''
        print('end training',time.time()-t1)
        
    def train(self,RainData):
        #sampling and upgrading
        history = {'episode': [], 'Batch_reward': [], 'Episode_reward': [], 'Loss': []}
        #self.critic.compile(loss='mse', optimizer=Adam(1e-3))
        #self.critic.compile(loss='mse', optimizer=Adam(1e-3))
        
        for j in range(self.params['training_step']):
            Q_value, Q_loss = 0, 0
            reward_sum = 0
            for i in range(self.params['num_rain']):
                
                #print('training step:',j,' sampling num:',i)
                #Sampling: each rainfall represent one round of sampling
                batch=0
                s = self.env.reset(RainData[i])
                done, batch = False, 0
                states_tem, actions_tem, rewards_tem, logits_tem, value_tem = [],[],[],[],[]
                while not done:
                    logits, action = self.choose_action(s,True)
                    snext,reward,flooding,CSO,done = self.env.step(action[0].tolist())
                    #collect s, a, reward, value, log,done
                    self.remember(s, action[0], logits, reward, snext, done)
                    s = snext
                    batch+=1
                    reward_sum += reward
                
                batch=0
                for _ in range(20):
                    states,actions,logits,rewards,snext = self.process_batch(batch)
                    dr = self.discount_reward(states,rewards,np.array(snext))
                    self.update(states, logits, actions, dr)
                
                    
            # 减小egreedy的epsilon参数。
            if self.params['epsilon'] >= self.params['ep_min'] and j % 3 == 0 and j!=0:
                self.params['epsilon'] *= self.params['ep_decay']
            
            history['episode'].append(j)
            history['Episode_reward'].append(reward_sum)
            #history['kl'].append(loss)
            print('Episode: {} | Episode reward: {} | e:{:.2f}'.format(j, reward_sum, self.params['epsilon']))
            self.critic.save_weights('./model/PPOcritic.h5')
            self.actor.save_weights('./model/PPOactor.h5')
            if j % 3 == 0:
                np.save('./Results/Train.npy',history)
            
        return history
        
        
    def load_model(self):
        self.critic.load_weights('./model/PPOcritic.h5')
        self.actor.load_weights('./model/PPOactor.h5')
        
    def test(self,rain):
        # simulation on given rainfall
        test_history = {'time':[] ,'state': [], 'action': [], 'reward': [], 'F':[], 'C':[]}
        s = self.env.reset(rain)
        done, t= False, 0
        test_history['time'].append(t)
        test_history['state'].append(s)
        while not done:
            logits, action = self.choose_action(s,False)
            snext,reward,F,C,done = self.env.step(action[0].tolist())
            s = snext
            t +=1
            
            test_history['time'].append(t)
            test_history['state'].append(s)
            test_history['action'].append(action)
            test_history['reward'].append(reward)
            test_history['F'].append(F)
            test_history['C'].append(C)
        
        return test_history