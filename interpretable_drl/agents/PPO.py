# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:03:30 2022

@author: chong
"""


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
# import tensorflow_probability as tfp # 用于概率分布

from collections import deque
import numpy as np
import os

class PPO:

    def __init__(self, params, env):
        # tf.compat.v1.disable_eager_execution() # 如果使用 TF 2.x, 通常不需要这行
        self.params = params
        # self.memory_buffer = deque(maxlen=1000) # 内存缓冲区在外部管理，这里可以移除
        self.env = env

        # 获取状态和动作维度，确保从 params 获取
        self.state_dim = params.get('state_dim')
        self.action_dim = params.get('action_dim') # 这是泵的数量
        if self.state_dim is None or self.action_dim is None:
             raise ValueError("params 字典中必须包含 'state_dim' 和 'action_dim'")

        # --- 网络输入 ---
        # Critic 和 Actor 都接收 state 作为输入
        # 注意：在内生框架中，传入的 state 维度是 state_dim (原始状态 + embedding)
        self.critic_input = layers.Input(shape=(self.state_dim,), name='critic_input')
        self.actor_input = layers.Input(shape=(self.state_dim,), name='actor_input')

        # --- 构建网络 ---
        self.critic = self._build_critic(self.critic_input)
        self.actor = self._build_actor(self.actor_input, True) # Actor 输出 logits

        # --- 优化器 ---
        self.policy_optimizer = Adam(learning_rate=self.params.get('policy_learning_rate', 0.001))
        self.value_optimizer = Adam(learning_rate=self.params.get('value_learning_rate', 0.001))

        # --- PPO 超参数 ---
        self.gamma = self.params.get('gamma', 0.99) # 折扣因子
        self.clip_ratio = self.params.get('clip_ratio', 0.2) # PPO 裁剪比率
        # self.lam = self.params.get('lam', 0.95) # GAE lambda (如果使用 GAE)

        # --- 动作边界 (如果需要的话，虽然这里是 0/1) ---
        self.bound_low = self.params.get('bound_low', 0)
        self.bound_high = self.params.get('bound_high', 1)

    def _build_critic(self, s):
        """构建 Critic 网络 (估计状态价值 V(s))"""
        layer_nodes = self.params.get('evalnet_V', [{'num': 30}] * self.params.get('evalnet_layer_V', 3)) # 获取层配置或使用默认值
        x = s
        for i, layer_conf in enumerate(layer_nodes):
            x = layers.Dense(layer_conf['num'], activation='relu', name=f'critic_dense_{i}')(x)
        # 输出单个价值估计值
        output = layers.Dense(1, activation='linear', name='critic_output')(x)
        model = models.Model(inputs=s, outputs=output) # 输出形状 (batch, 1)
        return model

    def _build_actor(self, s, trainable):
        """构建 Actor 网络 (输出动作的 logits)"""
        layer_nodes = self.params.get('actornet_A', [{'num': 30}] * self.params.get('actornet_layer_A', 3)) # 获取层配置或使用默认值
        x = s
        for i, layer_conf in enumerate(layer_nodes):
            x = layers.Dense(layer_conf['num'], activation='relu', trainable=trainable, name=f'actor_dense_{i}')(x)
        # 输出每个独立二元动作的 logits
        logits = layers.Dense(self.action_dim, activation='linear', name='actor_logits')(x)
        model = models.Model(inputs=s, outputs=logits)
        return model

    def choose_action(self, state, train_mode=True):
        """
        根据当前状态选择动作，并计算对数概率。
        适用于独立的二元动作空间 (例如多个泵的开关)。

        Args:
            state: 当前状态 (numpy array, 形状应为 (state_dim,) 或 (1, state_dim))
            train_mode (bool): 是否处于训练模式。True: 采样动作；False: 确定性动作。

        Returns:
            tuple: (action, logp_action)
                   - action: 选择的动作 (numpy array, 形状 (action_dim,), 元素为 0 或 1)
                   - logp_action: 所选动作的对数概率 (标量)
        """
        # 确保 state 是二维的 (batch_size=1)
        if len(state.shape) == 1:
            state = np.expand_dims(state, axis=0)

        # 获取当前策略的 logits
        logits = self.actor(state) # 使用 tf.function 可能更快，但 predict 也可以

        # --- 方法 1: 使用 TensorFlow Probability ---
        # # 创建独立的伯努利分布
        # probs = tf.sigmoid(logits) # 转换 logits 为概率
        # action_distribution = tfp.distributions.Bernoulli(probs=probs, dtype=tf.int32)

        # if train_mode:
        #     # 训练时：从分布中采样动作
        #     action = action_distribution.sample()
        # else:
        #     # 评估时：采取概率最高的动作 (对于伯努利，即 > 0.5)
        #     action = tf.cast(probs > 0.5, dtype=tf.int32)

        # # 计算采样/选定动作的对数概率
        # # Bernoulli.log_prob 会为每个维度计算 logp，我们需要求和得到整个动作向量的 logp
        # logp_action = tf.reduce_sum(action_distribution.log_prob(action), axis=1)

        # --- 方法 2: 手动计算 (如果不想用 TFP) ---
        probs = tf.sigmoid(logits)
        if train_mode:
            # 采样: 为每个维度独立采样
            action = tf.cast(tf.random.uniform(shape=tf.shape(probs)) < probs, dtype=tf.int32)
        else:
            # 确定性动作
            action = tf.cast(probs > 0.5, dtype=tf.int32)
        
        # 手动计算 log 概率: logp = sum( a*log(p) + (1-a)*log(1-p) )
        # 注意数值稳定性，使用 log_sigmoid
        log_probs = tf.math.log_sigmoid(logits)
        log_one_minus_probs = tf.math.log_sigmoid(-logits) # log(1-sigmoid(x)) = log_sigmoid(-x)
        logp_action = tf.reduce_sum(
            tf.cast(action, tf.float32) * log_probs + (1.0 - tf.cast(action, tf.float32)) * log_one_minus_probs,
            axis=1
        )

        # 返回 numpy 数组/标量
        action_np = action.numpy().flatten() # 形状 (action_dim,)
        logp_action_np = logp_action.numpy()[0] # 标量

        # 注意：不再返回原始 logits，因为 logp 更有用
        return logp_action_np, action_np

    def calculate_logp(self, states, actions):
        """
        计算给定状态下，采取特定动作的对数概率 (根据当前 Actor 网络)。
        用于在 update 中计算 logp_new。

        Args:
            states: 状态 (tf.Tensor, float32)
            actions: 采取的动作 (tf.Tensor, int32 or float32, 0/1s)

        Returns:
            logp: 对数概率 (tf.Tensor, float32)
        """
        logits = self.actor(states)
        # 使用 sigmoid 交叉熵计算每个动作维度的负对数概率
        # labels 是实际采取的动作, logits 是当前网络的输出
        neg_logp_per_dim = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(actions, tf.float32),
            logits=logits
        )
        # 对所有动作维度求和得到整个动作向量的负对数概率
        neg_logp = tf.reduce_sum(neg_logp_per_dim, axis=1)
        return -neg_logp # 返回正的对数概率

    def update(self, states, actions, rewards, next_states, dones, logp_olds):
        """
        使用收集到的数据更新 Actor 和 Critic 网络 (修正版)。

        Args:
            states: 状态 (numpy array)
            actions: 采取的动作 (numpy array, 0/1s)
            rewards: 奖励 (numpy array)
            next_states: 下一个状态 (numpy array)
            dones: 是否终止 (numpy array, bool or int)
            logp_olds: 采取动作时的旧对数概率 (numpy array)
        """
        # --- 1. 计算 TD 目标和 TD 误差 (作为优势的代理) ---
        # 需要 Critic 对 states 和 next_states 的价值估计
        # 确保输入是正确的形状 (batch, feature_dim)
        states_tf = tf.cast(states, dtype=tf.float32)
        next_states_tf = tf.cast(next_states, dtype=tf.float32)
        rewards_tf = tf.cast(rewards, dtype=tf.float32)
        dones_tf = tf.cast(dones, dtype=tf.float32)
        logp_olds_tf = tf.cast(logp_olds, dtype=tf.float32)
        actions_tf = tf.cast(actions, dtype=tf.int32) # 确保动作为整数类型

        # 获取 V(s) 和 V(s_next)
        values = self.critic(states_tf)           # shape (batch, 1)
        values_next = self.critic(next_states_tf) # shape (batch, 1)

        # 如果是终止状态，V(s_next) 应为 0
        values_next = values_next * (1.0 - dones_tf) # dones 是 0 或 1

        # 计算 TD 目标: R + gamma * V(s_next)
        td_targets = rewards_tf + self.gamma * tf.squeeze(values_next) # squeeze -> (batch,)
        td_targets = tf.expand_dims(td_targets, axis=1) # 变回 (batch, 1)

        # 计算 TD 误差 (作为优势): TD Target - V(s)
        advantages = td_targets - values # shape (batch, 1)
        # 不需要 detach，因为 value loss 会处理 critic 的梯度

        # --- 2. 更新 Actor (策略损失) ---
        with tf.GradientTape() as tape:
            # 计算当前策略下采取旧动作的对数概率 logp_new
            logp_news = self.calculate_logp(states_tf, actions_tf) # shape (batch,)

            # 计算概率比 ratio = exp(logp_new - logp_old)
            # logp_olds_tf 也需要是 (batch,)
            ratio = tf.exp(logp_news - tf.squeeze(logp_olds_tf)) # squeeze logp_old -> (batch,)
            ratio = tf.expand_dims(ratio, axis=1) # 变回 (batch, 1) 以匹配 advantage

            # 计算 PPO 裁剪目标
            # adv 需要被 detach 吗？理论上 policy loss 不应影响 value function，但 TD error 本身依赖 V(s)
            # 通常做法是 detach advantage: adv_detached = tf.stop_gradient(advantages)
            adv_detached = tf.stop_gradient(advantages)

            # 未裁剪的目标: ratio * advantage
            unclipped_objective = ratio * adv_detached
            # 裁剪的目标: clip(ratio, 1-epsilon, 1+epsilon) * advantage
            clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            clipped_objective = clipped_ratio * adv_detached

            # PPO 策略损失 (取两者中较小的，然后求平均值的相反数)
            policy_loss = -tf.reduce_mean(tf.minimum(unclipped_objective, clipped_objective))

        # 计算并应用策略梯度
        policy_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

        # --- 3. 更新 Critic (价值损失) ---
        with tf.GradientTape() as tape:
            # Critic 的目标是拟合 TD Target
            current_values = self.critic(states_tf) # 重新计算以确保在 tape 中
            # 价值损失: (TD Target - V(s))^2 的均值
            value_loss = tf.reduce_mean(tf.square(td_targets - current_values))

        # 计算并应用价值梯度
        value_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.critic.trainable_variables))

        # 返回损失值用于监控
        return policy_loss.numpy(), value_loss.numpy()
        
    def train(self,RainData):
        #sampling and upgrading
        history = {'episode': [], 'Batch_reward': [], 'Episode_reward': [], 'Loss': []}
        #self.critic.compile(loss='mse', optimizer=Adam(1e-3))
        #self.critic.compile(loss='mse', optimizer=Adam(1e-3))
        
        for j in range(self.params['training_step']):
            reward_sum = 0
            count = 0
            for i in range(self.params['num_rain']):
                print('training step:',j,' sampling num:',i)
                #Sampling: each rainfall represent one round of sampling
                batch=0
                s = self.env.reset(RainData[i])
                done, batch = False, 0
                states_tem, actions_tem, rewards_tem, logits_tem, value_tem = [],[],[],[],[]
                while not done:
                    logits, action = self.choose_action(s,False)
                    snext,reward,flooding,CSO,done = self.env.step(action[0].tolist())
                    #collect s, a, reward, value, log,done
                    #self.remember(s, a, reward, value, logits_t, done)
                    value_tem.append(self.critic(np.array([s])))
                    states_tem.append(s)
                    actions_tem.append(action[0])
                    logits_tem.append(logits)
                    rewards_tem.append(reward)
                    s = snext
                    batch+=1
                    reward_sum += reward
                    
                states = np.array(states_tem)
                actions = np.array(actions_tem)
                logits = np.array(logits_tem)
                rewards = np.array(rewards_tem)
                dr = self.discount_reward(states,rewards,np.array(snext))
                    
                self.update(states, logits, actions, dr)
                    
                count += 1
                # 减小egreedy的epsilon参数。
                if self.params['epsilon'] >= self.params['ep_min']:
                    self.params['epsilon'] *= self.params['ep_decay']
                if i % 5 == 0:
                    history['episode'].append(i)
                    history['Episode_reward'].append(reward_sum)
                    #history['kl'].append(loss)
                    #print('Episode: {} | Episode reward: {} | kl: {:.3f} | e:{:.2f}'.format(i, reward_sum, kl, self.params['epsilon']))
            self.critic.save_weights('./model/PPOcritic_'+str(j)+'.h5')
            self.actor.save_weights('./model/PPOactor_'+str(j)+'.h5')
        return history
        
        
    def save_model(self, save_dir):
        """
        保存模型权重到指定目录
        
        Args:
            save_dir: 保存目录路径
        """
        os.makedirs(save_dir, exist_ok=True)
        critic_path = os.path.join(save_dir, 'PPOcritic.h5')
        actor_path = os.path.join(save_dir, 'PPOactor.h5')
        
        self.critic.save_weights(critic_path)
        self.actor.save_weights(actor_path)
        print(f"PPO模型已保存到 {save_dir}")
        
    def load_model(self, load_dir=None):
        """
        从指定目录加载模型权重
        
        Args:
            load_dir: 加载目录路径，如果为None则使用默认路径
        """
        if load_dir is None:
            # 使用默认路径
            critic_path = './model/PPOcritic.h5'
            actor_path = './model/PPOactor.h5'
        else:
            critic_path = os.path.join(load_dir, 'PPOcritic.h5')
            actor_path = os.path.join(load_dir, 'PPOactor.h5')
        
        try:
            self.critic.load_weights(critic_path)
            self.actor.load_weights(actor_path)
            print(f"PPO模型已加载自 {critic_path} 和 {actor_path}")
        except Exception as e:
            print(f"加载PPO模型失败: {e}")
        
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