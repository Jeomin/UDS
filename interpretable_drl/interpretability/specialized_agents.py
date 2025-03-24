# -*- coding: utf-8 -*-
"""
专门化agent管理器，管理多个针对不同场景的DRLagent
"""
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models

class SpecializedAgentManager:
    """专门化agent管理器，管理多个针对不同场景的DRLagent"""
    
    def __init__(self, env, num_classes, agent_class, base_params=None):
        """
        初始化专门化agent管理器
        
        Args:
            env: 环境对象
            num_classes: 场景类别数量
            agent_class: agent类（如DQN或PPO）
            base_params: 基础参数字典
        """
        self.env = env
        self.num_classes = num_classes
        self.agent_class = agent_class
        self.base_params = base_params or {}
        
        # 创建多个专门化agent
        self.agents = []
        for i in range(num_classes):
            # 为每个agent创建特定参数
            params = self.base_params.copy()
            # 扩展状态维度，包含嵌入
            params['state_dim'] = params.get('state_dim', 0) + num_classes
            # 不同agent可以有不同超参数
            params['class_id'] = i
            
            # 创建agent
            agent = agent_class(params, env)
            self.agents.append(agent)
    
    def choose_action(self, state, embedding, train_mode=False):
        """
        根据状态和embedding选择动作
        
        Args:
            state: 当前状态
            embedding: 策略嵌入向量
            train_mode: 是否处于训练模式
            
        Returns:
            action: 选择的动作 (可能是直接动作或(logits, action)对)
            class_id: 选择的场景类别
        """
        # 找出概率最高的类别
        class_id = np.argmax(embedding)
        
        # 确保class_id在有效范围内
        if class_id >= len(self.agents):
            class_id = 0  # 使用默认agent
            
        agent = self.agents[class_id]
        
        # 合并状态和embedding
        combined_state = np.concatenate([state, embedding])
        
        # 根据agent类型选择动作
        result = agent.choose_action(combined_state, train_mode)
        
        # 返回原始结果
        return result, class_id

    
    def train_agents(self, dataset_by_class):
        """
        训练各个专门化agent
        
        Args:
            dataset_by_class: 按类别整理的训练数据
            
        Returns:
            losses: 各agent的训练损失
        """
        losses = {}
        for class_id, data in dataset_by_class.items():
            if len(data) > 10:  # 确保有足够数据
                print(f"训练场景 {class_id} 的agent，数据量: {len(data)}")
                
                # 获取agent
                agent = self.agents[class_id]
                
                # 提取并准备训练数据
                states, embeddings, actions, rewards, logits, next_states = self._prepare_training_data(data)
                
                if hasattr(agent, 'update') and all(lg is not None for lg in logits):
                    # 合并状态和嵌入
                    combined_states = np.array([np.concatenate([s, e]) for s, e in zip(states, embeddings)])
                    
                    # 计算折扣回报
                    if len(next_states) > 0:
                        dr = agent.discount_reward(combined_states, np.array(rewards), next_states[-1])
                    else:
                        rewards_array = np.array(rewards)
                        dr = np.reshape(rewards_array, (rewards_array.size, 1))

                    policy_loss, _ = agent.update(combined_states, np.array(logits), np.array(actions), dr)
                    losses[class_id] = float(policy_loss) if policy_loss is not None else 0.0
                    print(f"  场景 {class_id} agent训练完成: 损失={losses[class_id]:.4f}")
                
                elif hasattr(agent, 'train_on_batch'):
                    try:
                        loss = agent.train_on_batch(states, embeddings, actions, rewards, next_states)
                        losses[class_id] = loss
                        print(f"  场景 {class_id} agent训练完成: 损失={loss:.4f}")
                    except Exception as e:
                        print(f"  使用train_on_batch训练场景 {class_id} agent出错: {e}")
                else:
                    print(f"  场景 {class_id} agent没有可用的批次训练方法")
        
        return losses

    def _prepare_training_data(self, data):
        """准备训练数据"""
        states = []
        embeddings = []
        actions = []
        rewards = []
        logits = []
        next_states = []
        
        for item in data:
            state, embedding, action, reward, _, next_state, _ = item if len(item) == 7 else (*item, None, None)
            
            states.append(state)
            embeddings.append(embedding)
            rewards.append(reward)

            if isinstance(action, tuple) and len(action) == 2:
                # (logits, action_values)
                act_logits, act_values = action
                logits.append(act_logits)
                actions.append(act_values)
            else:
                logits.append(np.zeros(7))  # 默认logits
                actions.append(action)
            
            if next_state is not None:
                next_states.append(next_state)
        
        return states, embeddings, actions, rewards, logits, next_states
    
    def save_models(self, output_dir):
        """
        保存所有agent模型
        
        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        for i, agent in enumerate(self.agents):
            agent_dir = os.path.join(output_dir, f"agent_{i}")
            os.makedirs(agent_dir, exist_ok=True)
            agent.save_model(agent_dir)
            print(f"保存agent {i} 到 {agent_dir}")
    
    def load_models(self, input_dir):
        """
        加载所有agent模型
        
        Args:
            input_dir: 输入目录
        """
        for i, agent in enumerate(self.agents):
            agent_dir = os.path.join(input_dir, f"agent_{i}")
            if os.path.exists(agent_dir):
                agent.load_model(agent_dir)
                print(f"加载agent {i} 成功")
            else:
                print(f"警告: agent {i} 的目录不存在 {agent_dir}")