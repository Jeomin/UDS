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
            action: 选择的动作
            class_id: 选择的场景类别
        """
        try:
            # 找出概率最高的类别
            class_id = np.argmax(embedding)
            
            agent = self.agents[class_id]
            
            combined_state = np.concatenate([state, embedding])
            
            # 选择动作
            result = agent.choose_action(combined_state, train_mode)
            
            # 处理不同类型的返回值
            if isinstance(result, tuple) and len(result) >= 2:
                action, other = result[0], result[1]
            else:
                action, other = result, None
                
            return action, class_id
        except Exception as e:
            print(f"选择动作出错: {e}")
            # 返回随机动作作为备选
            action = [np.random.randint(2) for _ in range(self.agents[0].params['action_dim'])]
            return action, class_id
    
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
                
                # 提取训练数据
                states = []
                embeddings = []
                actions = []
                rewards = []
                next_states = []
                next_embeddings = []
                dones = []
                
                for i, item in enumerate(data):
                    state, embedding, action, reward, _, next_state, done = item if len(item) == 7 else (*item, None, None)
                    states.append(state)
                    embeddings.append(embedding)
                    actions.append(action)
                    rewards.append(reward)
                    
                    if next_state is not None:
                        next_states.append(next_state)
                        
                        # 为下一状态生成嵌入
                        if len(data) > i+1:
                            next_embeddings.append(data[i+1][1])  # 下一个数据的嵌入
                        else:
                            next_embeddings.append(embedding)  # 使用当前嵌入
                            
                        dones.append(done)
                
                # 训练agent
                agent = self.agents[class_id]
                loss = agent.train_on_batch(
                    states, embeddings, actions, rewards, 
                    next_states, next_embeddings, dones
                )
                
                losses[class_id] = loss
                print(f"  场景 {class_id} agent训练完成: 损失={loss:.4f}")
        
        return losses
    
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