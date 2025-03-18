# -*- coding: utf-8 -*-
"""
Interpretable agent class that adds explainability to RL agents
"""
from .base_agent import BaseAgent
import numpy as np
import datetime

class InterpretableAgent(BaseAgent):
    """
    Adds explainability to reinforcement learning agents
    """
    
    def __init__(self, base_agent, explainer, tree_model=None):
        """
        初始化可解释性agent
        
        Args:
            base_agent: 基础强化学习agent
            explainer: 解释器对象
            tree_model: 树形agent模型（可选）
        """
        self.base_agent = base_agent
        self.env = base_agent.env
        self.params = base_agent.params
        self.explainer = explainer
        self.tree_model = tree_model
        
        # 状态-动作-结果数据集
        self.state_action_result_dataset = []
        
    def choose_action(self, state, train_log=False):
        """
        选择动作并提供解释
        
        Args:
            state: 当前状态
            train_log (bool): 是否处于训练模式
            
        Returns:
            action: 选择的动作
            explanation (optional): 动作的解释
        """
        # 使用基础agent选择动作
        action = self.base_agent.choose_action(state, train_log)
        
        # 如果不在训练模式，生成解释
        if not train_log:
            # 如果有树模型，使用它来获取决策路径
            decision_path = None
            if self.tree_model:
                try:
                    decision_path, _ = self.tree_model.predict_with_path(state)
                except:
                    pass
            
            # 生成解释
            explanation = self.explainer.explain_action(state, action, decision_path)
            return action, explanation
            
        return action
    
    def train(self, rainfall_data):
        """
        训练agent并收集解释数据
        
        Args:
            rainfall_data: 训练用降雨数据
            
        Returns:
            history: 训练历史
        """
        # 训练基础agent
        history = self.base_agent.train(rainfall_data)
        
        # 收集用于解释的数据
        self._collect_explanation_data(rainfall_data[:10])  # 限制数量以避免内存问题
        
        # 如果有树模型，训练它
        if self.tree_model and self.state_action_result_dataset:
            states = np.array([item[0] for item in self.state_action_result_dataset])
            actions = np.array([item[1] for item in self.state_action_result_dataset])
            self.tree_model.train(states, actions)
        
        return history
    
    def _collect_explanation_data(self, rainfall_data):
        """
        收集用于解释的数据
        
        Args:
            rainfall_data: 降雨数据
        """
        for rain in rainfall_data:
            s = self.env.reset(rain)
            done = False
            
            while not done:
                a = self.base_agent.choose_action(s, False)
                action_value = None
                
                # 尝试获取动作值
                if hasattr(self.base_agent, 'get_action_value'):
                    try:
                        action_value = self.base_agent.get_action_value(s)
                    except:
                        pass
                
                # 处理不同类型的动作返回值
                if isinstance(a, tuple):
                    action = a[0]
                else:
                    action = a
                    
                s_next, reward, flooding, cso, done = self.env.step(action)
                
                # 存储状态、动作、动作值、洪水、CSO
                self.state_action_result_dataset.append(
                    (s, action, action_value, flooding, cso)
                )
                
                s = s_next
    
    def load_model(self, model_path=None):
        """
        加载模型
        
        Args:
            model_path: 模型路径（可选）
        """
        # 加载基础agent模型
        self.base_agent.load_model(model_path)
        
        # 加载树模型（如果有）
        if self.tree_model and model_path:
            try:
                tree_path = model_path.replace('.h5', '_tree.pkl')
                self.tree_model.load(tree_path)
            except Exception as e:
                print(f"无法加载树模型: {e}")
    
    def test(self, rainfall):
        """
        测试agent并生成解释
        
        Args:
            rainfall: 测试用降雨数据
            
        Returns:
            test_history: 测试历史，包括解释
        """
        # 使用基础agent测试
        test_history = self.base_agent.test(rainfall)
        
        # 为每个时间步添加解释
        explanations = []
        states = test_history.get('state', [])
        actions = test_history.get('action', [])
        
        for i in range(min(len(states), len(actions))):
            state = states[i]
            action = actions[i]
            
            # 获取决策路径
            decision_path = None
            if self.tree_model:
                try:
                    decision_path, _ = self.tree_model.predict_with_path(state)
                except:
                    pass
            
            # 生成解释
            explanation = self.explainer.explain_action(state, action, decision_path)
            explanations.append(explanation)
        
        # 将解释添加到历史中
        test_history['explanations'] = explanations
        
        return test_history