# -*- coding: utf-8 -*-
"""
软决策树实现，用于创建可解释的决策模型
基于论文: "Improving the interpretability of deep reinforcement learning in urban
drainage system operation"
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pickle
import os

class SoftDecisionTree:
    """
    可微分的软决策树，通过连续的分裂决策实现完全可微分
    """
    
    def __init__(self, input_dim, output_dim, depth=4, temperature=1.0):
        """
        初始化软决策树
        
        Args:
            input_dim (int): 输入特征维度
            output_dim (int): 输出动作维度
            depth (int): 树的深度
            temperature (float): 控制分裂软度的温度参数
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.temperature = temperature
        self.num_leaves = 2**depth
        
        # 内部节点的分裂参数
        self.internal_nodes = 2**(depth-1) - 1
        self.split_weights = {}  # 每个内部节点的分裂权重
        self.split_thresholds = {}  # 每个内部节点的分裂阈值
        
        # 叶节点的输出权重
        self.leaf_weights = np.random.normal(0, 0.1, (self.num_leaves, self.output_dim))
        
        # 初始化分裂参数
        for i in range(self.internal_nodes):
            self.split_weights[i] = np.random.normal(0, 0.1, input_dim)
            self.split_thresholds[i] = np.zeros(1)
        
        # TensorFlow模型
        self.model = self._build_model()
        
    def _build_model(self):
        """
        构建TensorFlow软决策树模型
        
        Returns:
            model: Keras模型
        """
        inputs = layers.Input(shape=(self.input_dim,))
        
        # 构建决策树结构
        leaf_outputs = []
        path_probs = [tf.ones_like(inputs[:, 0:1])]  # 根节点的路径概率为1
        node_queue = [(0, 0, path_probs[0])]  # (节点ID, 深度, 路径概率)
        
        while node_queue:
            node_id, node_depth, path_prob = node_queue.pop(0)
            
            # 如果是叶节点
            if node_depth == self.depth - 1:
                leaf_id = node_id - (2**(node_depth) - 1)
                leaf_weight = tf.constant(self.leaf_weights[leaf_id], dtype=tf.float32)
                leaf_output = tf.reshape(path_prob, [-1, 1]) * leaf_weight
                leaf_outputs.append(leaf_output)
            else:
                # 计算决策概率
                split_weight = tf.constant(self.split_weights[node_id], dtype=tf.float32)
                split_threshold = tf.constant(self.split_thresholds[node_id], dtype=tf.float32)
                
                weighted_sum = tf.reduce_sum(inputs * split_weight, axis=1, keepdims=True)
                decision_prob = tf.sigmoid((weighted_sum - split_threshold) / self.temperature)
                
                # 左子节点（概率为1-决策概率）
                left_path_prob = path_prob * (1 - decision_prob)
                left_node_id = 2 * node_id + 1
                left_node_depth = node_depth + 1
                node_queue.append((left_node_id, left_node_depth, left_path_prob))
                
                # 右子节点（概率为决策概率）
                right_path_prob = path_prob * decision_prob
                right_node_id = 2 * node_id + 2
                right_node_depth = node_depth + 1
                node_queue.append((right_node_id, right_node_depth, right_path_prob))
        
        # 组合所有叶节点的输出
        if leaf_outputs:
            final_output = tf.add_n(leaf_outputs)
        else:
            final_output = layers.Dense(self.output_dim)(inputs)  # 后备输出
        
        return models.Model(inputs=inputs, outputs=final_output)
    
    def complex_split(self, features, weights, threshold, temperature=1.0):
        """
        复杂分裂函数，结合多个特征进行分裂决策
        
        Args:
            features: 输入特征
            weights: 特征权重
            threshold: 分裂阈值
            temperature: 温度参数
            
        Returns:
            decision_prob: 决策概率（向右的概率）
        """
        weighted_sum = np.sum(features * weights - threshold)
        decision_prob = 1.0 / (1.0 + np.exp(-weighted_sum / temperature))
        return decision_prob
    
    def predict(self, state):
        """
        预测动作
        
        Args:
            state: 输入状态
            
        Returns:
            action: 预测动作
        """
        if isinstance(state, list):
            state = np.array(state)
        
        # 确保状态形状正确
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        # 使用模型预测
        return self.model.predict(state)[0]
    
    def predict_with_path(self, state):
        """
        预测动作并返回决策路径
        
        Args:
            state: 输入状态
            
        Returns:
            path: 决策路径
            action: 预测动作
        """
        if isinstance(state, list):
            state = np.array(state)
        
        # 确保状态形状正确
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        # 当前节点为根节点
        node_id = 0
        path = [node_id]
        
        # 遍历树层次
        for depth in range(self.depth - 1):
            # 确保节点ID有效
            if node_id >= len(self.split_weights):
                break
                
            # 计算决策概率
            decision_prob = self.complex_split(
                state[0], 
                self.split_weights[node_id], 
                self.split_thresholds[node_id][0],
                self.temperature
            )
            
            # 确定下一个节点
            if decision_prob > 0.5:
                # 向右走
                node_id = 2 * node_id + 2
            else:
                # 向左走
                node_id = 2 * node_id + 1
                
            path.append(node_id)
        
        # 计算叶节点编号
        leaf_id = node_id - (2**(len(path)-1) - 1)
        if leaf_id < 0 or leaf_id >= len(self.leaf_weights):
            leaf_id = 0  # 默认值
        
        # 获取叶节点输出
        action = self.leaf_weights[leaf_id]
        
        return path, action
    
    def get_decision_rule(self, node_id):
        """
        获取决策规则字符串表示
        
        Args:
            node_id: 节点索引
            
        Returns:
            rule: 决策规则字符串
        """
        if node_id >= len(self.split_weights):
            return "未知规则"
            
        weights = self.split_weights[node_id]
        threshold = self.split_thresholds[node_id][0]
        
        # 找出权重最大的特征
        max_idx = np.argmax(np.abs(weights))
        feature_weight = weights[max_idx]
        
        if feature_weight > 0:
            rule = f"feature_{max_idx} > {threshold:.4f}"
        else:
            rule = f"feature_{max_idx} <= {threshold:.4f}"
            
        return rule
    
    def get_path_rules(self, path):
        """
        获取路径上的所有决策规则
        
        Args:
            path: 决策路径
            
        Returns:
            rules: 决策规则列表
        """
        rules = []
        for node_id in path[:-1]:  # 叶节点没有规则
            rule = self.get_decision_rule(node_id)
            rules.append(rule)
        return rules
    
    def train(self, states, actions, epochs=100, batch_size=32, learning_rate=0.001):
        """
        训练软决策树
        
        Args:
            states: 状态数据
            actions: 动作数据
            epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            
        Returns:
            history: 训练历史
        """
        # 编译模型
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse'
        )
        
        # 确保数据格式正确
        states = np.array(states)
        actions = np.array(actions)
        
        # 训练模型
        history = self.model.fit(
            states, actions,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # 提取训练后的权重
        # 这里需要根据实际模型结构进行调整
        
        return history.history
    
    def save(self, filepath):
        """
        保存模型
        
        Args:
            filepath: 文件路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'depth': self.depth,
            'temperature': self.temperature,
            'split_weights': self.split_weights,
            'split_thresholds': self.split_thresholds,
            'leaf_weights': self.leaf_weights
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        # 保存Keras模型
        self.model.save(filepath.replace('.pkl', '.h5'))
    
    def load(self, filepath):
        """
        加载模型
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.input_dim = model_data['input_dim']
        self.output_dim = model_data['output_dim']
        self.depth = model_data['depth']
        self.temperature = model_data['temperature']
        self.split_weights = model_data['split_weights']
        self.split_thresholds = model_data['split_thresholds']
        self.leaf_weights = model_data['leaf_weights']
        
        # 重建模型
        self.model = self._build_model()
        
        # 尝试加载Keras模型权重
        try:
            self.model.load_weights(filepath.replace('.pkl', '.h5'))
        except:
            print("警告：无法加载模型权重")