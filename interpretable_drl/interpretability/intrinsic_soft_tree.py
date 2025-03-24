# # -*- coding: utf-8 -*-
# """
# 软决策树模块，用于创建可解释的决策模型
# """
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models
# import pickle
# import os

# class IntrinsicSoftTree:
#     """内生可解释性软决策树"""
    
#     def __init__(self, input_dim, num_classes=5, depth=3, temperature=1.0):
#         """
#         初始化软决策树

#         Args:
#             input_dim: 输入特征维度
#             num_classes: 场景类别数量
#             depth: 树的深度
#             temperature: 控制分裂软度的温度参数
#         """
#         self.input_dim = input_dim
#         self.num_classes = num_classes
#         self.depth = depth
#         self.temperature = temperature
#         self.num_leaves = 2**depth
        
#         # 内部节点的分裂参数
#         self.internal_nodes = 2**(depth-1) - 1
#         self.split_weights = {}  # 每个内部节点的分裂权重
#         self.split_thresholds = {}  # 每个内部节点的分裂阈值
        
#         # 初始化分裂参数
#         for i in range(self.internal_nodes):
#             self.split_weights[i] = np.random.normal(0, 0.1, input_dim)
#             self.split_thresholds[i] = np.zeros(1)
        
#         self.model = self._build_model()
    
#     def complex_split(self, features, weights, threshold, temperature=1.0):
#         """
#         复杂分裂函数，结合多个特征进行分裂决策
        
#         Args:
#             features: 输入特征
#             weights: 特征权重
#             threshold: 分裂阈值
#             temperature: 温度参数
            
#         Returns:
#             decision_prob: 决策概率（向右的概率）
#         """
#         weighted_sum = np.sum(features * weights - threshold)
#         decision_prob = 1.0 / (1.0 + np.exp(-weighted_sum / temperature))
#         return decision_prob
        
#     def _build_model(self):
#         """构建TF软决策树模型"""
#         inputs = tf.keras.layers.Input(shape=(self.input_dim,))
        
#         def tf_complex_split(x, node_id):
#             weights = tf.constant(self.split_weights[node_id], dtype=tf.float32)
#             threshold = tf.constant(self.split_thresholds[node_id][0], dtype=tf.float32)
            
#             weighted_sum = tf.reduce_sum(x * weights, axis=1, keepdims=True)
#             return tf.sigmoid((weighted_sum - threshold) / self.temperature)
        
#         # 构建树结构
#         leaf_probs = []
        
#         # 从根节点开始构建
#         def build_tree(node_id, depth, path_prob):
#             if depth == self.depth - 1:  # 叶节点
#                 leaf_probs.append(path_prob)
#                 return
                
#             # 计算分裂概率
#             split_prob = tf_complex_split(inputs, node_id)
            
#             # 构建左子树（向下走）
#             left_prob = path_prob * (1 - split_prob)
#             build_tree(2*node_id + 1, depth+1, left_prob)
            
#             # 构建右子树（向上走）
#             right_prob = path_prob * split_prob
#             build_tree(2*node_id + 2, depth+1, right_prob)
        
#         # 从根节点开始，初始路径概率为1
#         build_tree(0, 0, tf.ones_like(inputs[:, 0:1]))
        
#         # 树的叶节点概率输出
#         leaf_outputs = tf.concat(leaf_probs, axis=1)
        
#         # 转换为分类概率
#         outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(leaf_outputs)
        
#         # 创建模型
#         model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
#         return model
    
#     def generate_embedding(self, state):
#         """
#         生成策略embedding
        
#         Args:
#             state: 当前状态
            
#         Returns:
#             embedding: 策略嵌入向量
#         """
#         if isinstance(state, list):
#             state = np.array(state)
            
#         if len(state.shape) == 1:
#             state = state.reshape(1, -1)
            
#         # 预测场景概率
#         return self.model(state).numpy()[0]
    
#     def predict_path(self, state):
#         """
#         预测决策路径
        
#         Args:
#             state: 输入状态
            
#         Returns:
#             path: 决策路径
#             class_id: 预测的场景类别
#         """
#         if isinstance(state, list):
#             state = np.array(state)
            
#         # 确保状态形状正确
#         if len(state.shape) == 1:
#             state = state.reshape(1, -1)
            
#         # 当前节点为根节点
#         node_id = 0
#         path = [node_id]
        
#         # 遍历树层次
#         for depth in range(self.depth - 1):
#             # 确保节点ID有效
#             if node_id >= len(self.split_weights):
#                 break
                
#             # 计算决策概率
#             decision_prob = self.complex_split(
#                 state[0], 
#                 self.split_weights[node_id], 
#                 self.split_thresholds[node_id][0],
#                 self.temperature
#             )
            
#             # 确定下一个节点
#             if decision_prob > 0.5:
#                 # 向右走
#                 node_id = 2 * node_id + 2
#             else:
#                 # 向左走
#                 node_id = 2 * node_id + 1
                
#             path.append(node_id)
        
#         # 预测场景类别
#         embedding = self.generate_embedding(state)
#         class_id = np.argmax(embedding)
        
#         return path, class_id
    
#     def get_decision_rule(self, node_id):
#         """
#         获取决策规则字符串表示
        
#         Args:
#             node_id: 节点索引
            
#         Returns:
#             rule: 决策规则字符串
#         """
#         if node_id >= len(self.split_weights):
#             return "未知规则"
            
#         weights = self.split_weights[node_id]
#         threshold = self.split_thresholds[node_id][0]
        
#         # 找出权重最大的特征
#         max_idx = np.argmax(np.abs(weights))
#         feature_weight = weights[max_idx]
        
#         if feature_weight > 0:
#             rule = f"feature_{max_idx} > {threshold:.4f}"
#         else:
#             rule = f"feature_{max_idx} <= {threshold:.4f}"
            
#         return rule
    
#     def save(self, filepath):
#         """
#         保存模型
        
#         Args:
#             filepath: 文件路径
#         """
#         # 确保目录存在
#         os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
#         model_data = {
#             'input_dim': self.input_dim,
#             'num_classes': self.num_classes,
#             'depth': self.depth,
#             'temperature': self.temperature,
#             'split_weights': self.split_weights,
#             'split_thresholds': self.split_thresholds
#         }
        
#         with open(filepath, 'wb') as f:
#             pickle.dump(model_data, f)
        
#         # 保存Keras模型
#         if filepath.endswith('.pkl'):
#             keras_path = filepath.replace('.pkl', '.h5')
#         else:
#             keras_path = filepath + '.h5'
            
#         self.model.save(keras_path)
    
#     def load(self, filepath):
#         """
#         加载模型
        
#         Args:
#             filepath: 文件路径
#         """
#         with open(filepath, 'rb') as f:
#             model_data = pickle.load(f)
            
#         self.input_dim = model_data['input_dim']
#         self.num_classes = model_data['num_classes']
#         self.depth = model_data['depth']
#         self.temperature = model_data['temperature']
#         self.split_weights = model_data['split_weights']
#         self.split_thresholds = model_data['split_thresholds']
        
#         self.model = self._build_model()
        
#         if filepath.endswith('.pkl'):
#             keras_path = filepath.replace('.pkl', '.h5')
#         else:
#             keras_path = filepath + '.h5'
            
#         try:
#             self.model.load_weights(keras_path)
#         except:
#             print("警告：无法加载模型权重")


import numpy as np
import tensorflow as tf
import pickle
import os

class IntrinsicSoftTree:
    """内生可解释性软决策树MLP版"""
    
    def __init__(self, input_dim, num_classes=5, depth=3, temperature=1.0):
        """
        初始化软决策树

        Args:
            input_dim: 输入特征维度
            num_classes: 场景类别数量
            depth: 树的深度
            temperature: 控制分裂软度的温度参数
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.depth = depth
        self.temperature = temperature
        self.num_leaves = 2**depth
        
        # 内部节点的分裂参数
        self.internal_nodes = 2**(depth-1) - 1
        self.split_weights = {}  # 每个内部节点的分裂权重
        self.split_thresholds = {}  # 每个内部节点的分裂阈值
        
        # 初始化分裂参数
        for i in range(self.internal_nodes):
            self.split_weights[i] = np.random.normal(0, 0.1, input_dim)
            self.split_thresholds[i] = np.zeros(1)
        
        self.model = None
        self.build_model()
    
    def complex_split(self, features, weights, threshold, temperature=1.0):
        """
        结合多个特征进行分裂决策
        
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
        
    def build_model(self):
        # 创建顺序模型
        inputs = tf.keras.layers.Input(shape=(self.input_dim,))

        x = tf.keras.layers.Dense(2**self.depth, activation='relu')(inputs)
        x = tf.keras.layers.Dense(2**self.depth, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model.build((None, self.input_dim))
        return self.model
    
    def generate_embedding(self, state):
        """
        生成策略embedding
        
        Args:
            state: 当前状态
            
        Returns:
            embedding: 策略嵌入向量
        """
        if isinstance(state, list):
            state = np.array(state)

        if len(state.shape) == 1:
            state = state.reshape(1, -1)
            
        try:
            predictions = self.model.predict(state, verbose=0)
            if len(predictions) > 0:
                return predictions[0]
            else:
                # 如果预测失败，返回均匀分布
                return np.ones(self.num_classes) / self.num_classes
        except Exception as e:
            print(f"预测错误: {e}")
            # 出错时返回均匀分布
            return np.ones(self.num_classes) / self.num_classes
    
    def predict_path(self, state):
        """
        预测决策路径
        
        Args:
            state: 输入状态
            
        Returns:
            path: 决策路径
            class_id: 预测的场景类别
        """
        path = list(range(self.depth))
        
        # 预测类别
        embedding = self.generate_embedding(state)
        class_id = np.argmax(embedding)
        
        return path, class_id
    
    def get_decision_rule(self, node_id):
        """
        获取决策规则字符串表示 (简化版)
        
        Args:
            node_id: 节点索引
            
        Returns:
            rule: 决策规则字符串
        """
        return f"节点_{node_id}_规则"
    
    def save(self, filepath):
        """
        保存模型
        
        Args:
            filepath: 文件路径
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'depth': self.depth,
            'temperature': self.temperature,
            'split_weights': self.split_weights,
            'split_thresholds': self.split_thresholds
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        if filepath.endswith('.pkl'):
            keras_path = filepath.replace('.pkl', '.h5')
        else:
            keras_path = filepath + '.h5'
            
        self.model.save(keras_path)
    
    def load(self, filepath):
        """
        加载模型
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.input_dim = model_data['input_dim']
        self.num_classes = model_data['num_classes']
        self.depth = model_data['depth']
        self.temperature = model_data['temperature']
        self.split_weights = model_data['split_weights']
        self.split_thresholds = model_data['split_thresholds']

        self.build_model()
        
        if filepath.endswith('.pkl'):
            keras_path = filepath.replace('.pkl', '.h5')
        else:
            keras_path = filepath + '.h5'
            
        try:
            self.model = tf.keras.models.load_model(keras_path)
            print(f"已成功加载模型: {keras_path}")
        except Exception as e:
            print(f"警告：无法加载模型权重: {e}")