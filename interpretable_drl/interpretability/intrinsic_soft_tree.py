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
import pickle
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import NotFittedError

class IntrinsicSoftTree:
    """
    可解释性决策树
    不是“软”决策树，是标准的不可微的决策树
    接口不变。
    """

    def __init__(self, input_dim, num_classes=5, depth=4, **kwargs):
        """
        初始化决策树分类器

        Args:
            input_dim: 输入特征维度
            num_classes: 场景类别数量
            depth: 树的最大深度
            **kwargs: 传递给 DecisionTreeClassifier 的其他参数 (例如 min_samples_leaf)
        """
        self.input_dim = input_dim # 存储以备参考，但 scikit-learn 不需要显式传递
        self.num_classes = num_classes
        self.depth = depth

        self.model = DecisionTreeClassifier(
            max_depth=self.depth,
            random_state=42,
            **kwargs # 允许传入其他 sklearn 参数
        )
        self._is_fitted = False # 标志 跟踪模型是否已训练

    def fit(self, states, labels):
        """
        训练决策树模型

        Args:
            states: 状态数据 (numpy array, shape [n_samples, n_features])
            labels: 场景类别标签 (numpy array, shape [n_samples,])
        """
        if states is None or labels is None or len(states) == 0 or len(labels) == 0:
             print("警告 (IntrinsicSoftTree.fit): 输入数据为空，跳过训练。")
             return
        if len(states) != len(labels):
             print(f"警告 (IntrinsicSoftTree.fit): states ({len(states)}) 和 labels ({len(labels)}) 长度不匹配，跳过训练。")
             return

        try:
            self.model.fit(states, labels)
            self._is_fitted = True
            print(f"决策树训练完成。深度: {self.model.get_depth()}, 叶节点数: {self.model.get_n_leaves()}")
        except Exception as e:
            print(f"错误 (IntrinsicSoftTree.fit): 训练决策树时出错: {e}")
            self._is_fitted = False


    def generate_embedding(self, state):
        """
        生成策略 embedding (叶节点或类别的概率分布)

        Args:
            state: 当前状态 (list or numpy array)

        Returns:
            embedding: 预测的类别概率分布 (numpy array, shape [num_classes,])
                       如果模型未训练或预测出错，返回均匀分布。
        """
        if not self._is_fitted:
            # print("警告 (generate_embedding): 模型尚未训练，返回均匀分布。")
            return np.ones(self.num_classes) / self.num_classes

        try:
            state_np = np.array(state).reshape(1, -1)
            # predict_proba 返回每个类别的概率
            probabilities = self.model.predict_proba(state_np)[0]

            if len(probabilities) < self.num_classes:
                 full_probabilities = np.zeros(self.num_classes)
                 present_classes = self.model.classes_
                 for class_index, prob in zip(present_classes, probabilities):
                     if class_index < self.num_classes:
                         full_probabilities[class_index] = prob
                 return full_probabilities
            else:
                 return probabilities[:self.num_classes]

        except NotFittedError:
            # print("警告 (generate_embedding): 模型尚未训练 (NotFittedError)，返回均匀分布。")
            return np.ones(self.num_classes) / self.num_classes
        except Exception as e:
            print(f"错误 (generate_embedding): 预测时出错: {e}，返回均匀分布。")
            return np.ones(self.num_classes) / self.num_classes

    def predict_path(self, state):
        """
        预测决策路径 (返回节点索引列表) 和最可能的类别

        Args:
            state: 输入状态 (list or numpy array)

        Returns:
            tuple: (path, class_id)
                   - path: 决策路径上的节点索引列表
                   - class_id: 预测的最可能的场景类别 ID
                   如果模型未训练或出错，返回 ([], -1)
        """
        if not self._is_fitted:
            return [], -1

        try:
            state_np = np.array(state).reshape(1, -1)
            # 获取决策路径信息
            decision_path_sparse = self.model.decision_path(state_np)
            # indices 包含路径上的所有节点索引
            path_indices = decision_path_sparse.indices.tolist()

            class_id = self.model.predict(state_np)[0]

            return path_indices, int(class_id)
        except NotFittedError:
            return [], -1
        except Exception as e:
            print(f"错误 (predict_path): 获取决策路径时出错: {e}")
            return [], -1

    def get_decision_rule(self, node_id, feature_names=None):
        """
        获取指定内部节点的决策规则字符串表示

        Args:
            node_id: 内部节点索引
            feature_names: 特征名称列表 (可选)

        Returns:
            rule (str): 决策规则字符串, 或 "叶节点" / "未知节点"
        """
        if not self._is_fitted or not hasattr(self.model, 'tree_'):
            return "未知节点 (模型未训练或无树结构)"

        tree = self.model.tree_

        if node_id < 0 or node_id >= tree.node_count:
             return f"未知节点 (ID {node_id} 超出范围)"

        # 检查是否是叶节点
        if tree.children_left[node_id] == tree.children_right[node_id]:
            return "叶节点"
        else:
            # 分裂特征和阈值
            feature_index = tree.feature[node_id]
            threshold = tree.threshold[node_id]

            # 特征名称
            if feature_names and feature_index < len(feature_names):
                feature_name = feature_names[feature_index]
            else:
                feature_name = f"Feature_{feature_index}"

            # 规则字符串
            rule = f"{feature_name} <= {threshold:.4f}"
            return rule

    def save(self, filepath):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'model': self.model,
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'depth': self.depth,
            '_is_fitted': self._is_fitted
        }
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"决策树模型已保存到 {filepath}")
        except Exception as e:
            print(f"错误 (save): 保存模型失败: {e}")

    def load(self, filepath):
        """加载模型"""
        if not os.path.exists(filepath):
             print(f"警告 (load): 模型文件不存在 {filepath}")
             self._is_fitted = False
             return

        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.input_dim = model_data['input_dim']
            self.num_classes = model_data['num_classes']
            self.depth = model_data['depth']
            self._is_fitted = model_data.get('_is_fitted', isinstance(self.model, DecisionTreeClassifier)) # 检查是否加载成功
            print(f"决策树模型已从 {filepath} 加载。是否已训练: {self._is_fitted}")
        except Exception as e:
            print(f"错误 (load): 加载模型失败: {e}")
            self._is_fitted = False