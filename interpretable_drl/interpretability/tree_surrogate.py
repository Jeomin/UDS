# -*- coding: utf-8 -*-
"""
树形替代模型，用于分析DRLagent的决策行为
基于论文: "Improving the interpretability of deep reinforcement learning in urban
drainage system operation"
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import math

class TreeSurrogateModel:
    """
    树形替代模型，用于近似和解释DRLagent的决策
    """
    
    def __init__(self, state_names=None, tree_depth=4):
        """
        初始化树形替代模型
        
        Args:
            state_names: 状态变量名称列表
            tree_depth: 决策树最大深度
        """
        self.state_names = state_names
        self.tree_depth = tree_depth
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def train(self, states, actions):
        """
        训练树形替代模型
        
        Args:
            states: 状态数据
            actions: 动作数据
            
        Returns:
            self: 训练后的模型
        """
        # 处理动作数据
        if isinstance(actions[0], (list, np.ndarray)):
            # 如果动作是数组（如PPO的泵状态），将其转换为字符串表示
            action_strs = []
            for action in actions:
                action_strs.append(str(list(action)))
                
            # 编码动作
            y = self.label_encoder.fit_transform(action_strs)
        else:
            # 如果动作是标量（如DQN的动作索引），直接使用
            y = actions
            
        self.model = DecisionTreeClassifier(
            max_depth=self.tree_depth,
            random_state=42
        )
        
        self.model.fit(states, y)
        
        return self
    
    def predict(self, state):
        """
        预测动作
        
        Args:
            state: 输入状态
            
        Returns:
            action: 预测的动作
        """
        if not self.model:
            raise ValueError("模型尚未训练")
            
        if len(np.array(state).shape) == 1:
            state = np.array([state])
            
        y_pred = self.model.predict(state)
        
        if hasattr(self, 'label_encoder') and self.label_encoder.classes_.size > 0:
            try:
                # 将预测结果转换回原始动作表示
                action_str = self.label_encoder.inverse_transform(y_pred)[0]
                return eval(action_str)
            except:
                # 如果解码失败，返回原始预测
                return y_pred[0]
        else:
            return y_pred[0]
    
    def get_decision_rules(self):
        """
        获取决策规则
        
        Returns:
            rules: 决策规则列表
        """
        if not self.model:
            raise ValueError("模型尚未训练")
            
        # 获取树结构
        tree = self.model.tree_
        
        # 特征名称
        feature_names = self.state_names if self.state_names else [f"Feature_{i}" for i in range(tree.n_features)]
        
        # 类别名称
        if hasattr(self, 'label_encoder') and self.label_encoder.classes_.size > 0:
            try:
                class_names = self.label_encoder.classes_
            except:
                class_names = [f"类别_{i}" for i in range(tree.n_classes[0])]
        else:
            class_names = [f"类别_{i}" for i in range(tree.n_classes[0])]
        
        # 递归提取规则
        def extract_rules(node_id, prefix=""):
            if tree.children_left[node_id] == tree.children_right[node_id]:  # 叶节点
                # 获取该叶节点的主要类别
                class_id = np.argmax(tree.value[node_id])
                if class_id < len(class_names):
                    rule = f"{prefix}那么 {class_names[class_id]}"
                else:
                    rule = f"{prefix}那么 类别_{class_id}"
                return [rule]
                
            # 获取分裂特征和阈值
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            
            # 构建规则
            if feature < len(feature_names):
                feature_name = feature_names[feature]
            else:
                feature_name = f"Feature_{feature}"
                
            # 左分支规则（<= threshold）
            left_prefix = prefix
            if left_prefix:
                left_prefix += " 和 "
            left_prefix += f"{feature_name} <= {threshold:.4f}"
            if not left_prefix.startswith("如果 "):
                left_prefix = "如果 " + left_prefix
                
            # 右分支规则（> threshold）
            right_prefix = prefix
            if right_prefix:
                right_prefix += " 和 "
            right_prefix += f"{feature_name} > {threshold:.4f}"
            if not right_prefix.startswith("如果 "):
                right_prefix = "如果 " + right_prefix
                
            # 递归处理子节点
            left_rules = extract_rules(tree.children_left[node_id], left_prefix)
            right_rules = extract_rules(tree.children_right[node_id], right_prefix)
            
            return left_rules + right_rules
            
        rules = extract_rules(0)
        return rules
    
    def visualize(self, feature_names=None, class_names=None, figsize=(12, 8)):
        """
        可视化决策树
        
        Args:
            feature_names: 特征名称列表
            class_names: 类别名称列表
            figsize: 图形大小
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 使用提供的特征名称或默认名称
        if feature_names is None:
            feature_names = self.state_names if self.state_names else [f"Feature_{i}" for i in range(self.model.n_features_in_)]
            
        # 使用提供的类别名称或编码器的类别
        if class_names is None and hasattr(self, 'label_encoder') and self.label_encoder.classes_.size > 0:
            try:
                class_names = self.label_encoder.classes_
                # 确保class_names是列表类型
                if isinstance(class_names, np.ndarray):
                    class_names = class_names.tolist()
            except:
                class_names = [f"类别_{i}" for i in range(self.model.n_classes_)]
                
        # 绘制决策树
        plot_tree(
            self.model,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            ax=ax
        )
        
        plt.title("决策树替代模型")
        
        return fig
    
    def get_leaf_gini(self):
        """
        获取叶节点的Gini不纯度
        
        Returns:
            leaf_gini: 叶节点Gini不纯度字典
        """
        if not self.model:
            raise ValueError("模型尚未训练")
            
        tree = self.model.tree_
        leaf_gini = {}
        
        # 查找叶节点
        for i in range(tree.node_count):
            if tree.children_left[i] == tree.children_right[i]:  # 叶节点
                # 计算Gini不纯度
                values = tree.value[i][0]
                total = np.sum(values)
                if total > 0:
                    gini = 1.0 - np.sum((values / total) ** 2)
                else:
                    gini = 0.0
                    
                leaf_gini[i] = gini
                
        return leaf_gini
    
    def calculate_interpretability_index(self):
        """
        计算可解释性指数I2（基于Gini不纯度）
        
        Returns:
            I2: 可解释性指数
        """
        leaf_gini = self.get_leaf_gini()
        
        if not leaf_gini:
            return 0.0
            
        # 计算I2 = exp(-平均Gini不纯度)
        avg_gini = np.mean(list(leaf_gini.values()))
        I2 = math.exp(-avg_gini)
        
        return min(1.0, I2)  # 确保I2不超过1
    
    def get_feature_importance(self):
        """
        获取特征重要性
        
        Returns:
            importance: 特征重要性字典
        """
        if not self.model:
            raise ValueError("模型尚未训练")
            
        importance = {}
        
        # 特征名称
        feature_names = self.state_names if self.state_names else [f"Feature_{i}" for i in range(self.model.n_features_in_)]
        
        # 特征重要性
        for i, imp in enumerate(self.model.feature_importances_):
            if i < len(feature_names):
                importance[feature_names[i]] = imp
            else:
                importance[f"Feature_{i}"] = imp
                
        return importance
    
    def save(self, filepath):
        """
        保存模型
        
        Args:
            filepath: 文件路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'state_names': self.state_names,
            'tree_depth': self.tree_depth,
            'label_encoder': self.label_encoder
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath):
        """
        加载模型
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.state_names = model_data['state_names']
        self.tree_depth = model_data['tree_depth']
        self.label_encoder = model_data['label_encoder']