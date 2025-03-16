# -*- coding: utf-8 -*-
"""
树形代理模型，用于解释DRL代理的决策逻辑
基于论文: "Improving the interpretability of deep reinforcement learning in urban
drainage system operation"
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import export_graphviz, plot_tree, export_text
import pickle
import os

class TreeSurrogateModel:
    """
    使用决策树作为DRL代理的代理模型，提供可解释性
    """
    
    def __init__(self, state_names=None, action_names=None, max_depth=5, min_samples_leaf=10):
        """
        初始化树形代理模型
        
        Args:
            state_names: 状态变量名称列表
            action_names: 动作名称列表
            max_depth: 决策树最大深度
            min_samples_leaf: 每个叶节点的最小样本数
        """
        self.state_names = state_names
        self.action_names = action_names
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.model = None
        self.is_classifier = False
    
    def train(self, states, actions):
        """
        训练树形代理模型
        
        Args:
            states: 状态数据
            actions: 动作数据
            
        Returns:
            self: 模型实例
        """
        # 确保数据格式正确
        states = np.array(states)
        actions = np.array(actions)
        
        # 确定是分类问题还是回归问题
        if len(actions.shape) == 1:
            # 检查是否为分类变量
            unique_values = np.unique(actions)
            if len(unique_values) < 20 or all(isinstance(val, (int, np.integer)) for val in unique_values):
                # 如果唯一值少于20或都是整数，认为是分类问题
                self.model = DecisionTreeClassifier(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf
                )
                self.is_classifier = True
            else:
                # 否则是回归问题
                self.model = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf
                )
                self.is_classifier = False
        else:
            # 多维输出，使用回归树
            self.model = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf
            )
            self.is_classifier = False
        
        # 训练模型
        self.model.fit(states, actions)
        
        return self
    
    def predict(self, state):
        """
        预测动作
        
        Args:
            state: 状态
            
        Returns:
            action: 预测的动作
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        # 确保状态格式正确
        if isinstance(state, list):
            state = np.array(state)
        
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
            
        return self.model.predict(state)[0]
    
    def predict_with_path(self, state):
        """
        预测动作并返回决策路径
        
        Args:
            state: 状态
            
        Returns:
            decision_path: 决策路径（节点编号列表）
            action: 预测的动作
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        # 确保状态格式正确
        if isinstance(state, list):
            state = np.array(state)
        
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        # 获取叶节点编号
        leaf_id = self.model.apply(state)[0]
        
        # 获取决策路径
        decision_path = self.model.decision_path(state)
        path_indices = decision_path.indices[decision_path.indptr[0]:decision_path.indptr[1]]
        
        # 预测动作
        action = self.model.predict(state)[0]
        
        return path_indices.tolist(), action
    
    def get_feature_importance(self):
        """
        获取特征重要性
        
        Returns:
            importance: 特征重要性字典
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        feature_importance = self.model.feature_importances_
        
        # 创建特征名称
        if self.state_names:
            feature_names = self.state_names
        else:
            feature_names = [f"特征_{i}" for i in range(len(feature_importance))]
            
        # 创建重要性字典
        importance = dict(zip(feature_names, feature_importance))
        
        # 按重要性排序
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return importance
    
    def visualize(self, output_file=None, feature_names=None, class_names=None, figsize=(15, 10)):
        """
        可视化决策树
        
        Args:
            output_file: 输出文件路径
            feature_names: 特征名称
            class_names: 类别名称
            figsize: 图表大小
            
        Returns:
            fig: 图表对象
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        # 设置特征名称和类别名称
        if feature_names is None:
            feature_names = self.state_names if self.state_names else None
            
        if class_names is None and self.is_classifier:
            class_names = self.action_names if self.action_names else None
            
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制决策树
        plot_tree(
            self.model,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            ax=ax
        )
        
        # 保存图表
        if output_file:
            # 确保目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            
        return fig
    
    def get_text_representation(self, feature_names=None):
        """
        获取决策树的文本表示
        
        Args:
            feature_names: 特征名称
            
        Returns:
            text: 决策树的文本表示
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        # 设置特征名称
        if feature_names is None:
            feature_names = self.state_names if self.state_names else None
            
        # 获取文本表示
        return export_text(self.model, feature_names=feature_names)
    
    def get_decision_rules(self):
        """
        获取决策规则
        
        Returns:
            rules: 决策规则列表
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        # 获取树结构
        tree = self.model.tree_
        
        # 特征名称
        feature_names = self.state_names if self.state_names else [f"特征_{i}" for i in range(tree.n_features)]
        
        # 类别名称
        if self.is_classifier:
            if self.action_names:
                class_names = self.action_names[:tree.n_classes[0]]
            else:
                class_names = [f"类别_{i}" for i in range(tree.n_classes[0])]
        
        # 递归提取决策规则
        def extract_rules(node_id, depth, path):
            # 如果是叶节点
            if tree.children_left[node_id] == -1:
                if self.is_classifier:
                    # 分类器：返回类别概率
                    class_prob = tree.value[node_id][0] / np.sum(tree.value[node_id][0])
                    max_class = np.argmax(class_prob)
                    rule = f"如果 {' 和 '.join(path)} 那么 {class_names[max_class]} (概率: {class_prob[max_class]:.2f})"
                else:
                    # 回归器：返回预测值
                    rule = f"如果 {' 和 '.join(path)} 那么 值 = {tree.value[node_id][0][0]:.4f}"
                return [rule]
            
            # 获取特征和阈值
            feature = feature_names[tree.feature[node_id]]
            threshold = tree.threshold[node_id]
            
            # 递归左右子树
            left_path = path + [f"{feature} <= {threshold:.4f}"]
            left_rules = extract_rules(tree.children_left[node_id], depth + 1, left_path)
            
            right_path = path + [f"{feature} > {threshold:.4f}"]
            right_rules = extract_rules(tree.children_right[node_id], depth + 1, right_path)
            
            return left_rules + right_rules
        
        # 从根节点开始提取规则
        rules = extract_rules(0, 0, [])
        
        return rules
    
    def calculate_gini_impurity(self):
        """
        计算决策树的Gini不纯度
        
        Returns:
            gini_impurity: Gini不纯度
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        # 获取树结构
        tree = self.model.tree_
        
        # 计算每个叶节点的Gini不纯度
        gini_values = []
        n_samples = []
        
        for i in range(tree.node_count):
            # 如果是叶节点
            if tree.children_left[i] == -1:
                if self.is_classifier:
                    # 分类器：计算Gini不纯度
                    probs = tree.value[i][0] / np.sum(tree.value[i][0])
                    gini = 1 - np.sum(np.square(probs))
                else:
                    # 回归器：使用方差作为不纯度（简化计算）
                    gini = np.var(tree.value[i]) if len(tree.value[i].flatten()) > 1 else 0
                
                gini_values.append(gini)
                n_samples.append(tree.n_node_samples[i])
        
        # 计算加权平均Gini不纯度
        if sum(n_samples) > 0:
            weighted_gini = sum(g * n for g, n in zip(gini_values, n_samples)) / sum(n_samples)
        else:
            weighted_gini = 0
            
        return weighted_gini
    
    def calculate_interpretability_index(self):
        """
        计算可解释性指数 I2
        
        Returns:
            I2: 可解释性指数
        """
        # 计算Gini不纯度
        gini = self.calculate_gini_impurity()
        
        # 计算I2指数（越接近1越好）
        I2 = np.exp(-gini)
        
        return I2
    
    def save(self, filepath):
        """
        保存模型
        
        Args:
            filepath: 文件路径
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存模型和元数据
        data = {
            'model': self.model,
            'state_names': self.state_names,
            'action_names': self.action_names,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'is_classifier': self.is_classifier
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath):
        """
        加载模型
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.state_names = data['state_names']
        self.action_names = data['action_names']
        self.max_depth = data['max_depth']
        self.min_samples_leaf = data['min_samples_leaf']
        self.is_classifier = data['is_classifier']
        
        return self