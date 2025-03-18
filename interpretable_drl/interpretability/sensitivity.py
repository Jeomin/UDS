# -*- coding: utf-8 -*-
"""
敏感性分析模块，用于分析输入状态对DRLagent决策的影响
基于论文: "Improving the interpretability of deep reinforcement learning in urban
drainage system operation"
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    SALib_available = True
except ImportError:
    print("警告: SALib库未安装，敏感性分析功能受限。请使用 pip install SALib 安装。")
    SALib_available = False

class SensitivityAnalysis:
    """
    使用Sobol'方法进行全局敏感性分析
    """
    
    def __init__(self, agent, state_names=None):
        """
        初始化敏感性分析器
        
        Args:
            agent: DRLagent对象
            state_names: 状态变量名称列表
        """
        self.agent = agent
        self.state_names = state_names
        
        # 检查agent是否有choose_action方法
        if not hasattr(self.agent, 'choose_action'):
            raise ValueError("提供的agent对象必须有choose_action方法")
        
        # 检查SALib是否可用
        if not SALib_available:
            print("警告: SALib库未安装，部分敏感性分析功能将无法使用。")
    
    def sample_states(self, state_bounds, n_samples=1024):
        """
        使用Saltelli抽样方法生成输入样本
        
        Args:
            state_bounds: 状态变量界限，格式为[(min1, max1), (min2, max2), ...]
            n_samples: 基础样本数量
            
        Returns:
            samples: 样本数组
            problem: 问题定义
        """
        if not SALib_available:
            raise ImportError("此功能需要SALib库。请使用 pip install SALib 安装。")
            
        # 准备问题定义
        names = []
        for i in range(len(state_bounds)):
            if self.state_names and i < len(self.state_names):
                names.append(self.state_names[i])
            else:
                names.append(f"x{i}")
        
        problem = {
            'num_vars': len(state_bounds),
            'names': names,
            'bounds': state_bounds
        }
        
        # 生成样本
        samples = saltelli.sample(problem, n_samples)
        
        return samples, problem
    
    def analyze(self, state_bounds, n_samples=1024, output_index=None):
        """
        执行Sobol敏感性分析
        
        Args:
            state_bounds: 状态变量界限，格式为[(min1, max1), (min2, max2), ...]
            n_samples: 基础样本数量
            output_index: 要分析的特定输出索引（如果agent返回向量）
            
        Returns:
            results: 敏感性分析结果
        """
        if not SALib_available:
            raise ImportError("此功能需要SALib库。请使用 pip install SALib 安装。")
            
        # 生成输入样本
        samples, problem = self.sample_states(state_bounds, n_samples)
        
        # 获取agent在样本上的输出
        Y = np.zeros(len(samples))
        for i, state in enumerate(samples):
            try:
                action = self.agent.choose_action(state, False)
                
                # 处理不同类型的动作
                if isinstance(action, tuple):
                    # (logits, action)
                    action_value = action[0]
                else:
                    action_value = action
                    
                if isinstance(action_value, (list, np.ndarray)):
                    if output_index is not None:
                        # 确保获取标量值
                        if isinstance(action_value[output_index], (list, np.ndarray)):
                            Y[i] = float(action_value[output_index][0])
                        else:
                            Y[i] = float(action_value[output_index])
                    else:
                        if isinstance(action_value[0], (list, np.ndarray)):
                            Y[i] = float(action_value[0][0])
                        else:
                            Y[i] = float(action_value[0])
                else:
                    Y[i] = float(action_value)
            except Exception as e:
                print(f"处理样本 {i} 时出错: {e}")
                Y[i] = 0
        
        # 执行Sobol分析
        Si = sobol.analyze(problem, Y)
        
        # 提取结果
        results = {
            'S1': Si['S1'],  # 一阶敏感性指数
            'S1_conf': Si['S1_conf'],  # 一阶敏感性指数的置信区间
            'ST': Si['ST'],  # 总敏感性指数
            'ST_conf': Si['ST_conf'],  # 总敏感性指数的置信区间
            'S2': Si.get('S2', None),  # 二阶敏感性指数
            'S2_conf': Si.get('S2_conf', None)  # 二阶敏感性指数的置信区间
        }
        
        return results
    
    def calculate_interpretability_index(self, sensitivity_results):
        """
        计算可解释性指数 I1
        
        Args:
            sensitivity_results: 敏感性分析结果
            
        Returns:
            I1: 可解释性指数
        """
        # 获取总敏感性指数
        ST = sensitivity_results['ST']
        
        # 计算I1（总敏感性指数的平方和）
        I1 = np.sum(np.square(ST))
        
        return I1
    
    def plot_sensitivity_indices(self, sensitivity_results, figsize=(12, 6)):
        """
        绘制敏感性指数
        
        Args:
            sensitivity_results: 敏感性分析结果
            figsize: 图表大小
            
        Returns:
            fig: 图表对象
        """
        # 获取数据
        S1 = sensitivity_results['S1']
        ST = sensitivity_results['ST']
        
        # 创建索引和标签
        indices = range(len(S1))
        
        if self.state_names and len(self.state_names) == len(S1):
            labels = self.state_names
        else:
            labels = [f"状态_{i}" for i in indices]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制条形图
        bar_width = 0.35
        x = np.arange(len(indices))
        ax.bar(x - bar_width/2, S1, bar_width, label='一阶敏感性指数')
        ax.bar(x + bar_width/2, ST, bar_width, label='总敏感性指数')
        
        # 添加标签和标题
        ax.set_xlabel('状态变量')
        ax.set_ylabel('敏感性指数')
        ax.set_title('状态变量敏感性分析')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        return fig
    
    def get_important_features(self, sensitivity_results, threshold=0.05):
        """
        获取重要特征列表
        
        Args:
            sensitivity_results: 敏感性分析结果
            threshold: 重要性阈值
            
        Returns:
            important_features: 重要特征列表
        """
        # 获取总敏感性指数
        ST = sensitivity_results['ST']
        
        # 创建特征名称
        if self.state_names and len(self.state_names) == len(ST):
            names = self.state_names
        else:
            names = [f"状态_{i}" for i in range(len(ST))]
        
        # 创建特征和敏感性指数的列表
        features = list(zip(names, ST))
        
        # 按敏感性降序排序
        features.sort(key=lambda x: x[1], reverse=True)
        
        # 过滤掉敏感性低于阈值的特征
        important_features = [f for f in features if f[1] >= threshold]
        
        return important_features
    
    def compare_agents(self, other_agent, state_bounds, n_samples=1024, output_index=None):
        """
        比较两个agent的敏感性
        
        Args:
            other_agent: 另一个DRLagent对象
            state_bounds: 状态变量界限
            n_samples: 样本数量
            output_index: 输出索引
            
        Returns:
            comparison: 比较结果
        """
        # 创建另一个敏感性分析器
        other_analyzer = SensitivityAnalysis(other_agent, self.state_names)
        
        # 分析两个agent
        results1 = self.analyze(state_bounds, n_samples, output_index)
        results2 = other_analyzer.analyze(state_bounds, n_samples, output_index)
        
        # 计算解释性指数
        I1_1 = self.calculate_interpretability_index(results1)
        I1_2 = other_analyzer.calculate_interpretability_index(results2)
        
        # 比较结果
        comparison = {
            'agent1_results': results1,
            'agent2_results': results2,
            'agent1_I1': I1_1,
            'agent2_I1': I1_2,
            'diff_ST': results1['ST'] - results2['ST'],
            'diff_I1': I1_1 - I1_2
        }
        
        return comparison
    
    def plot_comparison(self, comparison, figsize=(14, 8)):
        """
        绘制比较图
        
        Args:
            comparison: 比较结果
            figsize: 图表大小
            
        Returns:
            fig: 图表对象
        """
        # 获取数据
        ST1 = comparison['agent1_results']['ST']
        ST2 = comparison['agent2_results']['ST']
        
        # 创建索引和标签
        indices = range(len(ST1))
        
        if self.state_names and len(self.state_names) == len(ST1):
            labels = self.state_names
        else:
            labels = [f"状态_{i}" for i in indices]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        bar_width = 0.35
        x = np.arange(len(indices))
        ax.bar(x - bar_width/2, ST1, bar_width, label='agent1总敏感性')
        ax.bar(x + bar_width/2, ST2, bar_width, label='agent2总敏感性')
        
        ax.set_xlabel('状态变量')
        ax.set_ylabel('敏感性指数')
        ax.set_title('agent敏感性比较')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        
        # 添加I1值
        textstr = f"agent1 I1: {comparison['agent1_I1']:.4f}\nagent2 I1: {comparison['agent2_I1']:.4f}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        return fig