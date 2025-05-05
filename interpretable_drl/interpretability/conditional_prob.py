# -*- coding: utf-8 -*-
"""
条件概率分析模块，用于分析DRL控制动作的后果
基于论文: "Improving the interpretability of deep reinforcement learning in urban
drainage system operation"
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import KernelDensity

class ConditionalProbabilityAnalysis:
    """
    分析DRL控制动作后果的条件概率
    """
    
    def __init__(self):
        """
        初始化条件概率分析器
        """
        pass
    
    def analyze_state_action_result(self, dataset, condition=None):
        """
        分析状态-动作-结果数据集的条件概率
        
        Args:
            dataset: 包含(状态, 动作, 结果)元组的数据集
            condition: 过滤条件函数，接受(状态, 动作)返回布尔值
            
        Returns:
            stats_dict: 统计分析结果
        """
        # 如果提供了条件，过滤数据集
        if condition is not None:
            filtered_data = [item for item in dataset if condition(item[0], item[1])]
        else:
            filtered_data = dataset
            
        # 提取结果
        results = [item[2] for item in filtered_data]
        
        # 如果没有数据满足条件
        if not results:
            return {
                'count': 0,
                'mean': None,
                'std': None,
                'min': None,
                'max': None,
                'median': None,
                'distribution': None
            }
            
        # 计算统计量
        results_array = np.array(results)
        
        stats_dict = {
            'count': len(results),
            'mean': np.mean(results_array),
            'std': np.std(results_array),
            'min': np.min(results_array),
            'max': np.max(results_array),
            'median': np.median(results_array),
            'distribution': results_array
        }
        
        return stats_dict
    
    def compare_conditions(self, dataset, baseline_condition=None, target_condition=None):
        """
        比较不同条件下的结果分布
        
        Args:
            dataset: 包含(状态, 动作, 结果)元组的数据集
            baseline_condition: 基准条件函数
            target_condition: 目标条件函数
            
        Returns:
            comparison: 比较结果
        """
        # 获取基准分布
        baseline_stats = self.analyze_state_action_result(dataset, baseline_condition)
        
        # 获取目标分布
        target_stats = self.analyze_state_action_result(dataset, target_condition)
        
        # 计算归一化差异
        if baseline_stats['count'] > 0 and target_stats['count'] > 0:
            # T值：(基准均值 - 目标均值) / 基准标准差
            T = ((baseline_stats['mean'] - target_stats['mean']) / baseline_stats['std'] 
                if baseline_stats['std'] > 0 else 0)
            
            # 执行t检验
            try:
                t_stat, p_value = stats.ttest_ind(
                    baseline_stats['distribution'],
                    target_stats['distribution'],
                    equal_var=False  # 不假设等方差（Welch's t-test）
                )
            except:
                t_stat, p_value = None, None
        else:
            T = None
            t_stat = None
            p_value = None
            
        # 比较结果
        comparison = {
            'baseline': baseline_stats,
            'target': target_stats,
            'T': T,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05 if p_value is not None else None
        }
        
        return comparison
    
    def create_leaf_conditions(self, tree_model):
        """
        基于树模型创建叶节点条件
        
        Args:
            tree_model: 树形agent模型
            
        Returns:
            leaf_conditions: 叶节点条件函数字典
        """
        if not hasattr(tree_model, 'model') or tree_model.model is None:
            raise ValueError("树模型必须先训练")
            
        # 获取决策规则
        try:
            rules = tree_model.get_decision_rules()
        except:
            raise ValueError("无法获取决策规则")
        
        # 创建条件函数
        leaf_conditions = {}
        
        for i, rule in enumerate(rules):
            # 提取IF部分（规则条件）
            if_part = rule.split("那么")[0]
            if "如果" in if_part:
                if_part = if_part.replace("如果", "").strip()
            conditions = if_part.split(" 和 ")
            
            # 创建条件函数
            def create_condition(conditions=conditions):
                def condition_func(state, action):
                    # 检查状态是否满足所有条件
                    for cond in conditions:
                        # 解析条件
                        parts = cond.split()
                        if len(parts) < 3:
                            continue
                            
                        feature = parts[0]
                        operator = parts[1]
                        value = float(parts[2])
                        
                        # 获取特征索引
                        if feature.startswith("Feature_"):
                            try:
                                feature_idx = int(feature.split("_")[1])
                            except:
                                continue
                        elif feature.startswith("feature_"):
                            try:
                                feature_idx = int(feature.split("_")[1])
                            except:
                                continue
                        else:
                            # 如果有特征名称映射，需要在这里查找索引
                            # 简化处理，假设特征已经是索引
                            try:
                                feature_idx = int(feature)
                            except:
                                # 未知特征，跳过
                                continue
                            
                        # 状态长度检查
                        if feature_idx >= len(state):
                            continue
                            
                        # 检查条件
                        if operator == "<=":
                            if state[feature_idx] > value:
                                return False
                        elif operator == ">":
                            if state[feature_idx] <= value:
                                return False
                        elif operator == "<":
                            if state[feature_idx] >= value:
                                return False
                        elif operator == ">=":
                            if state[feature_idx] < value:
                                return False
                                
                    return True
                
                return condition_func
                
            leaf_conditions[i] = create_condition()
            
        return leaf_conditions
    
    def analyze_tree_leaves(self, tree_model, dataset):
        """
        分析树模型叶节点的条件概率
        
        Args:
            tree_model: 树形agent模型
            dataset: 包含(状态, 动作, 结果)元组的数据集
            
        Returns:
            leaf_analysis: 叶节点分析结果
        """
        # 创建叶节点条件
        leaf_conditions = self.create_leaf_conditions(tree_model)
        
        # 分析每个叶节点
        leaf_analysis = {}
        
        for leaf_id, condition in leaf_conditions.items():
            # 获取该叶节点的统计信息
            stats = self.analyze_state_action_result(dataset, condition)
            leaf_analysis[leaf_id] = stats
            
        return leaf_analysis
    
    def calculate_leaf_effectiveness(self, tree_model, dataset):
        """
        计算叶节点控制有效性
        
        Args:
            tree_model: 树形agent模型
            dataset: 包含(状态, 动作, 结果)元组的数据集
            
        Returns:
            effectiveness: 叶节点有效性字典
        """
        # 分析叶节点
        leaf_analysis = self.analyze_tree_leaves(tree_model, dataset)
        
        # 计算整体基准统计信息
        overall_stats = self.analyze_state_action_result(dataset)
        
        # 计算每个叶节点的有效性
        effectiveness = {}
        
        for leaf_id, stats in leaf_analysis.items():
            # 如果叶节点有足够的样本
            if stats['count'] > 10 and overall_stats['count'] > 0:
                # 计算T值：(整体均值 - 叶节点均值) / 整体标准差
                T = ((overall_stats['mean'] - stats['mean']) / overall_stats['std'] 
                    if overall_stats['std'] > 0 else 0)
                
                # 执行t检验
                try:
                    t_stat, p_value = stats.ttest_ind(
                        overall_stats['distribution'],
                        stats['distribution'],
                        equal_var=False  # 不假设等方差
                    )
                except:
                    t_stat, p_value = None, None
                
                effectiveness[leaf_id] = {
                    'T': T,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05 if p_value is not None else None,
                    'sample_count': stats['count'],
                    'mean': stats['mean'],
                    'std': stats['std']
                }
            else:
                effectiveness[leaf_id] = {
                    'T': None,
                    't_statistic': None,
                    'p_value': None,
                    'significant': None,
                    'sample_count': stats['count'],
                    'mean': stats['mean'] if stats['count'] > 0 else None,
                    'std': stats['std'] if stats['count'] > 0 else None
                }
                
        return effectiveness
    
    def calculate_interpretability_index(self, effectiveness):
        """
        计算可解释性指数 I3
        
        Args:
            effectiveness: 叶节点有效性字典
            
        Returns:
            I3: 可解释性指数
        """
        # 收集有效T值
        T_values = [stats['T'] for leaf_id, stats in effectiveness.items() 
                    if stats['T'] is not None and stats['sample_count'] > 10]
        
        # 如果没有有效值，返回0
        if not T_values:
            return 0
            
        # 计算I3：所有T值的平均值
        I3 = np.mean(T_values)
        
        return I3
    
    def plot_distributions(self, dataset, conditions=None, labels=None, kde=True, bins=30, figsize=(12, 8)):
        """
        绘制不同条件下的结果分布
        
        Args:
            dataset: 包含(状态, 动作, 结果)元组的数据集
            conditions: 条件函数列表
            labels: 条件标签列表
            kde: 是否使用核密度估计
            bins: 直方图的箱数
            figsize: 图表大小
            
        Returns:
            fig: 图表对象
        """
        # 如果没有提供条件，使用无条件
        if conditions is None:
            conditions = [None]
            
        # 如果没有提供标签，使用默认标签
        if labels is None:
            labels = [f"条件_{i}" for i in range(len(conditions))]
            
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 为每个条件绘制分布
        for condition, label in zip(conditions, labels):
            # 获取统计信息
            stats = self.analyze_state_action_result(dataset, condition)
            
            # 如果没有数据，跳过
            if stats['count'] == 0:
                continue
                
            # 提取结果
            results = stats['distribution']
            
            # 绘制分布
            if kde:
                # 使用核密度估计
                try:
                    x = np.linspace(min(results), max(results), 1000)
                    kde = stats.gaussian_kde(results)
                    ax.plot(x, kde(x), label=f"{label} (n={stats['count']})")
                except:
                    # 如果KDE失败，改用直方图
                    ax.hist(results, bins=bins, alpha=0.5, density=True, label=f"{label} (n={stats['count']})")
            else:
                # 使用直方图
                ax.hist(results, bins=bins, alpha=0.5, density=True, label=f"{label} (n={stats['count']})")
                
            # 添加均值线
            ax.axvline(stats['mean'], color='k', linestyle='--', alpha=0.3)
            ax.text(stats['mean'], 0.01, f"均值: {stats['mean']:.2f}", ha='center')
            
        # 添加标签和图例
        ax.set_xlabel('结果')
        ax.set_ylabel('概率密度')
        ax.set_title('不同条件下的结果分布')
        ax.legend()
        
        return fig
    
    def visualize_leaf_effectiveness(self, tree_model, effectiveness, figsize=(14, 8)):
        """
        可视化叶节点有效性
        
        Args:
            tree_model: 树形agent模型
            effectiveness: 叶节点有效性字典
            figsize: 图表大小
            
        Returns:
            fig: 图表对象
        """
        # 提取数据
        leaf_ids = list(effectiveness.keys())
        T_values = [stats['T'] if stats['T'] is not None else 0 for stats in effectiveness.values()]
        sample_counts = [stats['sample_count'] for stats in effectiveness.values()]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 设置条形图颜色（基于T值）
        colors = ['g' if T > 0 else 'r' if T < 0 else 'gray' for T in T_values]
        
        # 绘制条形图
        bars = ax.bar(leaf_ids, T_values, color=colors)
        
        # 添加样本数量标签
        for i, (bar, count) in enumerate(zip(bars, sample_counts)):
            height = bar.get_height()
            if height >= 0:
                y_pos = height + 0.1
            else:
                y_pos = height - 0.3
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'n={count}', ha='center', va='bottom', fontsize=8)
        
        # 添加标签和标题
        ax.set_xlabel('叶节点ID')
        ax.set_ylabel('T值 (控制有效性)')
        ax.set_title('叶节点控制有效性分析')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # 添加文本说明
        textstr = '\n'.join([
            '正T值: 控制减少了结果值',
            '负T值: 控制增加了结果值',
            f'平均T值: {np.mean(T_values):.4f}'
        ])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # 添加网格
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        return fig
    
    def create_effectiveness_report(self, tree_model, effectiveness, state_names=None):
        """
        创建叶节点有效性报告
        
        Args:
            tree_model: 树形agent模型
            effectiveness: 叶节点有效性字典
            state_names: 状态变量名称列表
            
        Returns:
            report: 报告文本
        """
        # 获取决策规则
        rules = tree_model.get_decision_rules()
        
        # 创建报告
        report = []
        report.append("# 叶节点有效性报告")
        report.append("")
        
        # 添加总体统计
        T_values = [stats['T'] for _, stats in effectiveness.items() if stats['T'] is not None]
        if T_values:
            report.append(f"## 总体有效性")
            report.append(f"* 平均T值: {np.mean(T_values):.4f}")
            report.append(f"* 有效叶节点数: {len(T_values)}/{len(effectiveness)}")
            report.append("")
        
        # 按T值排序的叶节点
        sorted_leaves = sorted(
            [(leaf_id, stats) for leaf_id, stats in effectiveness.items() if stats['T'] is not None],
            key=lambda x: x[1]['T'],
            reverse=True
        )
        
        # 添加最有效的叶节点
        if sorted_leaves:
            report.append("## 最有效的叶节点")
            for leaf_id, stats in sorted_leaves[:3]:
                if stats['T'] > 0:
                    report.append(f"### 叶节点 {leaf_id} (T = {stats['T']:.4f}, 样本数 = {stats['sample_count']})")
                    report.append(f"* 规则: {rules[leaf_id]}")
                    report.append(f"* 平均结果: {stats['mean']:.4f}")
                    report.append("")
            
            # 添加最无效的叶节点
            report.append("## 最无效的叶节点")
            for leaf_id, stats in sorted_leaves[-3:]:
                if stats['T'] < 0:
                    report.append(f"### 叶节点 {leaf_id} (T = {stats['T']:.4f}, 样本数 = {stats['sample_count']})")
                    report.append(f"* 规则: {rules[leaf_id]}")
                    report.append(f"* 平均结果: {stats['mean']:.4f}")
                    report.append("")
        
        return "\n".join(report)