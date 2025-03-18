# -*- coding: utf-8 -*-
"""
可视化工具模块，用于可视化解释结果
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def plot_control_history(history, plot_items=None, figsize=(15, 10)):
    """
    绘制控制历史
    
    Args:
        history: 历史数据字典
        plot_items: 要绘制的项目列表
        figsize: 图表大小
        
    Returns:
        fig
    """
    if plot_items is None:
        # 默认绘制状态、动作、奖励、洪水和CSO
        plot_items = ['state', 'action', 'reward', 'F', 'C']
        
    n_items = len(plot_items)
    
    fig, axes = plt.subplots(n_items, 1, figsize=figsize)
    if n_items == 1:
        axes = [axes]
        
    for i, item in enumerate(plot_items):
        if item in history:
            data = np.array(history[item])
            
            if len(data.shape) == 1:
                axes[i].plot(data)
                axes[i].set_title(f'{item}')
                axes[i].grid(True)
            elif len(data.shape) == 2:
                for j in range(data.shape[1]):
                    axes[i].plot(data[:, j], label=f'{item}_{j}')
                axes[i].set_title(f'{item}')
                axes[i].grid(True)
                if data.shape[1] <= 10:
                    axes[i].legend()
                    
    plt.tight_layout()
    return fig

def plot_explanations(history, explanations, figsize=(15, 15)):
    """
    绘制解释结果与控制历史的对比
    
    Args:
        history: 历史数据字典
        explanations: 解释数据列表
        figsize: 图表大小
        
    Returns:
        fig
    """
    fig, axes = plt.subplots(5, 1, figsize=figsize)
    
    # 1. 绘制洪水和CSO
    ax1 = axes[0]
    if 'F' in history and 'C' in history:
        ax1.plot(history['F'], label='洪水量')
        ax1.plot(history['C'], label='CSO排放量')
        ax1.set_title('洪水和CSO排放量')
        ax1.legend()
        ax1.grid(True)
        
    # 2. 绘制奖励
    ax2 = axes[1]
    if 'reward' in history:
        ax2.plot(history['reward'])
        ax2.set_title('奖励')
        ax2.grid(True)
        
    # 3. 绘制关键状态变量
    ax3 = axes[2]
    if 'state' in history:
        states = np.array(history['state'])
        # 选择一些关键状态变量（水位、流量和降雨）
        key_indices = [0, 1, 8, 9, 17]
        for i in key_indices:
            if i < states.shape[1]:
                ax3.plot(states[:, i], label=f'状态_{i}')
        ax3.set_title('关键状态变量')
        ax3.legend()
        ax3.grid(True)
        
    # 4. 绘制泵状态
    ax4 = axes[3]
    if 'action' in history:
        actions = np.array(history['action'])
        if len(actions.shape) == 2 and actions.shape[1] <= 10:
            for i in range(actions.shape[1]):
                ax4.plot(actions[:, i], label=f'泵_{i}')
        ax4.set_title('泵状态')
        ax4.legend()
        ax4.grid(True)
        
    # 5. 绘制场景变化
    ax5 = axes[4]
    if explanations:
        scenarios = [exp.get('scenario', 'Unknown') if isinstance(exp, dict) else 'Unknown' for exp in explanations]
        unique_scenarios = list(set(scenarios))
        scenario_indices = {scenario: i for i, scenario in enumerate(unique_scenarios)}
        scenario_values = [scenario_indices.get(scenario, -1) for scenario in scenarios]

        ax5.plot(scenario_values, 'o-')
        ax5.set_yticks(list(range(len(unique_scenarios))))
        ax5.set_yticklabels(unique_scenarios)
        ax5.set_title('场景变化')
        ax5.grid(True)
        
    plt.tight_layout()
    return fig

def plot_tree_surrogate_model(tree_model, state_names=None, action_names=None, figsize=(20, 15)):
    """
    绘制树形agent模型
    
    Args:
        tree_model: 树形agent模型
        state_names: 状态变量名称列表
        action_names: 动作名称列表
        figsize: 图表大小
        
    Returns:
        fig
    """
    return tree_model.visualize(
        feature_names=state_names,
        class_names=action_names,
        figsize=figsize
    )

def plot_feature_importance(tree_model, figsize=(12, 6)):
    """
    绘制特征重要性
    
    Args:
        tree_model: 树形agent模型
        figsize: 图表大小
        
    Returns:
        fig:
    """
    importance = tree_model.get_feature_importance()

    fig, ax = plt.subplots(figsize=figsize)

    names = list(importance.keys())
    values = list(importance.values())
    
    y_pos = np.arange(len(names))
    
    # 根据重要性排序
    sorted_indices = np.argsort(values)
    names = [names[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    
    ax.barh(y_pos, values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('特征重要性')
    ax.set_title('状态变量重要性排序')
    
    plt.tight_layout()
    return fig

def plot_sensitivity_heatmap(sensitivities, state_names=None, figsize=(12, 8)):
    """
    绘制敏感性热力图
    
    Args:
        sensitivities: 敏感性分析结果
        state_names: 状态变量名称列表
        figsize: 图表大小
        
    Returns:
        fig
    """
    # 获取总敏感性指数
    ST = sensitivities['ST']
    
    if state_names is None or len(state_names) != len(ST):
        state_names = [f'状态_{i}' for i in range(len(ST))]
        
    heatmap_data = np.array([ST]).T
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=['敏感性指数'],
                yticklabels=state_names,
                ax=ax)
    
    ax.set_title('状态变量敏感性分析')
    plt.tight_layout()
    
    return fig

def plot_conditional_probability(distributions, figsize=(12, 8)):
    """
    绘制条件概率分布
    
    Args:
        distributions: 条件概率分布字典
        figsize: 图表大小
        
    Returns:
        fig:
    """
    fig, ax = plt.subplots(figsize=figsize)

    leaf_ids = list(distributions.keys())
    means = [stats['mean'] for stats in distributions.values()]
    stds = [stats['std'] if 'std' in stats and stats['std'] is not None else 0 for stats in distributions.values()]
    counts = [stats['count'] for stats in distributions.values()]
    
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(means) if means else 0, max(means) if means else 1)
    colors = [cmap(norm(mean)) for mean in means]
    
    bars = ax.bar(range(len(leaf_ids)), means, yerr=stds, capsize=5, color=colors)
    
    for i, (bar, count, std) in enumerate(zip(bars, counts, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + std * 0.5,
                f'n={count}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(range(len(leaf_ids)))
    ax.set_xticklabels([f'叶节点_{leaf_id}' for leaf_id in leaf_ids])
    ax.set_xlabel('叶节点')
    ax.set_ylabel('结果均值')
    ax.set_title('不同叶节点的结果分布')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('均值')
    
    plt.tight_layout()
    return fig

def visualize_explanation(explanation, figsize=(10, 5)):
    """
    可视化单个解释
    
    Args:
        explanation: 解释数据字典
        figsize: 图表大小
        
    Returns:
        fig
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    scenario = explanation.get('scenario', 'Unknown')
    decision_reason = explanation.get('decision_reason', 'Unknown')
    predicted_consequence = explanation.get('predicted_consequence', 'Unknown')
    pump_status = explanation.get('pump_status', [])
    
    text = f"场景: {scenario}\n\n"
    text += f"决策理由: {decision_reason}\n\n"
    text += f"预期后果: {predicted_consequence}\n\n"
    
    if pump_status:
        text += "泵状态:\n"
        for status in pump_status:
            text += f"  {status}\n"
    
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top')
    
    ax.axis('off')
    
    return fig

def plot_interpretability_indices(I1, I2, I3, figsize=(8, 6)):
    """
    绘制三个可解释性指数
    
    Args:
        I1: 敏感性指数
        I2: Gini不纯度指数
        I3: 条件概率指数
        figsize: 图表大小
        
    Returns:
        fig
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    indices = ['I1 (敏感性)', 'I2 (决策树)', 'I3 (条件概率)']
    values = [I1, I2, I3]
    
    bars = ax.bar(indices, values, color=['blue', 'green', 'orange'])
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    ax.set_title('可解释性指数比较')
    ax.set_ylabel('指数值')
    ax.set_ylim(0, 1.2)
    textstr = '\n'.join([
        'I1: 敏感性指数 (越高越好)',
        'I2: Gini不纯度指数 (越高越好)',
        'I3: 条件概率指数 (越高越好)'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig

def plot_decision_path(soft_tree, state, figsize=(10, 8)):
    """
    可视化软决策树的决策路径
    
    Args:
        soft_tree: 软决策树对象
        state: 输入状态
        figsize: 图表大小
        
    Returns:
        fig: matplotlib图表
    """
    # 获取决策路径和预测
    path, prediction = soft_tree.predict_with_path(state)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    depth = soft_tree.depth
    
    coords = {}
    for i, node_id in enumerate(path):
        level = i
        x = level / (depth - 1)
        
        max_nodes_at_level = 2**level
        node_index_at_level = node_id - (2**level - 1)
        y = 1.0 - node_index_at_level / max_nodes_at_level
        
        coords[node_id] = (x, y)
    
    for i in range(len(path) - 1):
        parent_id = path[i]
        child_id = path[i+1]
        ax.plot([coords[parent_id][0], coords[child_id][0]],
                [coords[parent_id][1], coords[child_id][1]],
                'k-', linewidth=2)
    
    for i, node_id in enumerate(path):
        x, y = coords[node_id]
        
        if i == len(path) - 1:
            ax.plot(x, y, 'ro', markersize=15)
        else:
            ax.plot(x, y, 'bo', markersize=10)
            
        # 规则标签
        if i < len(path) - 1:
            rule = soft_tree.get_decision_rule(node_id)
            ax.text(x + 0.05, y, rule, fontsize=10, ha='left', va='center')
    
    x, y = coords[path[-1]]
    ax.text(x + 0.05, y, f"预测: {prediction}", fontsize=12, ha='left', va='center')
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title('决策路径可视化')
    ax.axis('off')
    
    return fig
