#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练解释性 DQN agent
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入必要的模块
from agents.base_agent import BaseAgent
import agents.DQN as DQN
from interpretability.explainer import Explainer
from interpretability.soft_tree import SoftDecisionTree
from interpretability.tree_surrogate import TreeSurrogateModel
from interpretability.sensitivity import SensitivityAnalysis
from interpretability.conditional_prob import ConditionalProbabilityAnalysis
from utils.visualization import (
    plot_control_history, plot_explanations, plot_tree_surrogate_model,
    plot_sensitivity_heatmap, plot_conditional_probability, plot_interpretability_indices
)
import environment.SWMM_ENV as SWMM_ENV


def parse_args():
    """
    解析命令行参数
    
    Returns:
        args: 解析后的参数
    """
    parser = argparse.ArgumentParser(description='训练解释性 DQN agent')
    
    # 模型参数
    parser.add_argument('--max-depth', type=int, default=4,
                       help='树模型最大深度')
    
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='软决策树温度参数')
    
    parser.add_argument('--num-train', type=int, default=50,
                       help='训练样本数量')
    
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    
    # 路径设置
    parser.add_argument('--output-dir', type=str, default='./output',
                   help='输出目录（相对于项目根目录）')
    
    parser.add_argument('--model-path', type=str, default=None,
                       help='加载预训练模型的路径')
                       
    parser.add_argument('--env-path', type=str, default='chaohu',
                  help='SWMM环境配置文件名称 不含路径')
    
    parser.add_argument('--rain-path', type=str, default='training_raindata.npy',
                  help='训练降雨数据文件名(不含路径)')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    return args


def load_rainfall_data(filepath):
    """
    加载降雨数据
    
    Args:
        filepath: 降雨数据文件路径
        
    Returns:
        rainfall_data: 降雨数据列表
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        rainfall_path = os.path.join(base_dir, "data", "rainfall", os.path.basename(filepath))
        
        rainfall_data = np.load(rainfall_path, allow_pickle=True).tolist()
        print(f"成功加载降雨数据: {len(rainfall_data)} 个样本")
        return rainfall_data
    except Exception as e:
        print(f"加载降雨数据出错: {e}")
        return None


def setup_environment(env_name):
    """
    设置 SWMM 环境
    
    Args:
        env_name: 环境配置文件名称（不含路径和扩展名）
        
    Returns:
        env: SWMM 环境
    """
    # 构建完整路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(base_dir, "data", "env", env_name)
    
    env_params = {
        'orf': env_path,
        'advance_seconds': 300
    }
    
    env = SWMM_ENV.SWMM_ENV(env_params)
    return env


def setup_agent(env, args):
    """
    设置 DQN agent
    
    Args:
        env: SWMM 环境
        args: 命令行参数
        
    Returns:
        agent: DQN agent
        tree_model: 树模型
        explainer: 解释器
    """
    # DQN 参数
    agent_params = {
        'state_dim': len(env.config['states']),
        'action_dim': 2**len(env.config['action_assets']),
        'evalnet_layer_A': 3,
        'evalnet_A': [{'num': 30}, {'num': 30}, {'num': 30}],
        'evalnet_layer_V': 3,
        'evalnet_V': [{'num': 30}, {'num': 30}, {'num': 30}],
        'targetnet_layer_A': 3,
        'targetnet_A': [{'num': 30}, {'num': 30}, {'num': 30}],
        'targetnet_layer_V': 3,
        'targetnet_V': [{'num': 30}, {'num': 30}, {'num': 30}],
        'num_rain': args.num_train,
        'training_step': 5,
        'gamma': 0.3,
        'epsilon': 0.1,
        'ep_min': 0.01,
        'ep_decay': 0.1
    }
    
    # 创建 DQN agent
    agent = DQN(agent_params, env)
    
    # 如果提供了模型路径，加载预训练模型
    if args.model_path:
        try:
            agent.model.load_weights(args.model_path)
            print(f"已加载预训练模型: {args.model_path}")
        except Exception as e:
            print(f"加载预训练模型出错: {e}")
    
    # 创建状态变量名称列表
    state_names = []
    for item in env.config['states']:
        if len(item) >= 2:
            if item[1] == 'depthN':
                state_names.append(f"{item[0]}水位")
            elif item[1] == 'flow':
                state_names.append(f"{item[0]}流量")
            elif item[1] == 'inflow':
                state_names.append(f"{item[0]}入流")
            else:
                state_names.append(f"降雨强度")
        else:
            state_names.append(f"状态_{len(state_names)}")
    
    # 创建软决策树
    soft_tree = SoftDecisionTree(
        input_dim=len(env.config['states']),
        output_dim=agent_params['action_dim'],
        depth=args.max_depth,
        temperature=args.temperature
    )
    
    # 创建树agent模型
    tree_model = TreeSurrogateModel(
        state_names=state_names,
        max_depth=args.max_depth
    )
    
    # 创建解释器
    explainer = Explainer(state_names=state_names)
    
    return agent, tree_model, explainer, soft_tree


def collect_data_for_surrogate(agent, env, rainfall_data, num_samples=10):
    """
    收集用于训练替代模型的数据
    
    Args:
        agent: DQN agent
        env: SWMM 环境
        rainfall_data: 降雨数据
        num_samples: 收集的样本数量
        
    Returns:
        dataset: 数据集，包含 (状态, 动作, 动作值, 洪水, CSO) 元组
    """
    dataset = []
    
    # 限制样本数量
    rainfall_samples = rainfall_data[:num_samples]
    
    for rain in rainfall_samples:
        s = env.reset(rain)
        done = False
        
        while not done:
            # 使用agent选择动作
            a = agent.choose_action(s, False)
            
            # 转换动作索引为泵状态
            if isinstance(a, (int, np.integer)):
                action = agent.action_table[a, :].tolist()
            else:
                action = a
            
            # 执行动作并获取下一状态和奖励
            s_next, reward, flooding, cso, done = env.step(action)
            
            # 保存数据
            dataset.append((s, a, None, flooding, cso))
            
            # 更新状态
            s = s_next
            
    return dataset


def train_surrogate_models(agent, env, dataset, soft_tree, tree_model, args, output_dir):
    """
    训练替代模型
    
    Args:
        agent: DQN agent
        env: SWMM 环境
        dataset: 数据集
        soft_tree: 软决策树
        tree_model: 树模型
        args: 命令行参数
        output_dir: 输出目录
        
    Returns:
        trained_soft_tree: 训练后的软决策树
        trained_tree_model: 训练后的树模型
    """
    # 提取状态和动作
    states = np.array([item[0] for item in dataset])
    actions = np.array([item[1] for item in dataset])
    
    # 训练软决策树
    print("训练软决策树...")
    soft_tree.train(states, actions, epochs=args.epochs)
    
    # 保存软决策树
    soft_tree.save(os.path.join(output_dir, 'soft_tree.pkl'))
    
    # 训练树模型
    print("训练树agent模型...")
    tree_model.train(states, actions)
    
    # 保存树模型
    tree_model.save(os.path.join(output_dir, 'tree_model.pkl'))
    
    # 绘制树模型
    fig = plot_tree_surrogate_model(tree_model)
    fig.savefig(os.path.join(output_dir, 'tree_model.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return soft_tree, tree_model


def analyze_interpretability(agent, env, tree_model, dataset, output_dir):
    """
    分析模型可解释性
    
    Args:
        agent: DQN agent
        env: SWMM 环境
        tree_model: 树模型
        dataset: 数据集
        output_dir: 输出目录
        
    Returns:
        I1: 敏感性指数
        I2: Gini不纯度指数
        I3: 条件概率指数
    """
    # 创建状态变量名称列表
    state_names = []
    for item in env.config['states']:
        if len(item) >= 2:
            if item[1] == 'depthN':
                state_names.append(f"{item[0]}水位")
            elif item[1] == 'flow':
                state_names.append(f"{item[0]}流量")
            elif item[1] == 'inflow':
                state_names.append(f"{item[0]}入流")
            else:
                state_names.append(f"降雨强度")
        else:
            state_names.append(f"状态_{len(state_names)}")
    
    # 敏感性分析
    print("进行敏感性分析...")
    sensitivity_analyzer = SensitivityAnalysis(agent, state_names)
    
    # 状态变量界限
    state_bounds = []
    for i in range(len(env.config['states'])):
        if i < 8:  # 水位相关状态
            state_bounds.append((0.0, 3.0))
        elif i < 17:  # 流量相关状态
            state_bounds.append((0.0, 2000.0))
        else:  # 降雨强度
            state_bounds.append((0.0, 300.0))
    
    # 执行敏感性分析
    try:
        sensitivity_results = sensitivity_analyzer.analyze(state_bounds, n_samples=500)
        
        # 保存敏感性分析结果
        np.save(os.path.join(output_dir, 'sensitivity_results.npy'), sensitivity_results)
        
        # 绘制敏感性热力图
        fig = plot_sensitivity_heatmap(sensitivity_results, state_names)
        fig.savefig(os.path.join(output_dir, 'sensitivity_heatmap.png'))
        plt.close(fig)
        
        # 计算敏感性指数 I1
        I1 = sensitivity_analyzer.calculate_interpretability_index(sensitivity_results)
    except Exception as e:
        print(f"敏感性分析出错: {e}")
        I1 = 0
    
    # 计算 Gini 不纯度指数 I2
    print("计算 Gini 不纯度指数...")
    I2 = tree_model.calculate_interpretability_index()
    
    # 条件概率分析
    print("进行条件概率分析...")
    probability_analyzer = ConditionalProbabilityAnalysis()
    
    # 准备条件概率分析数据
    probability_dataset = [(item[0], item[1], item[3] + item[4]) for item in dataset]  # 状态, 动作, 洪水+CSO
    
    # 分析叶节点条件概率
    leaf_analysis = probability_analyzer.analyze_tree_leaves(tree_model, probability_dataset)
    
    # 保存条件概率分析结果
    np.save(os.path.join(output_dir, 'conditional_probability_results.npy'), leaf_analysis)
    
    # 绘制条件概率分布
    fig = plot_conditional_probability(leaf_analysis)
    fig.savefig(os.path.join(output_dir, 'conditional_probability.png'))
    plt.close(fig)
    
    # 计算叶节点有效性
    effectiveness = probability_analyzer.calculate_leaf_effectiveness(tree_model, probability_dataset)
    
    # 计算条件概率指数 I3
    I3 = probability_analyzer.calculate_interpretability_index(effectiveness)
    
    # 绘制可解释性指数
    fig = plot_interpretability_indices(I1, I2, I3)
    fig.savefig(os.path.join(output_dir, 'interpretability_indices.png'))
    plt.close(fig)
    
    # 保存可解释性指标
    with open(os.path.join(output_dir, 'interpretability_indices.txt'), 'w') as f:
        f.write(f"敏感性指数 (I1): {I1}\n")
        f.write(f"Gini不纯度指数 (I2): {I2}\n")
        f.write(f"条件概率指数 (I3): {I3}\n")
    
    return I1, I2, I3


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 创建输出目录
    output_dir = os.path.join(
        args.output_dir,
        f"interpretable_dqn_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置环境
    env = setup_environment(args.env_path)
    
    # 设置agent和模型
    agent, tree_model, explainer, soft_tree = setup_agent(env, args)
    
    # 加载降雨数据
    rainfall_data = load_rainfall_data(args.rain_path)
    if rainfall_data is None:
        print("无法加载降雨数据，退出")
        return
    
    # 收集数据
    print("收集训练数据...")
    dataset = collect_data_for_surrogate(agent, env, rainfall_data, args.num_train)
    
    # 训练替代模型
    print("训练替代模型...")
    soft_tree, tree_model = train_surrogate_models(
        agent, env, dataset, soft_tree, tree_model, args, output_dir
    )
    
    # 设置解释器的树模型
    explainer.set_tree_model(tree_model)
    
    # 分析可解释性
    print("分析可解释性...")
    I1, I2, I3 = analyze_interpretability(agent, env, tree_model, dataset, output_dir)
    
    # 保存模型和可解释性分析结果
    print("保存结果...")
    agent.model.save_weights(os.path.join(output_dir, 'dqn_model.h5'))
    
    print(f"训练和分析完成，输出保存在 {output_dir}")
    print(f"可解释性指数: I1={I1}, I2={I2}, I3={I3}")


if __name__ == "__main__":
    main()