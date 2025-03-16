#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
解释性 DRL 框架的主入口
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import datetime

# 确保可以导入项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
import environment.SWMM_ENV as SWMM_ENV
import agents.DQN as DQN
import agents.PPO as PPO
from interpretability.explainer import Explainer
from interpretability.soft_tree import SoftDecisionTree
from interpretability.tree_surrogate import TreeSurrogateModel
from interpretability.sensitivity import SensitivityAnalysis
from interpretability.conditional_prob import ConditionalProbabilityAnalysis
from utils.visualization import (
    plot_control_history, plot_explanations, plot_tree_surrogate_model,
    plot_sensitivity_heatmap, plot_conditional_probability, plot_interpretability_indices
)


def parse_args():
    """
    解析命令行参数
    
    Returns:
        args: 解析后的参数
    """
    parser = argparse.ArgumentParser(description='解释性 DRL 框架')
    
    # 模式选择
    parser.add_argument('--mode', type=str, default='analyze',
                        choices=['train', 'test', 'analyze'],
                        help='运行模式: train-训练, test-测试, analyze-分析')
    
    # 模型选择
    parser.add_argument('--model', type=str, default='dqn',
                        choices=['dqn', 'ppo'],
                        help='模型类型: dqn或ppo')
    
    # 可解释性设置
    parser.add_argument('--tree-depth', type=int, default=4,
                        help='决策树深度')
    
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='软决策树温度参数')
    
    # 训练设置
    parser.add_argument('--num-train', type=int, default=50,
                        help='训练降雨样本数量')
    
    parser.add_argument('--num-test', type=int, default=5,
                        help='测试降雨样本数量')
    
    # 路径设置
    parser.add_argument('--model-path', type=str, default=None,
                        help='模型加载路径')
    
    parser.add_argument('--output-dir', type=str, default='output',
                        help='输出目录')
    
    parser.add_argument('--env-path', type=str, default='DQN/chaohu',
                       help='SWMM环境配置文件路径')
    
    # 杂项
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
        rainfall_data = np.load(filepath, allow_pickle=True).tolist()
        print(f"成功加载降雨数据: {len(rainfall_data)} 个样本")
        return rainfall_data
    except Exception as e:
        print(f"加载降雨数据出错: {e}")
        return None


def analyze_dqn(args):
    """
    分析 DQN 模型
    
    Args:
        args: 命令行参数
    """
    # 创建输出目录
    output_dir = os.path.join(
        args.output_dir,
        f"analyze_dqn_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置环境
    env_params = {
        'orf': args.env_path,
        'advance_seconds': 300
    }
    env = SWMM_ENV.SWMM_ENV(env_params)
    
    # 加载模型
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
    agent = DQN.DQN(agent_params, env)
    
    # 加载预训练模型
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
    
    # 加载降雨数据
    rain_path = 'DQN/test_raindata.npy'
    rainfall_data = load_rainfall_data(rain_path)
    if rainfall_data is None:
        print("无法加载降雨数据，退出")
        return
    
    # 收集数据
    print("收集测试数据...")
    dataset = []
    for i in range(min(args.num_test, len(rainfall_data))):
        print(f"测试降雨样本 {i+1}/{min(args.num_test, len(rainfall_data))}...")
        s = env.reset(rainfall_data[i])
        done = False
        states = []
        actions = []
        rewards = []
        floodings = []
        csos = []
        
        while not done:
            a = agent.choose_action(s, False)
            if isinstance(a, (int, np.integer)):
                action = agent.action_table[a, :].tolist()
            else:
                action = a
            
            s_next, reward, flooding, cso, done = env.step(action)
            
            states.append(s)
            actions.append(a)
            rewards.append(reward)
            floodings.append(flooding)
            csos.append(cso)
            
            dataset.append((s, a, None, flooding, cso))
            
            s = s_next
        
        # 保存测试结果
        test_result = {
            'state': states,
            'action': actions,
            'reward': rewards,
            'F': floodings,
            'C': csos
        }
        np.save(os.path.join(output_dir, f'test_{i+1}_results.npy'), test_result)
        
        # 可视化测试结果
        fig = plot_control_history(test_result)
        fig.savefig(os.path.join(output_dir, f'test_{i+1}_history.png'))
        plt.close(fig)
    
    # 训练树模型
    print("训练树代理模型...")
    tree_model = TreeSurrogateModel(
        state_names=state_names,
        max_depth=args.tree_depth
    )
    
    # 提取状态和动作
    states = np.array([item[0] for item in dataset])
    actions = np.array([item[1] for item in dataset])
    
    # 训练树模型
    tree_model.train(states, actions)
    
    # 保存树模型
    tree_model.save(os.path.join(output_dir, 'tree_model.pkl'))
    
    # 绘制树模型
    fig = plot_tree_surrogate_model(tree_model)
    fig.savefig(os.path.join(output_dir, 'tree_model.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
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
    
    print(f"分析完成，输出保存在 {output_dir}")
    print(f"可解释性指数: I1={I1}, I2={I2}, I3={I3}")


def analyze_ppo(args):
    """
    分析 PPO 模型
    
    Args:
        args: 命令行参数
    """
    # 导入分析函数
    from analyze_ppo import analyze_ppo as analyze_ppo_func
    
    # 调用分析函数
    analyze_ppo_func(args)


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 根据模式执行不同操作
    if args.mode == 'train':
        if args.model == 'dqn':
            # 导入 DQN 训练脚本
            from scripts.train_interpretable_dqn import main as train_dqn
            train_dqn()
        else:  # ppo
            # 导入 PPO 训练脚本
            from scripts.train_interpretable_ppo import main as train_ppo
            train_ppo()
    elif args.mode == 'test':
        # 导入测试脚本
        print("测试模式暂未实现")
    elif args.mode == 'analyze':
        # 导入分析脚本
        if args.model == 'dqn':
            # 分析 DQN 模型
            analyze_dqn(args)
        else:  # ppo
            # 分析 PPO 模型
            analyze_ppo(args)
    else:
        print(f"不支持的模式: {args.mode}")


if __name__ == "__main__":
    main()