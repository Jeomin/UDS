# -*- coding: utf-8 -*-
"""
分析 PPO 模型的可解释性
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import environment.SWMM_ENV as SWMM_ENV
import agents.PPO as PPO
from interpretability.explainer import Explainer
from interpretability.tree_surrogate import TreeSurrogateModel
from interpretability.sensitivity import SensitivityAnalysis
from interpretability.conditional_prob import ConditionalProbabilityAnalysis
from utils.visualization import (
    plot_control_history, plot_explanations, plot_tree_surrogate_model,
    plot_sensitivity_heatmap, plot_conditional_probability, plot_interpretability_indices
)


def analyze_ppo(args):
    """
    分析 PPO 模型
    
    Args:
        args: 命令行参数
    """
    # 创建输出目录
    output_dir = os.path.join(
        args.output_dir,
        f"analyze_ppo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        'action_dim': len(env.config['action_assets']),
        'actornet_layer_A': 3,
        'actornet_A': [{'num': 30}, {'num': 30}, {'num': 30}],
        'bound_low': 0,
        'bound_high': 1,
        'evalnet_layer_V': 3,
        'evalnet_V': [{'num': 30}, {'num': 30}, {'num': 30}],
        'clip_ratio': 0.01,
        'target_kl': 0.03,
        'lam': 0.01,
        'policy_learning_rate': 0.001,
        'value_learning_rate': 0.001,
        'num_rain': args.num_train,
        'training_step': 5,
        'gamma': 0.3,
        'epsilon': 0.1,
        'ep_min': 0.01,
        'ep_decay': 0.1
    }
    agent = PPO.PPO(agent_params, env)
    
    # 加载预训练模型
    if args.model_path:
        try:
            actor_path = os.path.join(os.path.dirname(args.model_path), 'PPOactor.h5')
            critic_path = os.path.join(os.path.dirname(args.model_path), 'PPOcritic.h5')
            
            agent.actor.load_weights(actor_path)
            agent.critic.load_weights(critic_path)
            print(f"已加载预训练模型: {actor_path}, {critic_path}")
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
    rain_path = 'test_raindata.npy'
    try:
        rainfall_data = np.load(rain_path, allow_pickle=True).tolist()
        print(f"成功加载降雨数据: {len(rainfall_data)} 个样本")
    except Exception as e:
        print(f"加载降雨数据出错: {e}")
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
        logits = []
        rewards = []
        floodings = []
        csos = []
        
        while not done:
            # PPO 返回 logits 和 action
            logit, action = agent.choose_action(s, False)
            
            s_next, reward, flooding, cso, done = env.step(action[0].tolist())
            
            states.append(s)
            actions.append(action[0])
            logits.append(logit[0])
            rewards.append(reward)
            floodings.append(flooding)
            csos.append(cso)
            
            dataset.append((s, action[0], logit[0], flooding, cso))
            
            s = s_next
        
        # 保存测试结果
        test_result = {
            'state': states,
            'action': actions,
            'logits': logits,
            'reward': rewards,
            'F': floodings,
            'C': csos
        }
        np.save(os.path.join(output_dir, f'test_{i+1}_results.npy'), test_result)
        
        # 可视化测试结果
        fig = plot_control_history(test_result)
        fig.savefig(os.path.join(output_dir, f'test_{i+1}_history.png'))
        plt.close(fig)
    
    print("训练树agent模型...")
    tree_model = TreeSurrogateModel(
        state_names=state_names,
        max_depth=args.tree_depth
    )
    
    # 提取状态和动作
    states = np.array([item[0] for item in dataset])
    actions = np.array([item[1] for item in dataset])
    
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
        # 对于 PPO，分析每个泵的敏感性
        I1 = 0
        for pump_idx in range(len(env.config['action_assets'])):
            print(f"分析泵 {pump_idx + 1}/{len(env.config['action_assets'])}")
            sensitivity_results = sensitivity_analyzer.analyze(
                state_bounds, 
                n_samples=500, 
                output_index=pump_idx
            )
            
            # 保存敏感性分析结果
            np.save(
                os.path.join(output_dir, f'sensitivity_results_pump_{pump_idx}.npy'), 
                sensitivity_results
            )
            
            # 绘制敏感性热力图
            fig = plot_sensitivity_heatmap(
                sensitivity_results, 
                state_names, 
                figsize=(12, 8)
            )
            fig.savefig(os.path.join(output_dir, f'sensitivity_heatmap_pump_{pump_idx}.png'))
            plt.close(fig)
            
            # 计算敏感性指数
            pump_I1 = sensitivity_analyzer.calculate_interpretability_index(sensitivity_results)
            I1 += pump_I1 / len(env.config['action_assets'])  # 取平均值
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
    
    # 绘制条件概率分s布
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