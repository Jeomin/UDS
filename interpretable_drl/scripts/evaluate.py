#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型评估模块
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_rainfall_data(filepath):
    """加载降雨数据"""
    try:
        rainfall_data = np.load(filepath, allow_pickle=True).tolist()
        print(f"成功加载降雨数据: {len(rainfall_data)} 个样本")
        return rainfall_data
    except Exception as e:
        print(f"加载降雨数据出错: {e}")
        return None


def setup_environment(env_path):
    """设置SWMM环境"""
    import environment.SWMM_ENV as SWMM_ENV
    
    env_params = {
        'orf': env_path,
        'advance_seconds': 300
    }
    
    env = SWMM_ENV.SWMM_ENV(env_params)
    return env


def evaluate_intrinsic(args):
    """评估内生可解释系统"""
    from interpretability.intrinsic_soft_tree import IntrinsicSoftTree
    from interpretability.specialized_agents import SpecializedAgentManager
    from interpretability.intrinsic_explainer import IntrinsicExplainer

    output_dir = os.path.join(
        args.output_dir,
        f"evaluate_intrinsic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)

    env = setup_environment(args.env_path)

    rainfall_data = load_rainfall_data(args.rain_path)
    if rainfall_data is None:
        print("无法加载降雨数据，退出")
        return

    test_data = rainfall_data[:args.num_test]

    if args.model_path is None:
        print("未指定模型路径，无法评估")
        return
        
    tree_path = os.path.join(args.model_path, "final_soft_tree.pkl")
    agent_dir = os.path.join(args.model_path, "agents")
    
    if not os.path.exists(tree_path) or not os.path.exists(agent_dir):
        print(f"模型文件不存在: {tree_path} 或 {agent_dir}")
        return
    
    print(f"加载决策树模型: {tree_path}")
    soft_tree = IntrinsicSoftTree(
        input_dim=len(env.config['states']),
        num_classes=args.num_classes,
        depth=args.tree_depth,
        temperature=args.temperature
    )
    soft_tree.load(tree_path)
    
    if args.model == 'ppo' or args.model == 'intrinsic':  # 默认使用PPO
        import agents.PPO as PPO
        agent_class = PPO.PPO
        agent_params = {
            'state_dim': len(env.config['states']),
            'action_dim': len(env.config['action_assets']),
            'actornet_layer_A': 3,
            'actornet_A': [{'num': 30}, {'num': 30}, {'num': 30}],
            'bound_low': 0,
            'bound_high': 1,
            'evalnet_layer_V': 3,
            'evalnet_V': [{'num': 30}, {'num': 30}, {'num': 30}],
        }
    else:  # DQN
        import agents.DQN as DQN
        agent_class = DQN.DQN
        agent_params = {
            'state_dim': len(env.config['states']),
            'action_dim': 2**len(env.config['action_assets']),
            'evalnet_layer_A': 3,
            'evalnet_A': [{'num': 30}, {'num': 30}, {'num': 30}],
            'evalnet_layer_V': 3,
            'evalnet_V': [{'num': 30}, {'num': 30}, {'num': 30}],
        }
    
    print(f"加载agent模型: {agent_dir}")
    agent_manager = SpecializedAgentManager(
        env=env,
        num_classes=args.num_classes,
        agent_class=agent_class,
        base_params=agent_params
    )
    agent_manager.load_models(agent_dir)
    
    # 创建解释器
    explainer = IntrinsicExplainer(
        soft_tree=soft_tree, 
        env_config=env.config,
        state_names=[f"特征_{i}" for i in range(len(env.config['states']))]
    )
    
    results = {
        'rewards': [],
        'flooding': [],
        'cso': [],
        'steps': [],
        'class_distribution': np.zeros(args.num_classes)
    }
    
    for i, rain in enumerate(test_data):
        print(f"评估样本 {i+1}/{len(test_data)}...")
        
        s = env.reset(rain)
        done = False
        episode_reward = 0
        episode_flooding = 0
        episode_cso = 0
        step_count = 0
        
        # 收集解释和状态
        explanations = []
        states = []
        actions = []
        class_ids = []
        
        while not done:
            embedding = soft_tree.generate_embedding(s)
            class_id = np.argmax(embedding)
            results['class_distribution'][class_id] += 1

            action, _ = agent_manager.choose_action(s, embedding, train_mode=False)

            s_next, reward, flooding, cso, done = env.step(action)

            explanation = explainer.generate_explanation(s, embedding, action, class_id)
            explanations.append(explanation)

            states.append(s)
            actions.append(action)
            class_ids.append(class_id)
            
            episode_reward += reward
            episode_flooding += flooding
            episode_cso += cso
            step_count += 1
            
            s = s_next
        
        results['rewards'].append(episode_reward)
        results['flooding'].append(episode_flooding)
        results['cso'].append(episode_cso)
        results['steps'].append(step_count)
        
        explanation_path = os.path.join(output_dir, f"explanations_sample_{i}.txt")
        with open(explanation_path, 'w') as f:
            for j, exp in enumerate(explanations):
                f.write(f"Step {j}:\n")
                for key, value in exp.items():
                    if key != 'pump_status':
                        f.write(f"  {key}: {value}\n")
                    else:
                        f.write(f"  {key}:\n")
                        for pump in value:
                            f.write(f"    {pump}\n")
                f.write("\n")
        
        history = {
            'states': states,
            'actions': actions,
            'class_ids': class_ids,
            'reward': episode_reward,
            'flooding': episode_flooding,
            'cso': episode_cso
        }
        np.save(os.path.join(output_dir, f"history_sample_{i}.npy"), history)
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(3, 1, 1)
        states_array = np.array(states)
        for j in range(min(5, states_array.shape[1])):  # 只显示前5个状态变量
            plt.plot(states_array[:, j], label=f'State {j}')
        plt.title(f'State Variables (sample {i})')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(class_ids)
        plt.title(f'Scene Classification (sample {i})')
        plt.yticks(range(args.num_classes))
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        actions_array = np.array(actions)
        if len(actions_array.shape) > 1:
            for j in range(min(7, actions_array.shape[1])):  # 最多显示7个泵
                plt.plot(actions_array[:, j], label=f'Pump {j}')
        else:
            plt.plot(actions_array, label='Action Index')
        plt.title(f'Pump Control (sample {i})')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"control_process_sample_{i}.png"))
        plt.close()
    
    avg_reward = np.mean(results['rewards'])
    avg_flooding = np.mean(results['flooding'])
    avg_cso = np.mean(results['cso'])
    avg_steps = np.mean(results['steps'])
    
    class_dist = results['class_distribution'] / np.sum(results['class_distribution'])
    
    with open(os.path.join(output_dir, "evaluation_summary.txt"), 'w') as f:
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型路径: {args.model_path}\n")
        f.write(f"测试样本数: {len(test_data)}\n")
        f.write(f"平均奖励: {avg_reward:.4f}\n")
        f.write(f"平均洪水: {avg_flooding:.4f}\n")
        f.write(f"平均CSO: {avg_cso:.4f}\n")
        f.write(f"平均步数: {avg_steps:.2f}\n\n")
        f.write("场景分布:\n")
        for i in range(len(class_dist)):
            f.write(f"  场景 {i}: {class_dist[i]:.4f}\n")
    
    plt.figure(figsize=(15, 10))
    
    # 绘制奖励、洪水和CSO
    plt.subplot(2, 1, 1)
    plt.bar(range(len(test_data)), results['rewards'], label='Reward')
    plt.bar(range(len(test_data)), results['flooding'], label='Flooding')
    plt.bar(range(len(test_data)), results['cso'], label='CSO')
    plt.title(f'Performance Metrics (Avg Reward: {avg_reward:.4f}, Flooding: {avg_flooding:.4f}, CSO: {avg_cso:.4f})')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.bar(range(len(class_dist)), class_dist)
    plt.title('Scene Distribution')
    plt.xlabel('Scene ID')
    plt.ylabel('Frequency')
    plt.xticks(range(len(class_dist)))
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation_summary.png"))
    plt.close()
    
    print(f"评估完成，结果保存在 {output_dir}")
    print(f"平均奖励: {avg_reward:.4f}")
    print(f"平均洪水: {avg_flooding:.4f}")
    print(f"平均CSO: {avg_cso:.4f}")


# def evaluate_dqn(args):
#     """评估DQN模型"""
#     from interpretability.tree_surrogate import TreeSurrogateModel
#     from interpretability.sensitivity import SensitivityAnalysis
#     from interpretability.conditional_prob import ConditionalProbabilityAnalysis
#     from interpretability.explainer import Explainer
#     import agents.DQN as DQN
    
#     # 创建输出目录
#     output_dir = os.path.join(
#         args.output_dir,
#         f"evaluate_dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#     )
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 设置环境
#     env = setup_environment(args.env_path)
    
#     # 加载降雨数据
#     rainfall_data = load_rainfall_data(args.rain_path)
#     if rainfall_data is None:
#         print("无法加载降雨数据，退出")
#         return
    
#     # 限制测试样本数量
#     test_data = rainfall_data[:args.num_test]
    
#     # 加载DQN模型
#     agent_params = {
#         'state_dim': len(env.config['states']),
#         'action_dim': 2**len(env.config['action_assets']),
#         'evalnet_layer_A': 3,
#         'evalnet_A': [{'num': 30}, {'num': 30}, {'num': 30}],
#         'evalnet_layer_V': 3,
#         'evalnet_V': [{'num': 30}, {'num': 30}, {'num': 30}],
#         'targetnet_layer_A': 3,
#         'targetnet_A': [{'num': 30}, {'num': 30}, {'num': 30}],
#         'targetnet_layer_V': 3,
#         'targetnet_V': [{'num': 30}, {'num': 30}, {'num': 30}],
#         'gamma': 0.3,
#         'epsilon': 0.0,  # 评估时不使用探索
#     }
    
#     agent = DQN.DQN(agent_params, env)
    
#     if args.model_path:
#         try:
#             agent.load_model(args.model_path)
#             print(f"已加载模型: {args.model_path}")
#         except Exception as e:
#             print(f"加载模型出错: {e}")
#             return
    
#     # 创建状态变量名称列表
#     state_names = []
#     for item in env.config['states']:
#         if len(item) >= 2:
#             if item[1] == 'depthN':
#                 state_names.append(f"{item[0]}水位")
#             elif item[1] == 'flow':
#                 state_names.append(f"{item[0]}流量")
#             elif item[1] == 'inflow':
#                 state_names.append(f"{item[0]}入流")
#             else:
#                 state_names.append(f"降雨强度")
#         else:
#             state_names.append(f"状态_{len(state_names)}")
    
#     # 收集数据
#     dataset = []
#     results = {
#         'rewards': [],
#         'flooding': [],
#         'cso': [],
#         'steps': []
#     }
    
#     # 评估每个测试样本
#     for i, rain in enumerate(test_data):
#         print(f"评估样本 {i+1}/{len(test_data)}...")
        
#         # 模拟控制过程
#         s = env.reset(rain)
#         done = False
#         episode_reward = 0
#         episode_flooding = 0
#         episode_cso = 0
#         step_count = 0
        
#         # 收集状态和动作
#         states = []
#         actions = []
#         rewards = []
#         floodings = []
#         csos = []
        
#         while not done:
#             # 选择动作
#             a = agent.choose_action(s, False)
            
#             # 获取泵状态
#             if isinstance(a, (int, np.integer)):
#                 action = agent.action_table[a, :].tolist()
#             else:
#                 action = a
            
#             # 执行动作
#             s_next, reward, flooding, cso, done = env.step(action)
            
#             # 记录数据
#             states.append(s)
#             actions.append(a)  # 保存动作索引
#             rewards.append(reward)
#             floodings.append(flooding)
#             csos.append(cso)
            
#             # 更新统计信息
#             episode_reward += reward
#             episode_flooding += flooding
#             episode_cso += cso
#             step_count += 1
            
#             # 为后解释收集数据
#             dataset.append((s, a, flooding + cso))
            
#             # 进入下一状态
#             s = s_next
        
#         # 保存结果
#         results['rewards'].append(episode_reward)
#         results['flooding'].append(episode_flooding)
#         results['cso'].append(episode_cso)
#         results['steps'].append(step_count)
        
#         # 保存控制历史
#         history = {
#             'states': states,
#             'actions': actions,
#             'rewards': rewards,
#             'floodings': floodings,
#             'csos': csos
#         }
#         np.save(os.path.join(output_dir, f"history_sample_{i}.n