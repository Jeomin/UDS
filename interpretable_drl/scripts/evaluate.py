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
import argparse
import yaml 
from interpretability.intrinsic_soft_tree import IntrinsicSoftTree
from interpretability.specialized_agents import SpecializedAgentManager
from interpretability.intrinsic_explainer import IntrinsicExplainer
import environment.SWMM_ENV as SWMM_ENV

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_rainfall_data(filepath):
    """加载降雨数据"""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        rainfall_path = os.path.join(base_dir, "data", "rainfall", os.path.basename(filepath))
        
        rainfall_data = np.load(rainfall_path, allow_pickle=True).tolist()
        print(f"成功加载降雨数据: {len(rainfall_data)} 个样本")
        return rainfall_data
    except Exception as e:
        print(f"加载降雨数据出错: {e}")
        return None


def setup_environment(env_path):
    """设置SWMM环境"""
    
    env_params = {
        'orf': env_path,
        'advance_seconds': 300
    }
    
    env = SWMM_ENV.SWMM_ENV(env_params)
    return env


def evaluate_intrinsic(args):
    """评估可解释系统"""
    output_dir = os.path.join(
        args.output_dir,
        f"evaluate_intrinsic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"评估结果将保存在: {output_dir}")

    env = setup_environment(args.env_path)

    rainfall_data = load_rainfall_data(args.rain_path)
    if rainfall_data is None:
        print("无法加载降雨数据，退出")
        return

    # 选择测试数据子集
    if args.num_test > len(rainfall_data):
        print(f"警告: 请求的测试样本数 ({args.num_test}) 大于可用数据 ({len(rainfall_data)})。将使用所有可用数据。")
        args.num_test = len(rainfall_data)
    elif args.num_test <= 0:
         print(f"警告: 测试样本数 ({args.num_test}) 无效。将使用 1 个样本。")
         args.num_test = 1

    test_data = rainfall_data[:args.num_test]
    print(f"使用 {len(test_data)} 个样本进行评估。")

    if args.model_path is None or not os.path.isdir(args.model_path):
        print(f"错误: 模型路径 '{args.model_path}' 无效或未指定。")
        return
        
    tree_path = os.path.join(args.model_path, "final_soft_tree.pkl")
    agent_dir = os.path.join(args.model_path, "agents")
    
    if not os.path.exists(tree_path):
        print(f"错误: 决策树模型文件不存在: {tree_path}")
        return
    if not os.path.exists(agent_dir):
         print(f"错误: Agents 目录不存在: {agent_dir}")
         return
    
    print(f"加载决策树模型: {tree_path}")
    soft_tree = IntrinsicSoftTree(
        input_dim=len(env.config['states']),
        num_classes=args.num_classes,
        depth=args.tree_depth,
        # temperature=args.temperature
    )
    soft_tree.load(tree_path)
    if not soft_tree._is_fitted:
        print("错误: 加载的决策树模型未训练或加载失败。")
        return
    
    print(f"从: {agent_dir} 加载 Agent Manager 和 Agents。")
    agent_manager = SpecializedAgentManager.load_manager(agent_dir, env)

    if agent_manager is None:
        print("错误: 加载 Agent Manager 失败。")
        return
    if agent_manager.num_classes != args.num_classes:
        print(f"警告: 加载的 Agent Manager 类别数 ({agent_manager.num_classes}) 与决策树 ({args.num_classes}) 不符。")
        return
    print(f"Agent Manager 加载成功，Agent 类型: {agent_manager.agent_class.__name__}")
    
    # 创建解释器
    explainer = IntrinsicExplainer(
        soft_tree=soft_tree, 
        env_config=env.config,
        state_names=[f"Feature_{i}" for i in range(len(env.config['states']))]
    )
    
    results = {
        'rewards': [],
        'flooding': [],
        'cso': [],
        'steps': [],
        'class_distribution': np.zeros(args.num_classes)
    }
    all_episodes_history = [] # 存储每个 episode 的详细历史
    num_actions_expected = len(env.config["action_assets"])
    
    for i, rain in enumerate(test_data):
        print(f"评估样本 {i+1}/{len(test_data)}...")

        s = env.reset(rain)
        done = False
        episode_reward = 0
        episode_flooding_total = 0 # 记录当前 episode 的最终累积值
        episode_cso_total = 0      # 记录当前 episode 的最终累积值
        step_count = 0

        explanations = []
        episode_history = {'states': [], 'embeddings': [], 'class_ids': [], 'actions': [], 'logps': [], 'rewards': [], 'floodings': [], 'csos': []}

        while not done:
            embedding = soft_tree.generate_embedding(s)
            class_id = np.argmax(embedding)
            results['class_distribution'][class_id] += 1

            result, _ = agent_manager.choose_action(s, embedding, train_mode=False)

            action_for_env = None
            action_for_history = None
            logp_for_history = None
            agent = agent_manager.agents[class_id]

            # --- PPO ---
            if hasattr(agent, 'calculate_logp') and isinstance(result, tuple) and len(result) == 2:
                logp_action_np, action_np = result
                action_for_env = action_np.tolist()
                action_for_history = action_np
                logp_for_history = logp_action_np
                if len(action_for_env) != num_actions_expected: action_for_env = [0] * num_actions_expected; action_for_history = np.array(action_for_env)

            # --- DQN ---
            elif isinstance(result, (int, np.integer)):
                action_index = result
                action_for_history = action_index
                if hasattr(agent, 'action_table') and agent.action_table is not None:
                    action_table = agent.action_table
                    if action_index < len(action_table): action_for_env = action_table[action_index, :].tolist()
                    else: action_for_env = [int(bit) for bit in format(action_index, f'0{num_actions_expected}b')]
                else: action_for_env = [int(bit) for bit in format(action_index, f'0{num_actions_expected}b')]
                if not isinstance(action_for_env, list): action_for_env = [0] * num_actions_expected
                if len(action_for_env) != num_actions_expected: action_for_env = [0] * num_actions_expected

            else:
                action_for_env = [0] * num_actions_expected
                action_for_history = np.array(action_for_env); logp_for_history = -np.inf

            if action_for_env is not None:
                s_next, reward, current_total_flooding, current_total_cso, done = env.step(action_for_env)
            else: raise ValueError("错误 (评估): 未能确定环境动作 action_for_env")

            # --- 生成解释和记录历史 ---
            explanation = explainer.generate_explanation(s, embedding, action_for_env, class_id)
            explanations.append(explanation)

            episode_history['states'].append(s)
            episode_history['embeddings'].append(embedding)
            episode_history['class_ids'].append(class_id)
            episode_history['actions'].append(action_for_history)
            episode_history['logps'].append(logp_for_history)
            episode_history['rewards'].append(reward)
            episode_history['floodings'].append(current_total_flooding) # 记录累积值
            episode_history['csos'].append(current_total_cso)      # 记录累积值

            episode_reward += reward
            episode_flooding_total = current_total_flooding # 更新最终累积值
            episode_cso_total = current_total_cso      # 更新最终累积值
            step_count += 1
            s = s_next

        # --- Episode 结束处理 ---
        results['rewards'].append(episode_reward)
        results['flooding'].append(episode_flooding_total) # 记录最终的累积洪水
        results['cso'].append(episode_cso_total)      # 记录最终的累积 CSO
        results['steps'].append(step_count)
        all_episodes_history.append(episode_history) # 保存当前 episode 的详细历史

        # 保存解释
        explanation_path = os.path.join(output_dir, f"explanations_sample_{i}.txt")
        try:
            with open(explanation_path, 'w', encoding='utf-8') as f:
                for j, exp in enumerate(explanations):
                    f.write(f"Step {j}:\n")
                    f.write(f"  Timestamp: {exp.get('timestamp', 'N/A')}\n")
                    f.write(f"  场景: {exp.get('scene', 'N/A')}\n")
                    path_str = " -> ".join(map(str, exp.get('decision_path', []))) if exp.get('decision_path') else "N/A"
                    f.write(f"  决策路径: {path_str}\n")
                    f.write(f"  理由: {exp.get('reason', 'N/A')}\n")
                    f.write(f"  预期后果: {exp.get('consequence', 'N/A')}\n")
                    pump_str = ", ".join(exp.get('pump_status', [])) if exp.get('pump_status') else "N/A"
                    f.write(f"  泵状态: {pump_str}\n")
                    f.write(f"  备选方案: {exp.get('alternatives', 'N/A')}\n\n")
        except Exception as e:
            print(f"错误: 无法写入解释文件 {explanation_path}: {e}")

        # 保存单个样本的历史
        # history_path = os.path.join(output_dir, f"history_sample_{i}.npy")
        # try:
        #     np.save(history_path, episode_history)
        # except Exception as e:
        #      print(f"错误: 无法保存样本历史文件 {history_path}: {e}")

        # 绘制单个样本的控制过程图
        # ... (复用 evaluate_system 中的绘图代码，但针对 episode_history) ...

    # 计算平均性能指标
    avg_reward = np.mean(results['rewards']) if results['rewards'] else 0
    avg_flooding = np.mean(results['flooding']) if results['flooding'] else 0
    avg_cso = np.mean(results['cso']) if results['cso'] else 0
    avg_steps = np.mean(results['steps']) if results['steps'] else 0

    # 计算场景分布频率
    total_dist_steps = np.sum(results['class_distribution'])
    class_dist_freq = results['class_distribution'] / max(1, total_dist_steps)

    # 保存评估摘要
    summary_file = os.path.join(output_dir, "evaluation_summary.txt")
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型路径: {args.model_path}\n")
            f.write(f"Agent 类型: {args.agent_type.upper()}\n")
            f.write(f"测试样本数: {len(test_data)}\n\n")
            f.write(f"--- 平均性能指标 ---\n")
            f.write(f"平均每 Episode 奖励: {avg_reward:.4f}\n")
            f.write(f"平均每 Episode 最终洪水总量: {avg_flooding:.4f}\n")
            f.write(f"平均每 Episode 最终 CSO 总量: {avg_cso:.4f}\n")
            f.write(f"平均每 Episode 步数: {avg_steps:.2f}\n\n")
            f.write("--- 场景分布频率 (基于总步数) ---\n")
            for i in range(len(class_dist_freq)):
                f.write(f"  场景 {i}: {class_dist_freq[i]*100:.2f}%\n")
        print(f"评估摘要已保存到 {summary_file}")
    except Exception as e:
         print(f"错误: 无法写入评估摘要文件 {summary_file}: {e}")

    # 保存整体评估结果 每个 episode 的指标
    results_file = os.path.join(output_dir, "evaluation_results.npy")
    try:
        np.save(results_file, results)
        print(f"详细评估结果已保存到 {results_file}")
    except Exception as e:
        print(f"错误: 无法保存详细评估结果文件 {results_file}: {e}")

    try:
        plt.figure(figsize=(12, 10))

        # 绘制每个 Episode 的性能指标
        plt.subplot(2, 1, 1)
        bar_width = 0.25
        index = np.arange(len(test_data))
        plt.bar(index - bar_width, results['rewards'], bar_width, label=f'Reward (Avg: {avg_reward:.2f})')
        # 洪水和 CSO 量级差异大，用次坐标轴/分开绘图
        # plt.bar(index, results['flooding'], bar_width, label=f'Flooding (Avg: {avg_flooding:.2f})')
        # plt.bar(index + bar_width, results['cso'], bar_width, label=f'CSO (Avg: {avg_cso:.2f})')
        plt.ylabel('Reward')
        plt.twinx()
        plt.plot(index, results['flooding'], label=f'Flooding (Avg: {avg_flooding:.2f})', color='orange', marker='o')
        plt.plot(index, results['cso'], label=f'CSO (Avg: {avg_cso:.2f})', color='green', marker='s')
        plt.ylabel('Total Volume')
        plt.title('Performance Metrics per Test Episode')
        plt.xlabel('Test Sample Index')
        plt.xticks(index)
        plt.legend(loc='upper left')
        plt.grid(True, axis='x')

        # 绘制场景分布频率
        plt.subplot(2, 1, 2)
        plt.bar(range(args.num_classes), class_dist_freq * 100)
        plt.title('Overall Scene Distribution Frequency')
        plt.xlabel('Scene Class ID')
        plt.ylabel('Frequency (%)')
        plt.xticks(range(args.num_classes))
        plt.grid(True, axis='y')

        plt.tight_layout()
        summary_plot_file = os.path.join(output_dir, "evaluation_summary.png")
        plt.savefig(summary_plot_file)
        plt.close()
        print(f"评估摘要图表已保存到 {summary_plot_file}")

    except Exception as e:
        print(f"错误: 绘制评估摘要图表失败: {e}")
        plt.close()

    print(f"\n--- 评估完成 ---")
    print(f"平均奖励: {avg_reward:.4f}")
    print(f"平均洪水: {avg_flooding:.4f}")
    print(f"平均CSO: {avg_cso:.4f}")
