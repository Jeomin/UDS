#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练内生可解释DRL系统
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime
import argparse
import tensorflow as tf
from sklearn.cluster import KMeans
import agents.PPO as PPO
import agents.DQN as DQN
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interpretability.intrinsic_soft_tree import IntrinsicSoftTree
from interpretability.specialized_agents import SpecializedAgentManager
from interpretability.intrinsic_explainer import IntrinsicExplainer


def parse_args():
    """
    解析命令行参数
    
    Returns:
        args: 解析后的参数
    """
    parser = argparse.ArgumentParser(description='训练内生可解释DRL系统')
    
    # 模型参数
    parser.add_argument('--num-classes', type=int, default=5,
                       help='场景类别数量')
    
    parser.add_argument('--tree-depth', type=int, default=4,
                       help='决策树深度')
    
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='软决策树温度参数')
    
    parser.add_argument('--num-train', type=int, default=50,
                       help='训练样本数量')
    
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    
    parser.add_argument('--agent-type', type=str, default='ppo', 
                       choices=['ppo', 'dqn'],
                       help='agent类型')
    
    parser.add_argument('--joint-training', type=bool, default=False,
                       help='是否使用联合训练')
    
    # 路径设置
    parser.add_argument('--output-dir', type=str, default='output',
                       help='输出目录')
    
    parser.add_argument('--env-path', type=str, default='chaohu',
                       help='SWMM环境配置文件路径')
    
    parser.add_argument('--rain-path', type=str, default='training_raindata.npy',
                       help='训练降雨数据路径')
    
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


def setup_environment(env_path):
    """
    设置SWMM环境
    
    Args:
        env_path: 环境配置文件路径
        
    Returns:
        env: SWMM环境
    """
    import environment.SWMM_ENV as SWMM_ENV
    
    env_params = {
        'orf': env_path,
        'advance_seconds': 300
    }
    
    env = SWMM_ENV.SWMM_ENV(env_params)
    return env


def pretrain_decision_tree(env, rainfall_data, num_classes, output_dir):
    """
    预训练软决策树，将状态空间分类为不同场景
    
    Args:
        env: SWMM环境
        rainfall_data: 降雨数据
        num_classes: 场景类别数量
        output_dir: 输出目录
        
    Returns:
        soft_tree: 训练后的软决策树
        kmeans: KMeans聚类模型
    """
    # 收集状态和奖励数据
    states = []
    rewards = []
    
    for rain in rainfall_data[:10]:  # TODO 一点点数据？这里预训练写的太蠢了
        s = env.reset(rain)
        done = False
        
        while not done:
            # 随机动作或使用启发式规则 为什么要这样构建数据，直接拿降雨当前状态可以吗
            action = [np.random.randint(2) for _ in range(len(env.config['action_assets']))]
            s_next, reward, flooding, cso, done = env.step(action)
            
            # 记录状态和奖励
            states.append(s)
            rewards.append([reward, flooding, cso])  # 使用多维奖励
            
            s = s_next
    
    # 使用聚类算法划分场景，为决策树训练打标真值
    print("使用KMeans聚类划分场景...")
    
    # 根据奖励、状态或两者结合进行聚类
    X = np.hstack([np.array(states), np.array(rewards)])
    
    kmeans = KMeans(n_clusters=num_classes)
    labels = kmeans.fit_predict(X)
    
    # 分析每个类别的特征
    for i in range(num_classes):
        mask = (labels == i)
        if np.sum(mask) > 0:
            class_states = np.array(states)[mask]
            class_rewards = np.array(rewards)[mask]
            
            # 计算该类别的平均状态和奖励
            mean_state = np.mean(class_states, axis=0)
            mean_reward = np.mean(class_rewards, axis=0)
            
            print(f"场景 {i}: 样本数 {np.sum(mask)}")
            print(f"  平均奖励: {mean_reward[0]:.4f}")
            print(f"  平均洪水: {mean_reward[1]:.4f}")
            print(f"  平均CSO: {mean_reward[2]:.4f}")
            
            # 找出该类别的特征值范围
            for j in range(min(5, len(mean_state))):  # 只打印前5个特征
                feature_min = np.min(class_states[:, j])
                feature_max = np.max(class_states[:, j])
                feature_mean = mean_state[j]
                
                print(f"  特征 {j}: 均值={feature_mean:.4f}, 范围=[{feature_min:.4f}, {feature_max:.4f}]")
    
    # 训练软决策树预测场景
    print("训练软决策树预测场景...")
    
    # 将标签转换为独热编码
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    
    # 创建软决策树
    soft_tree = IntrinsicSoftTree(
        input_dim=len(env.config['states']),
        num_classes=num_classes,
        depth=4,
        temperature=1.0
    )
    
    # 训练模型
    soft_tree.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = soft_tree.model.fit(
        np.array(states), one_hot_labels,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # 保存聚类结果和训练历史
    np.save(os.path.join(output_dir, 'cluster_labels.npy'), labels)
    np.save(os.path.join(output_dir, 'cluster_centers.npy'), kmeans.cluster_centers_)
    
    # 保存模型
    soft_tree.save(os.path.join(output_dir, 'pretrained_tree.pkl'))
    
    # 可视化训练过程
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss During Pretraining')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy During Pretraining')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tree_pretraining.png'))
    
    return soft_tree, kmeans


def initialize_specialized_agents(env, num_classes, agent_type='ppo'):
    """
    为每个场景初始化专门化agent
    
    Args:
        env: SWMM环境
        num_classes: 场景类别数量
        agent_type: agent类型
        
    Returns:
        agent_manager: 专门化agent管理器
    """
    print(f"初始化{num_classes}个专门化{agent_type.upper()}agent...")
    
    # 根据agent类型选择正确的模块
    if agent_type.lower() == 'ppo':
        import agents.PPO as PPO
        agent_class = PPO
        
        # 基础参数
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
            'gamma': 0.3,
            'epsilon': 0.1,
            'ep_min': 0.01,
            'ep_decay': 0.1
        }
    else:  # DQN
        import agents.DQN as DQN
        agent_class = DQN
        
        # 基础参数
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
            'gamma': 0.3,
            'epsilon': 0.1,
            'ep_min': 0.01,
            'ep_decay': 0.1
        }
    
    # 创建agent管理器
    agent_manager = SpecializedAgentManager(
        env=env,
        num_classes=num_classes,
        agent_class=agent_class,
        base_params=agent_params
    )
    
    return agent_manager


def simulate_episode(env, rain, soft_tree, agent_manager, epoch, exploration_phase=True):
    """
    模拟一个场景并收集数据 (适配修正后的 PPO)

    Args:
        env: SWMM环境
        rain: 降雨数据
        soft_tree: 软决策树 (MLP)
        agent_manager: 专门化agent管理器
        epoch: 当前训练轮数
        exploration_phase: 是否处于探索阶段 (传递给 agent.choose_action)

    Returns:
        episode_data: 模拟数据列表, 格式为:
                      [(state, embedding, action_info, reward, class_id, next_state, done), ...]
                      action_info (dict):
                          - PPO: {'action': action_np, 'logp_old': logp_action_np}
                          - DQN: {'action_index': action_index}
    """
    s = env.reset(rain)
    done = False
    episode_data = []
    num_actions_expected = len(env.config["action_assets"])

    while not done:
        embedding = soft_tree.generate_embedding(s)

        # 选择动作
        # PPO 返回 (action_np, logp_action_np)
        # DQN 返回 action_index
        result, class_id = agent_manager.choose_action(s, embedding, # Pass combined state to agent
                                                       train_mode=exploration_phase)

        action_for_env = None
        action_info_to_store = None
        agent = agent_manager.agents[class_id] # Get the specific agent instance

        # --- PPO 处理 ---
        # 检查 agent 类型，并且 result 是否是 PPO 的期望输出 (action, logp)
        if isinstance(agent, PPO):
            logp_action_np, action_np = result # PPO 返回 action (0/1 array) 和 logp
            action_for_env = action_np.tolist()
            action_info_to_store = {'action': action_np, 'logp_old': logp_action_np}

            # 检查动作维度
            if len(action_for_env) != num_actions_expected:
                print(f"警告: PPO 动作维度 {len(action_for_env)} 与预期 {num_actions_expected} 不匹配!")
                # 可以选择填充、截断或使用默认动作
                action_for_env = [0] * num_actions_expected
                # 相应地调整 action_info (如果需要)
                action_info_to_store['action'] = np.array(action_for_env)
                # logp_old 此时可能不准确，但暂时保留

        # --- DQN 处理 ---
        # 检查 agent 类型，并且 result 是否是 DQN 的期望输出 (action_index)
        elif isinstance(agent, DQN):
            # DQN 返回动作索引 (整数)
            action_index = result
            action_info_to_store = {'action_index': action_index}

            # 尝试从 agent 获取 action_table
            if hasattr(agent, 'action_table') and agent.action_table is not None:
                action_table = agent.action_table
                if action_index < len(action_table):
                    action_for_env = action_table[action_index, :].tolist()
                else:
                    print(f"警告: DQN 动作索引 {action_index} 超出 action_table 范围 {len(action_table)}!")
                    action_for_env = [int(bit) for bit in format(action_index, f'0{num_actions_expected}b')] # Fallback
            else:
                 # Fallback: 如果 agent 没有 action_table
                 print(f"警告: 未找到 Agent {class_id} (DQN?) 的 action_table，使用二进制转换。")
                 action_for_env = [int(bit) for bit in format(action_index, f'0{num_actions_expected}b')]

            # 确保 action_for_env 是列表
            if not isinstance(action_for_env, list):
                 action_for_env = [0] * num_actions_expected # Fallback

            # 检查动作维度
            if len(action_for_env) != num_actions_expected:
                 print(f"警告: DQN 动作维度 {len(action_for_env)} 与预期 {num_actions_expected} 不匹配!")
                 action_for_env = [0] * num_actions_expected

        else:
             # 如果 result 格式未知
             print(f"警告: Agent {class_id} 返回了未知格式的动作结果: {result}")
             action_for_env = [0] * num_actions_expected # 使用默认动作
             # action_info 需要一个默认值，或者标记为无效
             action_info_to_store = {'action': np.array(action_for_env), 'logp_old': -np.inf} # PPO 风格的默认值


        # 执行环境步骤
        if action_for_env is not None:
            s_next, reward, flooding, cso, done = env.step(action_for_env)
        else:
            # 如果 action_for_env 仍然是 None (理论上不应发生，因为有 fallback)
             raise ValueError("错误: 未能确定环境动作 action_for_env")

        # (state, embedding, action_info, reward, class_id, next_state, done)
        # action_info 是包含动作和可能的 logp 的字典
        episode_data.append((s, embedding, action_info_to_store, reward, class_id, s_next, done))

        s = s_next # 更新状态 (原始状态)

    return episode_data


def train_decision_tree_on_data(soft_tree, data):
    """
    使用收集的数据更新决策树
    
    Args:
        soft_tree: 软决策树
        data: 收集的数据
        
    Returns:
        loss: 训练损失
    """
    states = []
    class_ids = []
    
    for item in data:
        state, _, _, _, class_id, _, _ = item
        states.append(state)
        class_ids.append(class_id)
    
    states = np.array(states)
    one_hot_labels = tf.keras.utils.to_categorical(class_ids, num_classes=soft_tree.num_classes)
    
    history = soft_tree.model.fit(
        states, one_hot_labels,
        epochs=20,
        batch_size=32,
        verbose=1
    )
    
    return history.history['loss'][-1]


def evaluate_system(env, rainfall_data, soft_tree, agent_manager, output_dir, epoch):
    """
    评估整个系统的性能
    
    Args:
        env: SWMM环境
        rainfall_data: 降雨数据
        soft_tree: 软决策树
        agent_manager: 专门化agent管理器
        output_dir: 输出目录
        epoch: 当前训练轮数
        
    Returns:
        avg_reward: 平均奖励
        avg_flooding: 平均洪水量
        avg_cso: 平均CSO排放量
    """
    print(f"Epoch {epoch}: 评估系统性能...")
    
    total_reward = 0
    total_flooding = 0
    total_cso = 0
    total_steps = 0
    class_counts = {i: 0 for i in range(soft_tree.num_classes)}

    test_history = {
        'states': [],
        'embeddings': [],
        'class_ids': [],
        'actions': [],
        'rewards': [],
        'floodings': [],
        'csos': []
    }

    explainer = IntrinsicExplainer(
        soft_tree=soft_tree, 
        env_config=env.config,
        state_names=[f"特征_{i}" for i in range(len(env.config['states']))]
    )

    for i, rain in enumerate(rainfall_data):
        print(f"  评估降雨样本 {i+1}/{len(rainfall_data)}...")
        
        s = env.reset(rain)
        done = False
        episode_reward = 0
        episode_flooding = 0
        episode_cso = 0
        episode_steps = 0

        explanations = []
        
        while not done:
            embedding = soft_tree.generate_embedding(s)
            class_id = np.argmax(embedding)
            class_counts[class_id] += 1
            result, _ = agent_manager.choose_action(s, embedding, train_mode=False)
            
            if isinstance(result, tuple) and len(result) == 2:
                # PPO风格：返回(logits, action)
                logits, action = result
                if hasattr(action, 'shape') and len(action.shape) > 1:
                    action_for_env = action[0].tolist()
                else:
                    action_for_env = action.tolist() if hasattr(action, 'tolist') else action

                action_for_history = action
            else:
                # DQN风格：直接返回动作
                action = result
                if isinstance(action, (int, np.integer)):
                    if hasattr(agent_manager, 'action_table'):
                        action_for_env = agent_manager.action_table[action, :].tolist()
                    else:
                        # 如果没有action_table，使用默认二进制转换
                        action_for_env = [int(bit) for bit in format(action, f'0{len(env.config["action_assets"])}b')]
                else:
                    if hasattr(action, 'shape') and len(action.shape) > 1:
                        action_for_env = action[0].tolist()
                    else:
                        action_for_env = action.tolist() if hasattr(action, 'tolist') else action

                action_for_history = action

            s_next, reward, flooding, cso, done = env.step(action_for_env)

            explanation = explainer.generate_explanation(s, embedding, action_for_env, class_id)
            explanations.append(explanation)
            
            test_history['states'].append(s)
            test_history['embeddings'].append(embedding)
            test_history['class_ids'].append(class_id)
            test_history['actions'].append(action_for_history)
            test_history['rewards'].append(reward)
            test_history['floodings'].append(flooding)
            test_history['csos'].append(cso)
            
            episode_reward += reward
            episode_flooding += flooding
            episode_cso += cso
            episode_steps += 1

            s = s_next

        total_reward += episode_reward
        total_flooding += episode_flooding
        total_cso += episode_cso
        total_steps += episode_steps

        explanation_file = os.path.join(output_dir, f"explanations_epoch_{epoch}_sample_{i}.txt")
        with open(explanation_file, 'w', encoding='utf-8') as f:
            for j, exp in enumerate(explanations):
                f.write(f"Step {j}:\n")
                f.write(f"  场景: {exp['scene']}\n")
                f.write(f"  决策路径: {exp['decision_path']}\n")
                f.write(f"  理由: {exp['reason']}\n")
                f.write(f"  预期后果: {exp['consequence']}\n")
                f.write(f"  备选方案: {exp['alternatives']}\n\n")

    avg_reward = total_reward / max(1, len(rainfall_data))
    avg_flooding = total_flooding / max(1, len(rainfall_data))
    avg_cso = total_cso / max(1, len(rainfall_data))
    avg_steps = total_steps / max(1, len(rainfall_data))

    eval_summary_file = os.path.join(output_dir, f"evaluation_epoch_{epoch}.txt")
    with open(eval_summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Epoch: {epoch}\n")
        f.write(f"平均奖励: {avg_reward:.4f}\n")
        f.write(f"平均洪水: {avg_flooding:.4f}\n")
        f.write(f"平均CSO: {avg_cso:.4f}\n")
        f.write(f"平均步数: {avg_steps:.2f}\n\n")
        f.write("场景分布:\n")
        for class_id, count in class_counts.items():
            f.write(f"  场景 {class_id}: {count} 步 ({count/max(1, total_steps)*100:.2f}%)\n")
    
    np.save(os.path.join(output_dir, f"test_history_epoch_{epoch}.npy"), test_history)

    try:
        plt.figure(figsize=(15, 12))

        # 奖励图
        plt.subplot(4, 1, 1)
        plt.plot(test_history['rewards'])
        plt.title(f'Epoch {epoch} - Rewards per Step (Avg Reward per Episode: {avg_reward:.4f})')
        plt.ylabel('Reward')
        plt.grid(True)

        # 洪水与CSO图
        plt.subplot(4, 1, 2)
        plt.plot(test_history['floodings'], label=f'Cumulative Flooding (Total Avg: {avg_flooding:.2f})')
        plt.plot(test_history['csos'], label=f'Cumulative CSO (Total Avg: {avg_cso:.2f})')
        plt.title('Cumulative Flooding and CSO per Step')
        plt.ylabel('Volume')
        plt.legend()
        plt.grid(True)

        # 场景分类图
        plt.subplot(4, 1, 3)
        plt.plot(test_history['class_ids'], marker='.', linestyle='None', markersize=4)
        plt.title('Scene Classification per Step')
        plt.yticks(range(soft_tree.num_classes))
        plt.ylabel('Scene Class ID')
        plt.grid(True)

        # PPO Log Prob 图 (如果存在)
        valid_logps = [lp for lp in test_history['logps'] if lp is not None and lp != -np.inf]
        if valid_logps:
            plt.subplot(4, 1, 4)
            plt.plot(valid_logps)
            plt.title('PPO Action Log Probability per Step')
            plt.ylabel('Log Probability')
            plt.grid(True)

        plt.xlabel('Time Step')
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"evaluation_epoch_{epoch}.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"评估图表已保存到 {plot_file}")

    except Exception as e:
        print(f"错误: 绘制评估图表失败: {e}")
        plt.close()
    
    return avg_reward, avg_flooding, avg_cso


def train_intrinsic_interpretable(args):
    """
    训练内生可解释系统的主函数
    
    Args:
        args: 命令行参数
    """
    # 创建输出目录
    output_dir = os.path.join(
        args.output_dir,
        f"intrinsic_interpretable_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置环境
    env = setup_environment(args.env_path)
    
    # 加载降雨数据
    rainfall_data = load_rainfall_data(args.rain_path)
    if rainfall_data is None:
        print("无法加载降雨数据，退出")
        return
    
    # 1. 预训练软决策树
    print("预训练软决策树...")
    soft_tree, kmeans = pretrain_decision_tree(
        env, rainfall_data, 
        num_classes=args.num_classes,
        output_dir=output_dir
    )
    
    # 2. 初始化专门化agent
    print("初始化专门化agent...")
    agent_manager = initialize_specialized_agents(
        env, 
        num_classes=args.num_classes,
        agent_type=args.agent_type
    )
    
    # 3. 主训练循环
    total_epochs = args.epochs
    decision_tree_train_epochs = total_epochs // 3  # 决策树训练轮数
    
    # 训练历史
    history = {
        'tree_loss': [],
        'agent_loss': [],
        'rewards': [],
        'flooding': [],
        'cso': []
    }
    
    # 第一阶段：只训练决策树
    print("=== 阶段1: 只训练决策树 ===")
    for epoch in range(decision_tree_train_epochs):
        print(f"Epoch {epoch+1}/{total_epochs}")
        
        # 收集数据
        all_data = []
        
        for i, rain in enumerate(rainfall_data[:args.num_train]):
            print(f"  处理降雨样本 {i+1}/{min(args.num_train, len(rainfall_data))}...")
            data = simulate_episode(env, rain, soft_tree, agent_manager, epoch)
            all_data.extend(data)
        
        # 训练决策树
        tree_loss = train_decision_tree_on_data(soft_tree, all_data)
        history['tree_loss'].append(tree_loss)
        
        # 评估
        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_reward, avg_flooding, avg_cso = evaluate_system(
                env, rainfall_data[args.num_train:args.num_train+2], 
                soft_tree, agent_manager, output_dir, epoch
            )
            history['rewards'].append(avg_reward)
            history['flooding'].append(avg_flooding)
            history['cso'].append(avg_cso)
    
    # 第二阶段：只训练专门化agent
    print("=== 阶段2: 只训练专门化agent ===")
    for epoch in range(decision_tree_train_epochs, 2 * decision_tree_train_epochs):
        print(f"Epoch {epoch+1}/{total_epochs}")
        
        # 收集数据
        all_data = []
        class_data = {i:[] for i in range(args.num_classes)}
        
        for i, rain in enumerate(rainfall_data[:args.num_train]):
            print(f"  处理降雨样本 {i+1}/{min(args.num_train, len(rainfall_data))}...")
            data = simulate_episode(env, rain, soft_tree, agent_manager, epoch)
            all_data.extend(data)
            
            # 按场景类别整理数据
            for item in data:
                _, _, _, _, class_id, _, _ = item
                if class_id in class_data:
                    class_data[class_id].append(item)
        
        # 训练专门化agent
        losses = agent_manager.train_agents(class_data)
        avg_loss = np.mean(list(losses.values())) if losses else 0
        history['agent_loss'].append(avg_loss)
        
        # 评估
        if (epoch + 1) % 5 == 0:
            avg_reward, avg_flooding, avg_cso = evaluate_system(
                env, rainfall_data[args.num_train:args.num_train+2], 
                soft_tree, agent_manager, output_dir, epoch
            )
            history['rewards'].append(avg_reward)
            history['flooding'].append(avg_flooding)
            history['cso'].append(avg_cso)
    
    # 第三阶段：交替训练或联合训练
    print("=== 阶段3: 交替训练/联合训练 ===")
    for epoch in range(2 * decision_tree_train_epochs, total_epochs):
        print(f"Epoch {epoch+1}/{total_epochs}")
        
        # 收集数据
        all_data = []
        class_data = {i:[] for i in range(args.num_classes)}
        
        for i, rain in enumerate(rainfall_data[:args.num_train]):
            print(f"  处理降雨样本 {i+1}/{min(args.num_train, len(rainfall_data))}...")
            data = simulate_episode(env, rain, soft_tree, agent_manager, epoch)
            all_data.extend(data)
            
            # 按场景类别整理数据
            for item in data:
                _, _, _, _, class_id, _, _ = item
                if class_id in class_data:
                    class_data[class_id].append(item)
        
        if args.joint_training:
            # 联合训练
            print("  联合训练决策树和agent...")
            tree_loss = train_decision_tree_on_data(soft_tree, all_data)
            losses = agent_manager.train_agents(class_data)
            avg_loss = np.mean(list(losses.values())) if losses else 0
            
            history['tree_loss'].append(tree_loss)
            history['agent_loss'].append(avg_loss)
        else:
            # 交替训练
            if epoch % 2 == 0:
                print("  训练决策树...")
                tree_loss = train_decision_tree_on_data(soft_tree, all_data)
                history['tree_loss'].append(tree_loss)
            else:
                print("  训练专门化agent...")
                losses = agent_manager.train_agents(class_data)
                avg_loss = np.mean(list(losses.values())) if losses else 0
                history['agent_loss'].append(avg_loss)
        
        # 评估
        if (epoch + 1) % 5 == 0 or epoch == total_epochs - 1:
            avg_reward, avg_flooding, avg_cso = evaluate_system(
                env, rainfall_data[args.num_train:args.num_train+2], 
                soft_tree, agent_manager, output_dir, epoch
            )
            history['rewards'].append(avg_reward)
            history['flooding'].append(avg_flooding)
            history['cso'].append(avg_cso)
    
    # 保存最终模型
    print("保存最终模型...")
    soft_tree.save(os.path.join(output_dir, "final_soft_tree.pkl"))
    agent_manager.save_models(os.path.join(output_dir, "agents"))
    
    # 保存训练历史
    np.save(os.path.join(output_dir, "training_history.npy"), history)
    
    # 绘制训练历史
    plt.figure(figsize=(15, 10))
    
    # 绘制决策树损失
    plt.subplot(3, 1, 1)
    plt.plot(history['tree_loss'], label='Tree Loss')
    plt.title('Decision Tree Training Loss')
    plt.grid(True)
    plt.legend()
    
    # 绘制agent损失
    plt.subplot(3, 1, 2)
    plt.plot(history['agent_loss'], label='Agent Loss')
    plt.title('Agent Training Loss')
    plt.grid(True)
    plt.legend()
    
    # 绘制评估指标
    plt.subplot(3, 1, 3)
    plt.plot(history['rewards'], label='Reward')
    plt.plot(history['flooding'], label='Flooding')
    plt.plot(history['cso'], label='CSO')
    plt.title('Evaluation Metrics')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close()
    
    print(f"训练完成，输出保存在 {output_dir}")
    
    return soft_tree, agent_manager


if __name__ == "__main__":
    args = parse_args()
    train_intrinsic_interpretable(args)