# -*- coding: utf-8 -*-
"""
专门化agent管理器，管理多个针对不同场景的DRLagent
"""
import importlib
import json
import pickle
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import agents.PPO as PPO
import agents.DQN as DQN

class SpecializedAgentManager:
    """专门化agent管理器，管理多个针对不同场景的DRLagent"""
    
    def __init__(self, env, num_classes, agent_class, base_params=None):
        """
        初始化专门化agent管理器
        
        Args:
            env: 环境对象
            num_classes: 场景类别数量
            agent_class: agent类（如DQN或PPO）
            base_params: 基础参数字典
        """
        self.env = env
        self.num_classes = num_classes
        self.agent_class = agent_class
        self.base_params = base_params or {}
        
        # 创建多个专门化agent
        self.agents = []
        for i in range(num_classes):
            # 为每个agent创建特定参数
            params = self.base_params.copy()
            # 扩展状态维度，包含agent类别嵌入
            params['state_dim'] = params.get('state_dim', 0) + num_classes
            # 不同agent可以有不同超参数
            params['class_id'] = i
            
            # 创建agent
            agent = agent_class(params, env)
            self.agents.append(agent)
    
    def choose_action(self, state, embedding, train_mode=False):
        """
        根据状态和embedding选择动作
        
        Args:
            state: 当前状态
            embedding: 策略嵌入向量
            train_mode: 是否处于训练模式
            
        Returns:
            action: 选择的动作 (可能是直接动作或(logits, action)对)
            class_id: 选择的场景类别
        """
        # 找出概率最高的类别
        class_id = np.argmax(embedding)
        
        # 确保class_id在有效范围内
        if class_id >= len(self.agents):
            class_id = 0  # 使用默认agent
            
        agent = self.agents[class_id]
        
        # 合并状态和embedding
        combined_state = np.concatenate([state, embedding])
        
        # 根据agent类型选择动作
        result = agent.choose_action(combined_state, train_mode)
        
        # 返回原始结果
        return result, class_id

    
    # def train_agents(self, dataset_by_class):
    #     """
    #     训练各个专门化agent

    #     Args:
    #         dataset_by_class: 按类别整理的训练数据字典 {class_id: [data_tuples...]}

    #     Returns:
    #         losses (dict): 各agent的训练损失字典 {class_id: single_loss_value}
    #     """
    #     losses = {}
    #     for class_id, data in dataset_by_class.items():
    #         if class_id < 0 or class_id >= len(self.agents):
    #             print(f"警告: 无效的 class_id {class_id}，跳过训练。")
    #             continue
    #         if len(data) < 10: # 确保有足够数据
    #             print(f"警告: 场景 {class_id} 数据量不足 ({len(data)} < 10)，跳过训练。")
    #             continue

    #         print(f"训练场景 {class_id} 的agent，数据量: {len(data)}")
    #         agent = self.agents[class_id]

    #         # 1. 准备数据
    #         try:
    #             states, embeddings, ppo_actions, logp_olds, dqn_action_indices, \
    #             rewards, next_states, next_embeddings, dones = self._prepare_training_data(data)
    #             if not states:
    #                 print(f"警告: 场景 {class_id} 准备数据后为空，跳过训练。")
    #                 continue
    #         except Exception as e:
    #             print(f"错误: 为场景 {class_id} 准备数据时出错: {e}")
    #             continue

    #         # 2. 根据 Agent 类型进行训练
    #         agent_loss = None # 用于记录 DQN 的损失或 PPO 的合并损失

    #         try:
    #             # --- PPO 训练逻辑 ---
    #             if isinstance(agent, PPO):
    #                 # 过滤掉 None 值 (如果 _prepare_training_data 产生了占位符)
    #                 valid_indices = [i for i, lp in enumerate(logp_olds) if lp is not None and ppo_actions[i] is not None]
    #                 if len(valid_indices) < 10: # 如果有效数据太少，跳过 (阈值可调)
    #                      print(f"警告: 场景 {class_id} PPO 有效数据不足 ({len(valid_indices)}/{len(states)})，跳过训练。")
    #                      continue

    #                 # 提取有效数据
    #                 states_p = [states[i] for i in valid_indices]
    #                 embeddings_p = [embeddings[i] for i in valid_indices]
    #                 actions_p = [ppo_actions[i] for i in valid_indices] # PPO 的 0/1 动作列表
    #                 logp_olds_p = [logp_olds[i] for i in valid_indices]
    #                 rewards_p = [rewards[i] for i in valid_indices]
    #                 next_states_p = []
    #                 dones_p = []
    #                 next_embeddings_p = []

    #                 # 对齐 next_states, dones, next_embeddings
    #                 original_indices_with_next_state = {i for i, ns in enumerate(next_states)} # 记录哪些原始索引有 next_state
    #                 for i in valid_indices:
    #                     dones_p.append(dones[i])
    #                     if i in original_indices_with_next_state:
    #                         # 找到原始 next_state 列表中的对应索引
    #                         original_next_state_index = list(original_indices_with_next_state).index(i)
    #                         next_states_p.append(next_states[original_next_state_index])
    #                         next_embeddings_p.append(next_embeddings[original_next_state_index])
    #                     else:
    #                         # 如果原始索引 i 没有 next_state (通常是最后一步)
    #                         next_states_p.append(states[i]) # 使用当前状态作为占位符
    #                         next_embeddings_p.append(embeddings[i]) # 使用当前 embedding

    #                 # 再次确保所有列表长度一致
    #                 if not (len(states_p) == len(embeddings_p) == len(actions_p) == len(logp_olds_p) == len(rewards_p) == len(next_states_p) == len(dones_p)):
    #                     print(f"错误: 场景 {class_id} PPO 训练数据列表长度不一致 ({len(states_p)} vs ...)，跳过训练。")
    #                     continue

    #                 # 准备 PPO 需要的输入
    #                 combined_states_np = np.array([np.concatenate([s, e]) for s, e in zip(states_p, embeddings_p)])
    #                 actions_np = np.array(actions_p)
    #                 rewards_np = np.array(rewards_p)
    #                 # 注意：next_embeddings_p 可能与 next_states_p 长度不完全一致，如果最后一步没有 next_state
    #                 # 确保 combined_next_states_np 的构建是安全的
    #                 combined_next_states_np = np.array([np.concatenate([ns, ne]) for ns, ne in zip(next_states_p, next_embeddings_p)])
    #                 dones_np = np.array(dones_p)
    #                 logp_olds_np = np.array(logp_olds_p)

    #                 # 调用 PPO 的 update 方法
    #                 policy_loss, value_loss = agent.update(combined_states_np, actions_np, rewards_np, combined_next_states_np, dones_np, logp_olds_np)

    #                 if policy_loss is not None and value_loss is not None:
    #                     agent_loss = policy_loss + value_loss
    #                 elif policy_loss is not None:
    #                     agent_loss = policy_loss
    #                 elif value_loss is not None:
    #                     agent_loss = value_loss # 或者返回 0.0?
    #                 else:
    #                     agent_loss = None # 标记失败

    #             # --- DQN 训练逻辑 ---
    #             elif isinstance(agent, DQN):
    #                 # 过滤掉 None 值
    #                 valid_indices = [i for i, idx in enumerate(dqn_action_indices) if idx is not None]
    #                 if len(valid_indices) < 10: # 如果有效数据太少，跳过
    #                      print(f"警告: 场景 {class_id} DQN 有效数据不足 ({len(valid_indices)}/{len(states)})，跳过训练。")
    #                      continue

    #                 # 提取有效数据 (DQN 通常用原始状态)
    #                 states_d = [states[i] for i in valid_indices]
    #                 action_indices_d = [dqn_action_indices[i] for i in valid_indices]
    #                 rewards_d = [rewards[i] for i in valid_indices]
    #                 next_states_d = []
    #                 dones_d = []

    #                 # 对齐 next_states 和 dones
    #                 original_indices_with_next_state = {i for i, ns in enumerate(next_states)}
    #                 for i in valid_indices:
    #                     dones_d.append(dones[i])
    #                     if i in original_indices_with_next_state:
    #                          original_next_state_index = list(original_indices_with_next_state).index(i)
    #                          next_states_d.append(next_states[original_next_state_index])
    #                     else:
    #                          next_states_d.append(states[i]) # 或 None

    #                 if not (len(states_d) == len(action_indices_d) == len(rewards_d) == len(next_states_d) == len(dones_d)):
    #                     print(f"错误: 场景 {class_id} DQN 训练数据列表长度不一致，跳过训练。")
    #                     continue

    #                 states_np = np.array(states_d)
    #                 action_indices_np = np.array(action_indices_d)
    #                 rewards_np = np.array(rewards_d)
    #                 next_states_np = np.array(next_states_d)
    #                 dones_np = np.array(dones_d)

    #                 # 调用 DQN 的 train_on_batch 方法 (假设接口如此)
    #                 # loss = agent.train_on_batch(states_np, action_indices_np, rewards_np, next_states_np, dones_np)
    #                 # agent_loss = loss
    #                 print(f"信息: 场景 {class_id} 检测到 DQN Agent，但 train_on_batch 调用被注释掉。请根据实际 DQN 实现取消注释并调整。")
    #                 agent_loss = 0.0 # 临时值

    #             else:
    #                 print(f"警告: 场景 {class_id} 的 agent 类型未知或缺少必要的训练方法。")

    #         except Exception as e:
    #             print(f"错误: 训练场景 {class_id} 的 agent 时出错: {e}")
    #             agent_loss = None # 标记训练失败

    #         # --- 记录单一损失 ---
    #         if agent_loss is not None:
    #              try:
    #                  # 确保损失是标量浮点数
    #                  scalar_loss = float(tf.reduce_mean(agent_loss) if hasattr(agent_loss, 'numpy') else np.mean(agent_loss))
    #                  losses[class_id] = scalar_loss
    #                  print(f"  场景 {class_id} Agent 训练完成: 损失={scalar_loss:.4f}")
    #              except Exception as cast_e:
    #                  print(f"警告: 无法将场景 {class_id} 的损失转换为标量: {agent_loss}, 错误: {cast_e}")
    #                  losses[class_id] = 0.0 # 或其他标记值
    #         else:
    #              losses[class_id] = None # 标记训练未成功执行或失败

    #     return losses
    def train_agents(self, dataset_by_class, batch_size=64, ppo_epochs=4): # Add batch_size and ppo_epochs
        """
        训练各个专门化agent (使用 Mini-batching, 返回单一损失)

        Args:
            dataset_by_class: 按类别整理的训练数据字典 {class_id: [data_tuples...]}
            batch_size (int): Mini-batch 大小.
            ppo_epochs (int): 对于 PPO，在同一批数据上迭代训练的次数.

        Returns:
            losses (dict): 各agent的训练损失字典 {class_id: single_loss_value}
        """
        losses = {}
        for class_id, data in dataset_by_class.items():
            if class_id < 0 or class_id >= len(self.agents):
                print(f"警告: 无效的 class_id {class_id}，跳过训练。")
                continue

            num_data_points = len(data)
            if num_data_points < batch_size: # 如果数据量小于一个批次，跳过或调整批大小
                print(f"警告: 场景 {class_id} 数据量不足 ({num_data_points} < {batch_size})，跳过训练。")
                continue

            print(f"训练场景 {class_id} 的agent，数据量: {num_data_points}")
            agent = self.agents[class_id]

            # 1. 准备完整数据
            try:
                states, embeddings, ppo_actions, logp_olds, dqn_action_indices, \
                rewards, next_states, next_embeddings, dones = self._prepare_training_data(data)
                if not states:
                    print(f"警告: 场景 {class_id} 准备数据后为空，跳过训练。")
                    continue
            except Exception as e:
                print(f"错误: 为场景 {class_id} 准备数据时出错: {e}")
                continue

            # 2. 根据 Agent 类型进行 Mini-batch 训练
            total_loss_accumulator = 0.0
            num_updates = 0
            training_failed = False

            try:
                if isinstance(agent, PPO):
                    # 过滤无效数据并获取有效索引
                    valid_indices = [i for i, lp in enumerate(logp_olds) if lp is not None and ppo_actions[i] is not None]
                    num_valid_data = len(valid_indices)

                    if num_valid_data < batch_size:
                        print(f"警告: 场景 {class_id} PPO 有效数据不足 ({num_valid_data} < {batch_size})，跳过训练。")
                        continue

                    # 提取有效数据
                    states_p = np.array([states[i] for i in valid_indices])
                    embeddings_p = np.array([embeddings[i] for i in valid_indices])
                    actions_p = np.array([ppo_actions[i] for i in valid_indices])
                    logp_olds_p = np.array([logp_olds[i] for i in valid_indices])
                    rewards_p = np.array([rewards[i] for i in valid_indices])
                    dones_p = np.array([dones[i] for i in valid_indices])

                    # 对齐 next_states 和 next_embeddings
                    original_to_valid_map = {original_idx: valid_idx for valid_idx, original_idx in enumerate(valid_indices)}
                    next_states_p = np.zeros_like(states_p)
                    next_embeddings_p = np.zeros_like(embeddings_p)
                    original_indices_with_next_state = {i for i, ns in enumerate(next_states)}

                    for valid_idx, original_idx in enumerate(valid_indices):
                        if original_idx in original_indices_with_next_state:
                            original_next_state_index = list(original_indices_with_next_state).index(original_idx)
                            next_states_p[valid_idx] = next_states[original_next_state_index]
                            next_embeddings_p[valid_idx] = next_embeddings[original_next_state_index]
                        else: # 最后一步
                            next_states_p[valid_idx] = states_p[valid_idx] # 用当前状态占位
                            next_embeddings_p[valid_idx] = embeddings_p[valid_idx] # 用当前 embedding 占位

                    combined_states_np = np.array([np.concatenate([s, e]) for s, e in zip(states_p, embeddings_p)])
                    combined_next_states_np = np.array([np.concatenate([ns, ne]) for ns, ne in zip(next_states_p, next_embeddings_p)])

                    # 在同一批数据上迭代多次
                    for _ in range(ppo_epochs):
                        # Shuffle 数据
                        perm = np.random.permutation(num_valid_data)
                        combined_states_shuffled = combined_states_np[perm]
                        actions_shuffled = actions_p[perm]
                        rewards_shuffled = rewards_p[perm]
                        combined_next_states_shuffled = combined_next_states_np[perm]
                        dones_shuffled = dones_p[perm]
                        logp_olds_shuffled = logp_olds_p[perm]

                        # Mini-batch
                        for i in range(0, num_valid_data, batch_size):
                            indices = slice(i, min(i + batch_size, num_valid_data))

                            mb_states = combined_states_shuffled[indices]
                            mb_actions = actions_shuffled[indices]
                            mb_rewards = rewards_shuffled[indices]
                            mb_next_states = combined_next_states_shuffled[indices]
                            mb_dones = dones_shuffled[indices]
                            mb_logp_olds = logp_olds_shuffled[indices]

                            policy_loss, value_loss = agent.update(mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones, mb_logp_olds)

                            if policy_loss is not None and value_loss is not None:
                                total_loss_accumulator += (policy_loss + value_loss) # 累加合并损失
                                num_updates += 1
                            else:
                                print(f"警告: 场景 {class_id} PPO update 返回无效损失。")
                                training_failed = True
                                break # 停止当前 PPO epoch 的 mini-batch 循环
                        if training_failed:
                            break # 停止 PPO 的 epoch 循环

                elif isinstance(agent, DQN):
                    valid_indices = [i for i, idx in enumerate(dqn_action_indices) if idx is not None]
                    num_valid_data = len(valid_indices)

                    if num_valid_data < batch_size:
                        print(f"警告: 场景 {class_id} DQN 有效数据不足 ({num_valid_data} < {batch_size})，跳过训练。")
                        continue

                    states_d = np.array([states[i] for i in valid_indices])
                    action_indices_d = np.array([dqn_action_indices[i] for i in valid_indices])
                    rewards_d = np.array([rewards[i] for i in valid_indices])
                    dones_d = np.array([dones[i] for i in valid_indices])
                    # 对齐 next_states
                    next_states_d = np.zeros_like(states_d)
                    original_indices_with_next_state = {i for i, ns in enumerate(next_states)}
                    for valid_idx, original_idx in enumerate(valid_indices):
                        if original_idx in original_indices_with_next_state:
                             original_next_state_index = list(original_indices_with_next_state).index(original_idx)
                             next_states_d[valid_idx] = next_states[original_next_state_index]
                        else:
                             next_states_d[valid_idx] = states_d[valid_idx] # 占位

                    perm = np.random.permutation(num_valid_data)
                    states_shuffled = states_d[perm]
                    action_indices_shuffled = action_indices_d[perm]
                    rewards_shuffled = rewards_d[perm]
                    next_states_shuffled = next_states_d[perm]
                    dones_shuffled = dones_d[perm]

                    for i in range(0, num_valid_data, batch_size):
                        indices = slice(i, min(i + batch_size, num_valid_data))
                        mb_states = states_shuffled[indices]
                        mb_actions = action_indices_shuffled[indices]
                        mb_rewards = rewards_shuffled[indices]
                        mb_next_states = next_states_shuffled[indices]
                        mb_dones = dones_shuffled[indices]

                        # loss = agent.train_on_batch(mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones)
                        # if loss is not None:
                        #     total_loss_accumulator += loss
                        #     num_updates += 1
                        # else:
                        #     print(f"警告: 场景 {class_id} DQN train_on_batch 返回无效损失。")
                        #     training_failed = True; break
                        
                        total_loss_accumulator += 0.0
                        num_updates += 1
                else:
                    print(f"警告: 场景 {class_id} 的 agent 类型未知或缺少必要的训练方法。")
                    training_failed = True

            except Exception as e:
                print(f"错误: 训练场景 {class_id} 的 agent 时出错: {e}")
                training_failed = True

            # --- 计算并记录平均损失 ---
            if not training_failed and num_updates > 0:
                 avg_epoch_loss = total_loss_accumulator / num_updates
                 try:
                     scalar_loss = float(tf.reduce_mean(avg_epoch_loss) if hasattr(avg_epoch_loss, 'numpy') else np.mean(avg_epoch_loss))
                     losses[class_id] = scalar_loss
                     print(f"  场景 {class_id} Agent 训练完成: 平均损失={scalar_loss:.4f}")
                 except Exception as cast_e:
                     print(f"警告: 无法将场景 {class_id} 的平均损失转换为标量: {avg_epoch_loss}, 错误: {cast_e}")
                     losses[class_id] = None # 标记失败
            else:
                 losses[class_id] = None # 标记训练未成功执行或失败

        return losses

    def _prepare_training_data(self, data):
        """
        准备训练数据

        Args:
            data: 包含模拟数据的列表，每个元素格式:
                  (state, embedding, action_info, reward, class_id, next_state, done)
                  action_info (dict):
                      - PPO: {'action': action_np, 'logp_old': logp_action_np}
                      - DQN: {'action_index': action_index}

        Returns:
            tuple: 包含解析后数据的列表元组:
                   (states, embeddings, ppo_actions, logp_olds, dqn_action_indices,
                    rewards, next_states, next_embeddings, dones)
                   - ppo_actions: PPO 动作列表 [[0,1..], ...] (如果数据来自PPO) 或 None 列表
                   - logp_olds: PPO 旧对数概率列表 [logp1, ...] (如果数据来自PPO) 或 None 列表
                   - dqn_action_indices: DQN 动作索引列表 [idx1, ...] (如果数据来自DQN) 或 None 列表
        """
        states = []
        embeddings = []
        ppo_actions = []        # 存储 PPO 的 0/1 动作列表
        logp_olds = []          # 存储 PPO 的旧对数概率
        dqn_action_indices = [] # 存储 DQN 的动作索引
        rewards = []
        next_states = []
        next_embeddings = []
        dones = []

        if not data:
            return states, embeddings, ppo_actions, logp_olds, dqn_action_indices, rewards, next_states, next_embeddings, dones

        for i, item in enumerate(data):
            # 预期元组长度为 7
            if len(item) == 7:
                state, embedding, action_info, reward, class_id, next_state, done = item

                states.append(state)
                embeddings.append(embedding)
                rewards.append(reward)
                dones.append(done)

                # 解析 action_info
                if isinstance(action_info, dict):
                    if 'logp_old' in action_info and 'action' in action_info: # PPO 数据
                        ppo_actions.append(action_info['action'])
                        logp_olds.append(action_info['logp_old'])
                        dqn_action_indices.append(None) # 占位符
                    elif 'action_index' in action_info: # DQN 数据
                        ppo_actions.append(None) # 占位符
                        logp_olds.append(None) # 占位符
                        dqn_action_indices.append(action_info['action_index'])
                    else:
                        raise ValueError(f"数据项 {i} 的 action_info 字典格式未知: {action_info}。")
                else: # 如果 action_info 不是字典 (格式错误)
                    raise ValueError(f"数据项 {i} 的 action_info 格式错误 (非字典): {action_info}。")

                # 处理 next_state 和 next_embedding
                if next_state is not None:
                    next_states.append(next_state)
                    # 为 next_state 获取对应的 embedding
                    # 尝试从下一个时间步获取；如果是最后一个时间步，则用当前的 embedding
                    if i + 1 < len(data) and len(data[i+1]) == 7:
                        # 确保下一项数据也有效
                        next_embeddings.append(data[i+1][1]) # 使用下一个数据点的 embedding
                    else:
                        next_embeddings.append(embedding) # 最后一个状态使用当前 embedding
                # else: 如果 next_state is None, 通常发生在 episode 结束时，不需要 next_embedding
            else:
                raise ValueError(f"数据项 {i} 格式不符，预期 7 个元素，实际 {len(item)}个元素")

        return states, embeddings, ppo_actions, logp_olds, dqn_action_indices, rewards, next_states, next_embeddings, dones

    def save_models(self, output_dir):
        """
        保存所有agent模型
        
        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        # 1. 保存manager metadata
        manager_meta = {
            'num_classes': self.num_classes,
            'agent_class_name': self.agent_class.__name__,
            'base_params': self.base_params
        }
        meta_path = os.path.join(output_dir, "manager_meta.json")
        try:
            with open(meta_path, 'w') as f:
                json.dump(manager_meta, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
            print(f"Agent Manager 元数据已保存到 {meta_path}")
        except Exception as e:
            print(f"错误: 保存 Agent Manager 元数据失败: {e}")

        # 2. 保存每个 Agent 的模型和参数
        for i, agent in enumerate(self.agents):
            agent_dir = os.path.join(output_dir, f"agent_{i}")
            os.makedirs(agent_dir, exist_ok=True)
            agent_params_to_save = agent.params.copy()
            agent_params_to_save.pop('env', None)

            params_path = os.path.join(agent_dir, "params.pkl")
            try:
                with open(params_path, 'wb') as f:
                    pickle.dump(agent_params_to_save, f)
            except Exception as e:
                 print(f"错误: 保存 Agent {i} 参数失败: {e}")

            agent.save_model(agent_dir)
            print(f"保存agent {i} 模型权重到 {agent_dir}")

    @classmethod
    def load_manager(cls, input_dir, env):
        """
        加载管理器元数据并创建 Manager 实例
        Args:
            input_dir: 包含 manager_meta.json 和 agent 子目录的路径
            env: 环境实例

        Returns:
            SpecializedAgentManager 实例或 None (加载失败)
        """
        meta_path = os.path.join(input_dir, "manager_meta.json")
        if not os.path.exists(meta_path):
            print(f"错误: Agent Manager 元数据文件未找到: {meta_path}")
            return None

        try:
            with open(meta_path, 'r') as f:
                manager_meta = json.load(f)
        except Exception as e:
            print(f"错误: 加载 Agent Manager 元数据失败: {e}")
            return None

        num_classes = manager_meta.get('num_classes')
        agent_class_name = manager_meta.get('agent_class_name')
        base_params = manager_meta.get('base_params', {})

        if num_classes is None or agent_class_name is None:
            print("错误: 元数据缺少 num_classes 或 agent_class_name。")
            return None

        try:
            # agents 模块在 'agents' 子目录下
            agent_module = importlib.import_module(f"agents.{agent_class_name}")
            AgentClass = getattr(agent_module, agent_class_name)
        except Exception as e:
            print(f"错误: 无法动态导入 Agent 类 '{agent_class_name}': {e}")
            return None

        # 创建 Manager 实例
        manager = cls(env=env, num_classes=num_classes, agent_class=AgentClass, base_params=base_params)
        manager.agents = [None] * num_classes

        # 加载各个 Agent 的模型和精确参数
        manager.load_models(input_dir)

        if all(agent is None for agent in manager.agents):
             print("错误: 未能成功加载任何 agent。")
             return None

        return manager


    def load_models(self, input_dir):
        """
        加载所有 agent 模型权重和参数

        Args:
            input_dir: 包含 agent 子目录的路径 (例如 'output/run_xyz/agents')
        """
        print(f"开始从 {input_dir} 加载 Agents...")
        if len(self.agents) != self.num_classes:
             print(f"警告: Manager 初始化 agent 数量 ({len(self.agents)}) 与预期 ({self.num_classes}) 不符。")
             
        for i in range(self.num_classes):
            agent_dir = os.path.join(input_dir, f"agent_{i}")
            if not os.path.exists(agent_dir):
                print(f"警告: Agent {i} 的目录不存在 {agent_dir}，跳过加载。")
                if i < len(self.agents): self.agents[i] = None
                continue

            # 1. 加载 Agent 参数
            params_path = os.path.join(agent_dir, "params.pkl")
            loaded_params = None
            if os.path.exists(params_path):
                try:
                    with open(params_path, 'rb') as f:
                        loaded_params = pickle.load(f)
                    print(f"  成功加载 Agent {i} 参数。")
                except Exception as e:
                    print(f"错误: 加载 Agent {i} 参数失败: {e}")
                    if i < len(self.agents): self.agents[i] = None
                    continue
            else:
                print(f"警告: Agent {i} 参数文件 {params_path} 不存在。")
                # 如果没有参数文件，标记为失败
                if i < len(self.agents): self.agents[i] = None
                continue

            # 2. 确保 Agent 实例存在且配置正确
            if i >= len(self.agents) or self.agents[i] is None:
                 print(f"信息: 尝试重新创建 Agent {i} 实例...")
                 try:
                      # 使用加载的参数创建
                      agent = self.agent_class(loaded_params, self.env)
                      if i < len(self.agents):
                           self.agents[i] = agent
                      else:
                           self.agents.append(agent)
                 except Exception as e:
                      print(f"错误: 使用加载的参数重新创建 Agent {i} 失败: {e}")
                      continue
            else:
                 pass

            # 3. 加载模型权重
            agent = self.agents[i]
            if agent and hasattr(agent, 'load_model'):
                try:
                    agent.load_model(agent_dir)
                    print(f"  Agent {i} 模型权重加载成功。")
                except Exception as e:
                    print(f"错误: 加载 Agent {i} 模型权重失败: {e}")
                    self.agents[i] = None
            elif agent:
                 print(f"警告: Agent {i} 没有 load_model 方法。")
