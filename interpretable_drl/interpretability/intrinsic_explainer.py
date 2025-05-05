# -*- coding: utf-8 -*-
"""
内生解释器模块，生成决策行为的解释
"""
import numpy as np
from datetime import datetime

class IntrinsicExplainer:
    """内生可解释模块，生成决策解释"""
    
    def __init__(self, soft_tree, env_config, state_names=None):
        """
        初始化内生解释器
        
        Args:
            soft_tree: 软决策树对象
            env_config: 环境配置
            state_names: 状态变量名称列表
        """
        self.soft_tree = soft_tree
        self.env_config = env_config
        self.state_names = state_names or [f"Feature_{i}" for i in range(len(env_config['states']))]
        
        # 场景描述模板
        self.scene_templates = [
            "系统存储已满",
            "峰值流量已经通过",
            "多个水流同时汇入一个位置",
            "降雨强度增加",
            "水位稳定",
            "水位上升",
            "水位下降"
        ]
        
        # 决策理由模板
        self.reason_templates = [
            "需要应对即将到来的峰值流量",
            "需要释放存储空间以应对未来降雨",
            "避免溢流",
            "减少洪水",
            "优化水位分布",
            "控制污水泵以满足处理厂需求",
            "控制雨水泵以减少CSO排放"
        ]
        
        # 后果预测模板
        self.consequence_templates = [
            "预期可以减少 {:.2f}% 的CSO排放和洪水",
            "预期可以平衡系统水位分布",
            "预期可以为未来降雨预留 {:.2f}% 的系统容量",
            "该控制可能导致 {:.2f}% 的溢流增加但降低了 {:.2f}% 的洪水风险",
            "该控制暂时增加CSO排放以避免更严重的洪水"
        ]
        
        # 泵名称
        self.pump_names = [
            "CC-S1", "CC-S2", "JK-S",  # 污水泵
            "CC-R1", "CC-R2", "JK-R1", "JK-R2"  # 雨水泵
        ]
    
    def generate_explanation(self, state, embedding, action, class_id):
        """
        生成决策解释
        
        Args:
            state: 当前状态
            embedding: 策略嵌入向量
            action: 选择的动作
            class_id: 场景类别ID
            
        Returns:
            explanation: 解释字典，包含场景、决策路径、理由、后果和备选方案
        """
        # 1. 场景解释（基于类别ID）
        if class_id < len(self.scene_templates):
            scene = self.scene_templates[class_id]
        else:
            scene = f"场景 {class_id}"
        
        # # 2. 决策路径（分析决策树分支）mlp
        # path = self._analyze_decision_path(state)
        
        # # 3. 决策理由
        # reason_id = class_id % len(self.reason_templates)
        # reason = self.reason_templates[reason_id]

        # # 添加特征重要性
        # features = self._identify_important_features(state, path)
        # if features:
        #     feature_str = ", ".join([f"{self.state_names[f]}={state[f]:.2f}" for f in features])
        #     reason += f"。关键因素: {feature_str}"

        # 2. 决策路径 (使用的 soft_tree 方法)
        path_nodes, predicted_class = self.soft_tree.predict_path(state)
        # predicted_class 应该与传入的 class_id (来自 embedding argmax) 大致相同
        if predicted_class != class_id:
             print(f"警告 (Explainer): 树预测类别 {predicted_class} 与 embedding 最大值类别 {class_id} 不符。")

        # 从节点路径生成规则描述
        path = self._analyze_decision_path(path_nodes)

        # 3. 决策理由 (使用新的 soft_tree 方法获取特征)
        reason_id = class_id % len(self.reason_templates)
        reason = self.reason_templates[reason_id]
        # 使用从路径中提取的特征
        features_indices = self._identify_important_features_from_path(path_nodes)
        if features_indices:
            feature_str = ", ".join([f"{self.state_names[f_idx]}={state[f_idx]:.2f}" for f_idx in features_indices if f_idx < len(self.state_names)])
            reason += f"。关键因素: {feature_str}"
        
        # 4. 预期后果
        consequence_id = class_id % len(self.consequence_templates)
        if consequence_id == 0:
            reduction = 10 + class_id * 5  # 示例减少百分比
            consequence = self.consequence_templates[consequence_id].format(reduction)
        elif consequence_id == 2:
            capacity = 15 + class_id * 3  # 示例系统容量
            consequence = self.consequence_templates[consequence_id].format(capacity)
        elif consequence_id == 3:
            overflow = 5 + class_id
            flood = 15 + class_id * 2
            consequence = self.consequence_templates[consequence_id].format(overflow, flood)
        else:
            consequence = self.consequence_templates[consequence_id]
        
        # 5. 备选方案（从其他叶节点选择）
        alternatives = self._generate_alternatives(state, embedding, class_id)
        
        # 6. 格式化泵状态
        pump_status = self._format_pump_status(action)
        
        # 整合解释
        explanation = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scene": scene,
            "decision_path": path,
            "reason": reason,
            "consequence": consequence,
            "alternatives": alternatives,
            "pump_status": pump_status
        }
        
        return explanation
    
    def _analyze_decision_path(self, path_nodes):
        """
        分析状态在决策树中的路径节点，生成规则描述列表

        Args:
            path_nodes: 决策路径上的节点索引列表

        Returns:
            path_description: 决策路径规则描述列表
        """
        if not path_nodes or not hasattr(self.soft_tree, 'model') or not hasattr(self.soft_tree.model, 'tree_'):
            return ["路径不可用"]

        tree = self.soft_tree.model.tree_
        path_description = []

        # 遍历路径上的内部节点 (最后一个是叶节点，没有规则)
        for i in range(len(path_nodes) - 1):
            node_id = path_nodes[i]
            child_id = path_nodes[i+1]

            if node_id < 0 or node_id >= tree.node_count: continue 
            # 检查是否是叶节点
            if tree.children_left[node_id] == tree.children_right[node_id]:
                 continue

            # 获取分裂规则
            rule = self.soft_tree.get_decision_rule(node_id, self.state_names)

            # 走向 (左 <=, 右 >)
            if child_id == tree.children_left[node_id]:
                # 左子节点表示满足 <= 条件
                path_description.append(f"节点 {node_id}: {rule} (True)")
            elif child_id == tree.children_right[node_id]:
                # 右子节点，表示满足 > 条件
                # 修改规则以表示 >
                rule_parts = rule.split("<=")
                if len(rule_parts) == 2:
                    negated_rule = f"{rule_parts[0].strip()} > {rule_parts[1].strip()}"
                    path_description.append(f"节点 {node_id}: {negated_rule} (True)")
                else: # 如果规则格式不符
                     path_description.append(f"节点 {node_id}: {rule} (False?)")
            else:
                 path_description.append(f"节点 {node_id}: {rule} (未知走向)")


        return path_description

    def _identify_important_features_from_path(self, path_nodes):
        """
        从决策路径中识别重要的特征索引

        Args:
            path_nodes: 决策路径上的节点索引列表

        Returns:
            important_features: 重要特征索引列表
        """
        if not path_nodes or not hasattr(self.soft_tree, 'model') or not hasattr(self.soft_tree.model, 'tree_'):
            return set()

        tree = self.soft_tree.model.tree_
        important_features = set()

        # 遍历路径上的内部节点
        for node_id in path_nodes[:-1]:
             if node_id < 0 or node_id >= tree.node_count: continue
             # 检查是否是内部节点
             if tree.children_left[node_id] != tree.children_right[node_id]:
                  feature_index = tree.feature[node_id]
                  if feature_index >= 0:
                       important_features.add(feature_index)

        return list(important_features)
    
    def _analyze_decision_path_mlp(self, state):
        """
        分析状态在决策树中的路径
        
        Args:
            state: 当前状态
            
        Returns:
            path_description: 决策路径描述
        """
        # 从软决策树获取路径
        try:
            path, _ = self.soft_tree.predict_path(state)
        except:
            # 如果获取路径失败，返回空路径
            return []
            
        # 实现决策路径分析算法
        path_description = []
        
        # 遍历路径上的节点
        for i in range(len(path) - 1):  # 最后一个是叶节点
            node_id = path[i]
            
            # 获取该节点的决策规则
            try:
                rule = self.soft_tree.get_decision_rule(node_id)
                path_description.append(rule)
            except:
                # 如果获取规则失败，使用默认描述
                path_description.append(f"决策节点 {node_id}")
        
        return path_description
    
    def _identify_important_features_mlp(self, state, path):
        """
        识别对决策最重要的特征
        
        Args:
            state: 当前状态
            path: 决策路径
            
        Returns:
            important_features: 重要特征索引列表
        """
        # 从决策路径中提取重要特征
        important_features = []
        
        for step in path:
            parts = step.split()
            if len(parts) >= 3 and parts[0].startswith("feature_"):
                try:
                    feature_idx = int(parts[0].replace("feature_", ""))
                    if feature_idx not in important_features:
                        important_features.append(feature_idx)
                except:
                    continue
        
        # 如果从路径中未找到重要特征
        if not important_features:
            for i, name in enumerate(self.state_names):
                if "水位" in name or "流量" in name or "降雨" in name:
                    important_features.append(i)
                    if len(important_features) >= 3:
                        break
        
        return important_features[:3]  # 返回最多3个重要特征
    
    def _generate_alternatives(self, state, embedding, current_class):
        """
        生成备选方案
        
        Args:
            state: 当前状态
            embedding: 策略嵌入向量
            current_class: 当前场景类别
            
        Returns:
            alternative: 备选方案描述
        """
        # 找出概率第二高的类别
        probs = embedding.copy()
        probs[current_class] = 0  # 排除当前类别
        alt_class = np.argmax(probs)
        
        # 确保备选方案与当前方案不同
        if alt_class == current_class:
            alt_class = (current_class + 1) % self.soft_tree.num_classes
            
        # 生成备选方案描述
        if alt_class < len(self.scene_templates):
            alt_scene = self.scene_templates[alt_class]
        else:
            alt_scene = f"场景 {alt_class}"
            
        alt_reason_id = alt_class % len(self.reason_templates)
        alt_reason = self.reason_templates[alt_reason_id]
        
        return f"备选方案: 基于'{alt_scene}'场景采取不同控制策略。理由: {alt_reason}"
    
    def _format_pump_status(self, action):
        """
        格式化泵状态为人类可读形式
        
        Args:
            action: 泵控制动作
            
        Returns:
            formatted_status: 格式化后的泵状态
        """
        formatted = []
        
        if isinstance(action, (list, np.ndarray)) and len(action) <= len(self.pump_names):
            for i, status in enumerate(action):
                if i < len(self.pump_names):
                    pump_name = self.pump_names[i]
                else:
                    pump_name = f"泵_{i+1}"
                    
                state_str = "开启" if status == 1 else "关闭"
                formatted.append(f"{pump_name}: {state_str}")
        # 如果action是一个整数（如DQN的动作索引）
        elif isinstance(action, (int, np.integer)):
            try:
                binary = format(action, '07b')
                for i, bit in enumerate(binary):
                    if i < len(self.pump_names):
                        pump_name = self.pump_names[i]
                    else:
                        pump_name = f"泵_{i+1}"
                        
                    state_str = "开启" if bit == '1' else "关闭"
                    formatted.append(f"{pump_name}: {state_str}")
            except:
                formatted.append(f"动作: {action}")
        else:
            formatted.append(f"动作: {action}")
        
        return formatted