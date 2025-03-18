# -*- coding: utf-8 -*-
"""
解释器生成控制行为的解释
"""
import numpy as np
from datetime import datetime

class Explainer:
    """
    生成关于控制决策的自然语言解释
    """
    
    def __init__(self, state_names=None, action_names=None):
        """
        初始化解释器
        
        Args:
            state_names: 状态变量名称列表
            action_names: 动作名称列表
        """
        # 保存状态和动作名称
        self.state_names = state_names
        self.action_names = action_names
        
        # 默认的特征名称
        self.feature_names = {
            0: "CC-storage水位",
            1: "JK-storage水位",
            2: "WS02006229水位",
            3: "WS02006116水位",
            4: "WS02006235水位",
            5: "WS02006251水位",
            6: "YS02001907水位",
            7: "YS02001649水位",
            8: "WS02006229流量",
            9: "WS02006116流量",
            10: "YS02001907流量",
            11: "YS02001649流量",
            12: "CC-1流量",
            13: "CC-2流量",
            14: "JK-1流量",
            15: "JK-2流量",
            16: "WSC流量",
            17: "降雨强度"
        }

        if state_names:
            for i, name in enumerate(state_names):
                if i < len(self.feature_names):
                    self.feature_names[i] = name
        
        self.pump_names = [
            "CC-S1", "CC-S2", "JK-S",
            "CC-R1", "CC-R2", "JK-R1", "JK-R2"
        ]
        
        # 场景模板
        self.scenario_templates = [
            "系统存储已满",
            "峰值流量已经通过",
            "多个水流同时汇入一个位置",
            "降雨强度增加",
            "水位稳定",
            "水位上升",
            "水位下降"
        ]
        
        # 决策理由模板
        self.decision_templates = [
            "避免溢流",
            "减少洪水",
            "优化水位分布",
            "为未来降雨预留容量",
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
    
    def explain_action(self, state, action, decision_path=None):
        """
        生成关于控制动作的解释
        
        Args:
            state: 当前状态
            action: 选择的动作
            decision_path: 决策树路径（可选）
            
        Returns:
            explanation: 解释字典，包含场景、决策理由和预期后果
        """
        # 处理动作
        if isinstance(action, list) and len(action) == 7:
            # 动作是泵开关状态列表
            pump_status = action
        else:
            # 尝试将动作转换为泵状态
            pump_status = self._convert_action_to_pump_status(action)
        
        # 根据状态识别场景
        scenario = self._identify_scenario(state)
        
        # 根据决策路径和状态生成决策理由
        if decision_path:
            decision_reason = self._generate_decision_reason(state, pump_status, decision_path)
        else:
            decision_reason = self._generate_default_decision_reason(state, pump_status)
        
        # 预测控制后果
        predicted_consequence = self._predict_consequences(state, pump_status)
        
        # 构建解释
        explanation = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": scenario,
            "decision_reason": decision_reason,
            "predicted_consequence": predicted_consequence,
            "pump_status": self._format_pump_status(pump_status)
        }
        
        return explanation
    
    def _identify_scenario(self, state):
        """
        根据状态识别当前场景
        
        Args:
            state: 当前状态
            
        Returns:
            scenario: 场景描述
        """
        try:
            # 提取关键状态变量
            rainfall = state[17] if len(state) > 17 else 0  # 降雨强度
            cc_storage_level = state[0] if len(state) > 0 else 0  # CC存储池水位
            jk_storage_level = state[1] if len(state) > 1 else 0  # JK存储池水位
            
            # 主要管道的流量
            main_pipe_flows = []
            if len(state) > 8:
                main_pipe_flows.append(state[8])  # WS02006229流量
            if len(state) > 9:
                main_pipe_flows.append(state[9])  # WS02006116流量
            if len(state) > 16:
                main_pipe_flows.append(state[16])  # WSC流量
            
            # 场景识别逻辑
            if rainfall > 20:
                scenario_idx = 3  # 降雨强度增加
            elif cc_storage_level > 1.5 and jk_storage_level > 3.5:
                scenario_idx = 0  # 系统存储已满
            elif all(flow < 0.5 for flow in main_pipe_flows) if main_pipe_flows else False:
                scenario_idx = 1  # 峰值流量已经通过
            elif any(flow > 2.0 for flow in main_pipe_flows) if main_pipe_flows else False:
                scenario_idx = 2  # 多个水流同时汇入一个位置
            elif 0.8 < cc_storage_level < 1.2 and 2.0 < jk_storage_level < 3.0:
                scenario_idx = 4  # 水位稳定
            elif cc_storage_level > 1.2 or jk_storage_level > 3.0:
                scenario_idx = 5  # 水位上升
            else:
                scenario_idx = 6  # 水位下降
                
            return self.scenario_templates[scenario_idx]
        except:
            # 默认场景
            return "常规操作条件"
    
    def _generate_decision_reason(self, state, pump_status, decision_path):
        """
        根据决策路径生成决策理由
        
        Args:
            state: 当前状态
            pump_status: 泵开关状态
            decision_path: 决策树路径
            
        Returns:
            reason: 决策理由
        """
        # 如果决策路径可用，尝试提取决策规则
        if decision_path and hasattr(self, 'tree_model') and self.tree_model:
            try:
                rules = self.tree_model.get_path_rules(decision_path)
                if rules:
                    # 将规则转换为自然语言
                    rule_texts = []
                    for rule in rules:
                        # 替换特征名称
                        for i, name in self.feature_names.items():
                            rule = rule.replace(f"feature_{i}", name)
                        rule_texts.append(rule)
                    
                    # 生成理由文本
                    reason = "根据以下条件做出决策: " + ", ".join(rule_texts)
                    return reason
            except:
                pass
        
        # 如果无法使用决策路径生成理由，使用默认方法
        return self._generate_default_decision_reason(state, pump_status)
    
    def _generate_default_decision_reason(self, state, pump_status):
        """
        生成默认决策理由
        
        Args:
            state: 当前状态
            pump_status: 泵开关状态
            
        Returns:
            reason: 决策理由
        """
        try:
            # 提取关键状态变量
            rainfall = state[17] if len(state) > 17 else 0  # 降雨强度
            cc_storage_level = state[0] if len(state) > 0 else 0  # CC存储池水位
            jk_storage_level = state[1] if len(state) > 1 else 0  # JK存储池水位
            
            # 计算开启的泵数量
            sewage_pumps_on = sum(pump_status[:3]) if len(pump_status) >= 3 else 0  # 污水泵
            storm_pumps_on = sum(pump_status[3:]) if len(pump_status) >= 4 else 0   # 雨水泵
            
            # 选择理由
            if rainfall > 10 and storm_pumps_on > 0:
                reason_idx = 1  # 减少洪水
            elif cc_storage_level > 1.0 or jk_storage_level > 3.0:
                if storm_pumps_on > 0:
                    reason_idx = 0  # 避免溢流
                else:
                    reason_idx = 2  # 优化水位分布
            elif rainfall > 5 and rainfall <= 10:
                reason_idx = 3  # 为未来降雨预留容量
            elif sewage_pumps_on > 0:
                reason_idx = 4  # 控制污水泵以满足处理厂需求
            else:
                reason_idx = 5  # 控制雨水泵以减少CSO排放
                
            return self.decision_templates[reason_idx]
        except:
            # 默认理由
            return "基于当前系统状态优化控制"
    
    def _predict_consequences(self, state, pump_status):
        """
        预测控制后果
        
        Args:
            state: 当前状态
            pump_status: 泵开关状态
            
        Returns:
            consequence: 预测后果
        """
        try:
            # 提取关键状态变量
            rainfall = state[17] if len(state) > 17 else 0  # 降雨强度
            cc_storage_level = state[0] if len(state) > 0 else 0  # CC存储池水位
            jk_storage_level = state[1] if len(state) > 1 else 0  # JK存储池水位
            
            # 计算开启的泵数量
            sewage_pumps_on = sum(pump_status[:3]) if len(pump_status) >= 3 else 0  # 污水泵
            storm_pumps_on = sum(pump_status[3:]) if len(pump_status) >= 4 else 0   # 雨水泵
            
            # 根据不同情况预测后果
            if rainfall > 15:
                # 强降雨情况
                if storm_pumps_on >= 3:
                    # 多个雨水泵开启，预计减少洪水
                    reduction = min(80, 30 + 10 * storm_pumps_on)
                    return self.consequence_templates[0].format(reduction)
                else:
                    # 雨水泵开启不足，可能增加溢流风险
                    overflow_increase = max(0, 20 - 5 * storm_pumps_on)
                    flood_decrease = 15 + 10 * storm_pumps_on
                    return self.consequence_templates[3].format(overflow_increase, flood_decrease)
            elif cc_storage_level > 1.0 or jk_storage_level > 3.0:
                # 水位高的情况
                if storm_pumps_on > 0:
                    # 开启雨水泵，预计减少溢流
                    reduction = 20 + 10 * storm_pumps_on
                    return self.consequence_templates[0].format(reduction)
                else:
                    # 暂时增加CSO排放
                    return self.consequence_templates[4]
            else:
                # 常规情况
                capacity_reserve = 30 + 5 * (4 - storm_pumps_on)
                return self.consequence_templates[2].format(capacity_reserve)
        except:
            # 默认后果
            return "预期将维持系统稳定运行"
    
    def _convert_action_to_pump_status(self, action):
        """
        将动作转换为泵状态
        
        Args:
            action: 动作（可能是索引或直接的泵状态列表）
            
        Returns:
            pump_status: 泵开关状态列表
        """
        if isinstance(action, list) and len(action) == 7:
            return action
            
        try:
            if isinstance(action, (int, np.integer)):
                binary = format(action, '07b')
                pump_status = [int(bit) for bit in binary]
                return pump_status
            elif isinstance(action, np.ndarray) and action.size == 7:
                return action.tolist()
            else:
                # 默认所有泵关闭
                return [0] * 7
        except:
            # 出错时返回默认值
            return [0] * 7
    
    def _format_pump_status(self, pump_status):
        """
        格式化泵状态为人类可读形式
        
        Args:
            pump_status: 泵开关状态列表
            
        Returns:
            formatted_status: 格式化后的泵状态
        """
        formatted = []
        for i, status in enumerate(pump_status):
            if i < len(self.pump_names):
                pump_name = self.pump_names[i]
            else:
                pump_name = f"泵_{i+1}"
                
            state_str = "开启" if status == 1 else "关闭"
            formatted.append(f"{pump_name}: {state_str}")
        
        return formatted
    
    def set_tree_model(self, tree_model):
        """
        设置树模型，用于生成基于决策路径的解释
        
        Args:
            tree_model: 树模型对象
        """
        self.tree_model = tree_model