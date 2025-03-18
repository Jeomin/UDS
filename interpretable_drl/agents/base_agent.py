# interpretable_drl/agents/base_agent.py
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract base class for all reinforcement learning agents
    """
    
    def __init__(self, params, env):
        """
        初始化agent
        
        Args:
            params (dict): agent参数
            env: 环境对象
        """
        self.params = params
        self.env = env
        
    @abstractmethod
    def choose_action(self, state, train_log=False):
        """
        选择动作
        
        Args:
            state: 当前状态
            train_log (bool): 是否处于训练模式
            
        Returns:
            action: 选择的动作
        """
        pass
    
    @abstractmethod
    def train(self, rainfall_data):
        """
        训练agent
        
        Args:
            rainfall_data: 训练用降雨数据
            
        Returns:
            history: 训练历史
        """
        pass
    
    @abstractmethod
    def load_model(self):
        """
        加载已训练的模型
        """
        pass
    
    @abstractmethod
    def test(self, rainfall):
        """
        测试agent
        
        Args:
            rainfall: 测试用降雨数据
            
        Returns:
            test_history: 测试历史
        """
        pass