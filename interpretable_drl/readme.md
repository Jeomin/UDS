# 决策树可解释DRL框架

本项目实现了一个基于决策树的可解释深度强化学习(DRL)框架，用于城市排水系统的实时控制。该框架结合了软决策树与专业化DRLagent，提供了更好的可解释性和性能。

## 1. 框架结构

本框架包含两种可解释性方法：
- **后解释框架**：传统方法，先训练DRL模型，然后使用解释方法分析其行为
- **内生可解释框架**：创新方法，在DRL训练过程中集成解释性组件

### 数据流
```
输入 → 决策树 → 策略嵌入 → DRLagent → 动作
 ↑    ↓       ↓        ↓       ↓
 └─────────────── 解释模块 ──────────────┘
```

每次做出决策时，解释模块会回答：
- 当前是什么场景？(基于决策树的场景识别)
- 为什么选择这个控制策略？(基于决策路径)
- 预期产生什么效果？(基于历史数据和模型预测)
- 备选方案？(基于决策树的其他分支)

## 2. 系统组件

### 输入
- 状态变量(s_t): 关键节点的水位流量、降雨强度、历史数据(前n个时间步)

### 软决策树模块
- 基于分裂决策的softmax
- 可端到端训练
- 可视化决策路径

### 策略嵌入(Embedding)
- 结合决策树路径信息和当前状态
- 生成包含场景语义的策略embedding

### DRL专门化agent
- 针对不同场景的多个专门化agent
- 每个agent接收状态与策略embedding
- 输出泵控制决策

### 解释模块
- 实时解释生成
- 决策路径可视化
- 策略选择理由
- 动作后果预测

## 3. 训练流程

### 预训练阶段
- 决策树预训练：仅训练决策树参数
- 各场景DRL预训练：分别训练每个agent

### 联合训练阶段
- 交替训练决策树和DRL
- 或完全联合训练

## 4. 使用方法

### 训练内生可解释模型
```bash
python main.py --mode intrinsic_train --num-classes 5 --tree-depth 4 --temperature 1.0 --epochs 100 --num-train 50 --agent-type ppo
```

### 训练传统DQN/PPO模型并进行后解释
```bash
python main.py --mode train --model ppo  # 或 dqn
```

### 评估模型
```bash
python main.py --mode evaluate --model intrinsic --model-path [model-path]
```

## 5. 目录结构

```
interpretable_drl/
│
├── agents/
│   ├── __init__.py             # 初始化包
│   ├── base_agent.py           # agent的基类 
│   ├── DQN.py                  # DQNagent实现
│   └── PPO.py                  # PPOagent实现
│
├── environment/
│   ├── __init__.py             # 初始化包
│   └── swmm_env.py             # SWMM环境封装
│
├── interpretability/
│   ├── __init__.py             # 初始化包
│   ├── soft_tree.py            # 软决策树实现
│   ├── intrinsic_soft_tree.py  # 内生软决策树
│   ├── specialized_agents.py   # 专门化agent管理器
│   ├── intrinsic_explainer.py  # 内生解释器
│   ├── tree_surrogate.py       # 树形替代模型
│   ├── sensitivity.py          # Sobol敏感性分析
│   ├── conditional_prob.py     # 条件概率分析 
│   └── explainer.py            # 解释器
│
├── scripts/
│   ├── train_intrinsic_interpretable.py    # 训练内生可解释系统
│   ├── train_interpretable_dqn.py          # 训练可解释DQN
│   ├── train_interpretable_ppo.py          # 训练可解释PPO
│   └── evaluate.py                         # 评估模型
│
└── main.py                     # 主入口程序
```

## 6. 依赖项

- Python 3.7+
- TensorFlow 2.12.0
- NumPy
- Matplotlib
- Scikit-learn
- SALib (敏感性分析)
- SWMM-Python API

## 7. 引用

如果您在研究中使用了本框架，请引用：
```
@article{tian2024improving,
  title={Improving the interpretability of deep reinforcement learning in urban drainage system operation},
  author={Tian, Wenchong and Fu, Guangtao and Xin, Kunlun and Zhang, Zhiyu and Liao, Zhenliang},
  journal={Water Research},
  volume={249},
  pages={120912},
  year={2024},
  publisher={Elsevier}
}
```