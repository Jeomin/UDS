interpretable_drl/
│
├── agents/
│   ├── __init__.py
│   ├── base_agent.py       # 代理的基类
│   ├── dqn_agent.py        # DQN代理接口（直接使用原始DQN.py）
│   ├── ppo_agent.py        # PPO代理接口（直接使用原始PPO.py）
│   └── interpretable_agent.py  # 可解释代理实现
│
├── environment/
│   ├── __init__.py
│   └── swmm_env.py         # SWMM环境接口（直接使用原始SWMM_ENV.py）
│
├── interpretability/
│   ├── __init__.py
│   ├── soft_tree.py        # 软决策树实现
│   ├── tree_surrogate.py   # 树形代理模型
│   ├── sensitivity.py      # Sobol敏感性分析
│   ├── conditional_prob.py # 条件概率分析
│   └── explainer.py        # 主要解释器接口
│
├── utils/
│   ├── __init__.py
│   ├── rainfall.py         # 降雨数据功能（直接使用原始Rainfall_data.py）
│   └── visualization.py    # 可视化工具
│
├── scripts/
│   ├── train_interpretable_dqn.py  # 训练解释性DQN
│   ├── train_interpretable_ppo.py  # 训练解释性PPO
│   └── evaluate.py                 # 评估模型
│
└── main.py                 # 主入口

SWMM环境所需的配置和数据文件
从SWMM_ENV.py文件和相关代码分析，SWMM环境需要以下关键文件：

基础配置文件:

.inp文件：SWMM的主要配置文件，定义了排水系统的物理组成部分
.yaml文件：环境配置文件，定义状态、动作和奖励目标


降雨数据:

training_raindata.npy：训练用的降雨数据
test_raindata.npy：测试用的降雨数据


动作配置:

DQN_action_table.csv：DQN算法的泵控制动作映射表

基础INP文件 (self.params['orf']+'.inp'):

这是原始的SWMM配置文件，包含了排水系统的全部配置信息
包括节点、管道、泵站、贮存池等物理组件的定义
没有降雨数据或者使用默认降雨数据


生成的带雨水数据的INP文件 (self.params['orf']+'_rain.inp'):

这是程序运行时动态生成的文件
基于原始INP文件，但添加了特定的降雨时间序列数据
用于单次仿真


可能的其他输出文件:

.rpt文件: 报告文件，包含仿真结果的详细统计信息
.out文件: 二进制输出文件，包含仿真的时间序列数据