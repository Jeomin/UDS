#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import datetime

def parse_args():
    """
    解析命令行参数
    
    Returns:
        args: 解析后的参数
    """
    parser = argparse.ArgumentParser(description='决策树可解释DRL框架')
    
    # 运行模式
    parser.add_argument('--mode', type=str, 
                        choices=['intrinsic_train', 'train_ppo', 'train_dqn', 'evaluate'],
                        default='intrinsic_train', 
                        help='运行模式')
    
    # 模型类型， 用于评估
    parser.add_argument('--model', type=str, 
                        choices=['ppo', 'dqn', 'intrinsic'],
                        default='ppo', 
                        help='模型类型')
    
    # agent-type参数
    parser.add_argument('--agent-type', type=str, 
                        choices=['ppo', 'dqn'],
                        default='ppo', 
                        help='agent类型 (用于内生可解释系统)')
    
    # 内生可解释框架参数
    parser.add_argument('--num-classes', type=int, default=4,
                       help='场景类别数量')
    
    parser.add_argument('--tree-depth', type=int, default=4,
                       help='决策树深度')
    
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='软决策树温度参数')
    
    parser.add_argument('--joint-training', action='store_true',
                       help='是否使用联合训练')
    
    # 训练参数
    parser.add_argument('--num-train', type=int, default=50,
                       help='训练样本数量')
    
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批大小')
    
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='学习率')
    
    # 评估参数
    parser.add_argument('--num-test', type=int, default=10,
                       help='测试样本数量')
    
    # 路径设置
    parser.add_argument('--output-dir', type=str, default='output',
                       help='输出目录')
    
    parser.add_argument('--model-path', type=str, default=None,
                       help='加载预训练模型的路径')
                       
    parser.add_argument('--env-path', type=str, default='chaohu',
                       help='SWMM环境配置文件路径')
    
    parser.add_argument('--rain-path', type=str, default='training_raindata.npy',
                       help='训练降雨数据路径')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    parser.add_argument('--verbose', action='store_true',
                       help='是否输出详细信息')
    
    args = parser.parse_args()
    return args


def train_dqn(args):
    """训练DQN模型并进行后解释"""
    from scripts.train_interpretable_dqn import train_interpretable_dqn
    train_interpretable_dqn(args)


def train_ppo(args):
    """训练PPO模型并进行后解释"""
    from scripts.train_interpretable_ppo import train_interpretable_ppo
    train_interpretable_ppo(args)


def train_intrinsic(args):
    """训练内生可解释系统"""
    from scripts.train_intrinsic_interpretable import train_intrinsic_interpretable
    train_intrinsic_interpretable(args)


def evaluate_model(args):
    print("评估可解释系统")
    from scripts.evaluate import evaluate_intrinsic
    evaluate_intrinsic(args)


def main():
    args = parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if args.mode == 'intrinsic_train':
        print("训练内生可解释系统")
        print(f"使用agent类型: {args.agent_type}")
        train_intrinsic(args)
    elif args.mode == 'train_ppo':
        print("训练PPO模型并进行后解释")
        train_ppo(args)
    elif args.mode == 'train_dqn':
        print("训练DQN模型并进行后解释")
        train_dqn(args)
    elif args.mode == 'evaluate':
        evaluate_model(args)
    else:
        print(f"未知运行模式: {args.mode}")


if __name__ == "__main__":
    main()