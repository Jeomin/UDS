# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 20:16:26 2022

@author: chongtm
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import yaml
import graphviz
from sklearn import tree


def Getdata():
    
    action_table=pd.read_csv('./action_table.csv').values[:,1:]
    dqn_data=np.load('./DQN/Results/0.npy',allow_pickle=True).tolist()
    dqn_s=np.array(dqn_data['state'])[1:,:]
    dqn_a=np.array(dqn_data['action'])
    
    ppo_data=np.load('./PPO/Results/0.npy',allow_pickle=True).tolist()
    ppo_s=np.array(ppo_data['state'])[1:,:]
    ppo_a=np.array(ppo_data['action'])[:,0,:]
    
    
    for idx in range(1,1000):
        dqn_data=np.load('./DQN/Results/'+str(idx)+'.npy',allow_pickle=True).tolist()
        dqn_s=np.concatenate((dqn_s,np.array(dqn_data['state'])[1:,:]),axis=0)
        dqn_a=np.concatenate((dqn_a,np.array(dqn_data['action'])),axis=0)
        
        ppo_data=np.load('./PPO/Results/'+str(idx)+'.npy',allow_pickle=True).tolist()
        ppo_s=np.concatenate((ppo_s,np.array(ppo_data['state'])[1:,:]),axis=0)
        ppo_a=np.concatenate((ppo_a,np.array(ppo_data['action'])[:,0,:]),axis=0)
    
    print(dqn_s.shape,dqn_a.shape,ppo_s.shape,ppo_a.shape)
    return dqn_s,dqn_a,ppo_s,ppo_a


def get_y(data):
    #从数据中泵的策略给出分类的y
    y=np.zeros((data.shape[0],1))
    output = {}
    t = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for m in range(2):
                    output[str(i)+','+str(j)+','+str(k)+','+str(m)]=t
                    t+=1
    
    for line in range(y.shape[0]):
        i1,i2,i3,i4=int(np.sum(data[line,0:2])),int(np.sum(data[line,2:4])),int(np.sum(data[line,4:6])),int(np.sum(data[line,6:7]))
        y[line]=output[str(i1)+','+str(i2)+','+str(i3)+','+str(i4)]
    
    return y

def get_y2(data):
    #从数据中泵的策略给出分类的y
    action_table=pd.read_csv('./action_table.csv').values[:,1:]
    
    y=np.zeros((data.shape[0],1))
    for line in range(y.shape[0]):
        y[line]=np.where((action_table==data[line,:]).all(axis=1))
    return y


if __name__=='__main__':
    dqn_s,dqn_a,ppo_s,ppo_a=Getdata()
    
    config = yaml.load(open("chaohu.yaml"), yaml.FullLoader)
    #根据得到的决策树，把几个筛选条件节点名字给出，便于画图
    feature_names=[]
    for i in range(len(config['states'])):
        if i == 8:
            feature_names.append('WS02006229 flow')
        elif i == 9:
            feature_names.append('WS02006116 flow')
        elif i==10:
            feature_names.append('YS02001907 flow')
        elif i==11:
            feature_names.append('YS02001649 flow')
        elif i==12:
            feature_names.append('CC-1 flow')
        elif i==13:
            feature_names.append('CC-2 flow')
        elif i==14:
            feature_names.append('JK-1 flow')
        elif i==15:
            feature_names.append('JK-2 flow')
        elif i==16:
            feature_names.append('WSC flow')
        elif i==17:
            feature_names.append('Rain intensity')
        else:
            feature_names.append(config['states'][i][0])
    
    
    dqn_y=get_y(dqn_a)
    tree_dqn=DecisionTreeClassifier(criterion="gini",min_impurity_decrease=0.001,ccp_alpha=0.0001)
    tree_dqn.fit(dqn_s,dqn_y)
    dot_dqn = tree.export_graphviz(tree_dqn, out_file=None, 
                         feature_names=feature_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
    graph = graphviz.Source(dot_dqn)
    graph.render('./dqn9_tree')
    
    
    ppo_y=get_y(ppo_a)
    tree_ppo=DecisionTreeClassifier(criterion="gini",min_impurity_decrease=0.001,ccp_alpha=0.0001)
    tree_ppo.fit(ppo_s,ppo_y)
    dot_ppo = tree.export_graphviz(tree_ppo, out_file=None, 
                         feature_names=feature_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
    graph = graphviz.Source(dot_ppo)
    graph.render('./ppo8_tree')
    
    
