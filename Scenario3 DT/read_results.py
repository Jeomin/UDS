# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 20:39:01 2022

@author: chong
"""
import numpy as np
import matplotlib.pyplot as plt
import yaml


def fig(title):
    for idx in range(6):
        dqn=np.load('./DQN/Results/'+str(idx)+'.npy',allow_pickle=True).tolist()
        ppo=np.load('./PPO/Results/'+str(idx)+'.npy',allow_pickle=True).tolist()
        hc=np.load('./HC/Results/HC'+str(idx)+'.npy',allow_pickle=True).tolist()
        
        figs,axes=plt.subplots(1,2,figsize=(20,20))
        axes.flatten()
        axes[0].plot(dqn[title],label='dqn')
        axes[0].plot(hc[title],label='hc')
        axes[0].plot(ppo[title],label='ppo')
        axes[0].legend()
        axes[0].set_title(title,fontsize=20)
        #axes[1].plot(dqn['C'],label='dqn')
        #axes[1].plot(hc['C'],label='hc')
        #axes[1].plot(ppo['C'],label='ppo')
        #axes[1].set_title('CSO',fontsize=20)
        #axes[1].legend()
        figs.savefig('./'+title+'.tif')



def figfc():
    for idx in range(2):
        dqn=np.load('./DQN/Results/'+str(idx)+'.npy',allow_pickle=True).tolist()
        ppo=np.load('./PPO/Results/'+str(idx)+'.npy',allow_pickle=True).tolist()
        hc=np.load('./HC/Results/HC'+str(idx)+'.npy',allow_pickle=True).tolist()
        opt=np.load('./OPT/Results/'+str(idx)+'.npy',allow_pickle=True).tolist()
        
        figs,axes=plt.subplots(1,3,figsize=(20,20))
        axes.flatten()
        axes[0].plot(dqn['F'],label='dqn')
        axes[0].plot(ppo['F'],label='ppo')
        axes[0].plot(hc['F'],label='hc')
        axes[0].plot(opt['F'],label='opt')
        axes[0].legend()
        axes[0].set_title('Flooding',fontsize=20)
        
        axes[1].plot(dqn['C'],label='dqn')
        axes[1].plot(ppo['C'],label='ppo')
        axes[1].plot(hc['C'],label='hc')
        axes[1].plot(opt['C'],label='opt')
        axes[1].set_title('CSO',fontsize=20)
        axes[1].legend()
        
        axes[2].plot(np.array(dqn['C'])+np.array(dqn['F']),label='dqn')
        axes[2].plot(np.array(ppo['C'])+np.array(ppo['F']),label='ppo')
        axes[2].plot(np.array(hc['C'])+np.array(hc['F']),label='hc')
        axes[2].plot(np.array(opt['C'])+np.array(opt['F']),label='opt')
        axes[2].set_title('Flooding+CSO',fontsize=20)
        axes[2].legend()
        
        figs.savefig('./fc'+str(idx)+'.tif')
        
def dist():
    total=[]
    for idx in range(2):
        dqn=np.load('./DQN/Results/'+str(idx)+'.npy',allow_pickle=True).tolist()
        ppo=np.load('./PPO/Results/'+str(idx)+'.npy',allow_pickle=True).tolist()
        hc=np.load('./HC/Results/HC'+str(idx)+'.npy',allow_pickle=True).tolist()
        opt=np.load('./OPT/Results/'+str(idx)+'.npy',allow_pickle=True).tolist()
        
        
        if (dqn['F'][-1]+dqn['C'][-1]>hc['F'][-1]+hc['C'][-1]) or (ppo['F'][-1]+ppo['C'][-1]> hc['F'][-1]+hc['C'][-1]):
            total.append(idx)
        if (dqn['F'][-1]+dqn['C'][-1]<opt['F'][-1]+opt['C'][-1]) or (ppo['F'][-1]+ppo['C'][-1]<opt['F'][-1]+opt['C'][-1]):
            total.append(idx)
        
    print(len(total))
        
        

def dist_maxmin():
    dqn=np.load('./DQN/Results/'+str(0)+'.npy',allow_pickle=True).tolist()
    data=np.array(dqn['state'])
    print(data.shape)
    
    
    for idx in range(1,1000):
        dqn=np.load('./DQN/Results/'+str(idx)+'.npy',allow_pickle=True).tolist()
        ppo=np.load('./PPO/Results/'+str(idx)+'.npy',allow_pickle=True).tolist()
        hc=np.load('./HC/Results/HC'+str(idx)+'.npy',allow_pickle=True).tolist()
        opt=np.load('./OPT/Results/'+str(idx)+'.npy',allow_pickle=True).tolist()
        
        
        data=np.concatenate((data,np.array(dqn['state']),np.array(ppo['state']),np.array(hc['state']),np.array(opt['state'])),axis=0)
    
    max_state,min_state=np.max(data,axis=0),np.min(data,axis=0)
    return max_state,min_state
        
      

config = yaml.load(open("chaohu.yaml"), yaml.FullLoader)
#根据得到的决策树，把几个筛选条件节点名字给出，便于画图

for i in range(len(config['states'])):
    print(i,config['states'][i])


#figfc()
#dist()
#max_state,min_state=dist_maxmin()
#print(max_state)
#print(min_state)

'''
[2.00643771e+00 4.23765898e+00 3.25000000e+00 3.35000000e+00
 2.00000000e+00 2.21000000e+00 1.96000000e+00 2.05000000e+00
 1.29008140e+03 1.38056153e+03 8.05242213e+02 2.19059766e+03
 6.83000000e+02 6.83000000e+02 1.00000000e+03 1.00000000e+03
 1.57499394e+03 2.79013935e+02]
[ 4.98391697e-01  9.70742428e-01  1.37786022e-01  0.00000000e+00
  4.07444591e-02  0.00000000e+00  2.13572798e-04  4.85072906e-03
  1.08618908e+01  0.00000000e+00 -8.00368399e+02  9.27032212e-02
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  1.11917992e+00  0.00000000e+00]

'''