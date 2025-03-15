# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 20:39:01 2022

@author: chong
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    
    raindata = np.load('./DQN/test_raindata.npy').tolist()
    
    figs,axes=plt.subplots(6,3,figsize=(10,60))
    #axes.flatten()
    
    font1={'family':'Times New Roman',
           'size':18}
    font2={'family':'Times New Roman',
           'size':10}
    
    for idx in range(6):
        rain,x,k=[],[],0
        for it in range(int(len(raindata[idx])/2)):
            if np.mod(it,5)==0:
                rain.append(0)
                x.append(k)
                k+=1
        
        
        for it in range(len(raindata[idx])):
            if np.mod(it,5)==0:
                rain.append(np.float64(raindata[idx][it][1]))
                x.append(k)
                k+=1
        
        
        
        dqn=np.load('./DQN/Results/'+str(idx)+'.npy',allow_pickle=True).tolist()
        ppo=np.load('./PPO/Results/'+str(idx)+'.npy',allow_pickle=True).tolist()
        hc=np.load('./HC/Results/HC'+str(idx)+'.npy',allow_pickle=True).tolist()
        #bl=np.load('./Baseline/Results/'+str(idx)+'.npy',allow_pickle=True).tolist()
        opt=np.load('./OPT/Results/'+str(idx)+'.npy',allow_pickle=True).tolist()
        
        
        top=np.max(np.array(hc['C'])/1000+np.array(hc['F'])/1000)
        if idx==0:
            axes[idx,0].plot(np.array(hc['F'])/1000,label='HC',linewidth=0.8)
            axes[idx,0].plot(np.array(dqn['F'])/1000,label='DQN',linewidth=0.8)
            axes[idx,0].plot(np.array(ppo['F'])/1000,label='PPO',linewidth=0.8)
            axes[idx,0].plot(np.array(opt['F'])/1000,label='OPT',linewidth=0.8)
            axes[idx,0].set_ylim(0,top*1.5)
            axes[idx,0].set_xticks([])
            axes[idx,0].set_title('Rain'+str(idx)+' flooding',font=font2)
            axes[idx,0].legend(loc='upper right',fontsize=6)
            ax2=axes[idx,0].twinx()
            ax2.set_yticks([])
            ax2.invert_yaxis()
            ax2.bar(x,rain,width=0.5)
            
            axes[idx,1].plot(np.array(hc['C'])/1000,label='HC',linewidth=0.8)
            axes[idx,1].plot(np.array(dqn['C'])/1000,label='DQN',linewidth=0.8)
            axes[idx,1].plot(np.array(ppo['C'])/1000,label='PPO',linewidth=0.8)
            axes[idx,1].plot(np.array(opt['C'])/1000,label='OPT',linewidth=0.8)
            axes[idx,1].set_ylim(0,top*1.5)
            axes[idx,1].set_xticks([])
            axes[idx,1].set_yticks([])
            axes[idx,1].set_title('Rain'+str(idx)+' CSO',font=font2)
            axes[idx,1].legend(loc='upper right',fontsize=6)
            ax2=axes[idx,1].twinx()
            ax2.set_yticks([])
            ax2.invert_yaxis()
            ax2.bar(x,rain,width=0.5)
            
            axes[idx,2].plot(np.array(hc['C'])/1000+np.array(hc['F'])/1000,label='HC',linewidth=0.8)
            axes[idx,2].plot(np.array(dqn['C'])/1000+np.array(dqn['F'])/1000,label='DQN',linewidth=0.8)
            axes[idx,2].plot(np.array(ppo['C'])/1000+np.array(ppo['F'])/1000,label='PPO',linewidth=0.8)
            axes[idx,2].plot(np.array(opt['C'])/1000+np.array(opt['F'])/1000,label='OPT',linewidth=0.8)
            axes[idx,2].set_ylim(0,top*1.5)
            axes[idx,2].set_xticks([])
            axes[idx,2].set_yticks([])
            axes[idx,2].set_title('Rain'+str(idx)+' floodin+CSO',font=font2)
            axes[idx,2].legend(loc='lower right',fontsize=6)
            ax2=axes[idx,2].twinx()
            ax2.invert_yaxis()
            ax2.bar(x,rain,width=0.5)
                
                
        elif idx==5:
            axes[idx,0].plot(np.array(hc['F'])/1000,label='HC',linewidth=0.8)
            axes[idx,0].plot(np.array(dqn['F'])/1000,label='DQN',linewidth=0.8)
            axes[idx,0].plot(np.array(ppo['F'])/1000,label='PPO',linewidth=0.8)
            axes[idx,0].plot(np.array(opt['F'])/1000,label='OPT',linewidth=0.8)
            axes[idx,0].set_ylim(0,top*1.5)
            axes[idx,0].set_xticks([0,np.array(dqn['F']).shape[0]],['0','480'])
            axes[idx,0].set_title('Rain'+str(idx)+' flooding',font=font2)
            axes[idx,0].legend(loc='upper right',fontsize=6)
            ax2=axes[idx,0].twinx()
            ax2.set_yticks([])
            ax2.invert_yaxis()
            ax2.bar(x,rain,width=0.5)
            
            axes[idx,1].plot(np.array(hc['C'])/1000,label='HC',linewidth=0.8)
            axes[idx,1].plot(np.array(dqn['C'])/1000,label='DQN',linewidth=0.8)
            axes[idx,1].plot(np.array(ppo['C'])/1000,label='PPO',linewidth=0.8)
            axes[idx,1].plot(np.array(opt['C'])/1000,label='OPT',linewidth=0.8)
            axes[idx,1].set_ylim(0,top*1.5)
            axes[idx,1].set_xticks([0,np.array(dqn['C']).shape[0]],['0','480'])
            axes[idx,1].set_title('Rain'+str(idx)+' CSO',font=font2)
            axes[idx,1].set_yticks([])
            axes[idx,1].legend(loc='upper right',fontsize=6)
            ax2=axes[idx,1].twinx()
            ax2.set_yticks([])
            ax2.invert_yaxis()
            ax2.bar(x,rain,width=0.5)
            
            axes[idx,2].plot(np.array(hc['C'])/1000+np.array(hc['F'])/1000,label='HC',linewidth=0.8)
            axes[idx,2].plot(np.array(dqn['C'])/1000+np.array(dqn['F'])/1000,label='DQN',linewidth=0.8)
            axes[idx,2].plot(np.array(ppo['C'])/1000+np.array(ppo['F'])/1000,label='PPO',linewidth=0.8)
            axes[idx,2].plot(np.array(opt['C'])/1000+np.array(opt['F'])/1000,label='OPT',linewidth=0.8)
            axes[idx,2].set_ylim(0,top*1.5)
            axes[idx,2].set_xticks([0,np.array(dqn['F']).shape[0]],['0','480'])
            axes[idx,2].set_title('Rain'+str(idx)+' floodin+CSO',font=font2)
            axes[idx,2].set_yticks([])
            axes[idx,2].legend(loc='lower right',fontsize=6)
            ax2=axes[idx,2].twinx()
            ax2.invert_yaxis()
            ax2.bar(x,rain,width=0.5)
            
        else:
            axes[idx,0].plot(np.array(hc['F'])/1000,label='HC',linewidth=0.8)
            axes[idx,0].plot(np.array(dqn['F'])/1000,label='DQN',linewidth=0.8)
            axes[idx,0].plot(np.array(ppo['F'])/1000,label='PPO',linewidth=0.8)
            axes[idx,0].plot(np.array(opt['F'])/1000,label='OPT',linewidth=0.8)
            axes[idx,0].set_title('Rain'+str(idx)+' flooding',font=font2)
            axes[idx,0].set_ylim(0,top*1.5)
            axes[idx,0].set_xticks([])
            axes[idx,0].legend(loc='upper right',fontsize=6)
            ax2=axes[idx,0].twinx()
            ax2.set_yticks([])
            ax2.invert_yaxis()
            ax2.bar(x,rain,width=0.5)
            
            axes[idx,1].plot(np.array(hc['C'])/1000,label='HC',linewidth=0.8)
            axes[idx,1].plot(np.array(dqn['C'])/1000,label='DQN',linewidth=0.8)
            axes[idx,1].plot(np.array(ppo['C'])/1000,label='PPO',linewidth=0.8)
            axes[idx,1].plot(np.array(opt['C'])/1000,label='OPT',linewidth=0.8)
            axes[idx,1].set_title('Rain'+str(idx)+' CSO',font=font2)
            axes[idx,1].set_ylim(0,top*1.5)
            axes[idx,1].set_xticks([])
            axes[idx,1].set_yticks([])
            axes[idx,1].legend(loc='upper right',fontsize=6)
            ax2=axes[idx,1].twinx()
            ax2.set_yticks([])
            ax2.invert_yaxis()
            ax2.bar(x,rain,width=0.5)
            
            axes[idx,2].plot(np.array(hc['C'])/1000+np.array(hc['F'])/1000,label='HC',linewidth=0.8)
            axes[idx,2].plot(np.array(dqn['C'])/1000+np.array(dqn['F'])/1000,label='DQN',linewidth=0.8)
            axes[idx,2].plot(np.array(ppo['C'])/1000+np.array(ppo['F'])/1000,label='PPO',linewidth=0.8)
            axes[idx,2].plot(np.array(opt['C'])/1000+np.array(opt['F'])/1000,label='OPT',linewidth=0.8)
            axes[idx,2].set_title('Rain'+str(idx)+' floodin+CSO',font=font2)
            axes[idx,2].set_ylim(0,top*1.5)
            axes[idx,2].set_xticks([])
            axes[idx,2].set_yticks([])
            axes[idx,2].legend(loc='lower right',fontsize=6)
            ax2=axes[idx,2].twinx()
            ax2.invert_yaxis()
            ax2.bar(x,rain,width=0.5)
        
    
    figs.text(0.06,0.34,'Volumes of Flooding and CSO (10$^{3}$ m$^{3}$)',rotation=90,font=font1)
    figs.text(0.94,0.4,'Rain intensity (mm/hour)',rotation=-90,font=font1)
    figs.text(0.43,0.06,'Time (minutes)',font=font1)
    #figs.legend(fontsize=15,ncol=4,loc = (0.3,0.01))
        
        
    figs.savefig('./fc_results.tif',dpi=600)


def results1():
    r=[]
    for idx in range(6):
        dqn=np.load('./DQN/Results/'+str(idx)+'.npy',allow_pickle=True).tolist()
        ppo=np.load('./PPO/Results/'+str(idx)+'.npy',allow_pickle=True).tolist()
        hc=np.load('./HC/Results/HC'+str(idx)+'.npy',allow_pickle=True).tolist()
        opt=np.load('./OPT/Results/'+str(idx)+'.npy',allow_pickle=True).tolist()
        bl=np.load('./Baseline/Results/'+str(idx)+'.npy',allow_pickle=True).tolist()
        
        maxred=(opt['C'][-1]+opt['F'][-1])/1000
        bl=(bl['C'][-1]+bl['F'][-1])/1000
        
        tem1=[hc['C'][-1]/1000, hc['F'][-1]/1000, (hc['C'][-1]+hc['F'][-1])/1000, '-']
        tem2=[dqn['C'][-1]/1000, dqn['F'][-1]/1000, (dqn['C'][-1]+dqn['F'][-1])/1000,(bl-((dqn['C'][-1]+dqn['F'][-1])/1000))/(bl-maxred)]
        tem3=[ppo['C'][-1]/1000, ppo['F'][-1]/1000, (ppo['C'][-1]+ppo['F'][-1])/1000,(bl-((ppo['C'][-1]+ppo['F'][-1])/1000))/(bl-maxred)]
        tem4=[opt['C'][-1]/1000, opt['F'][-1]/1000, (opt['C'][-1]+opt['F'][-1])/1000,'-']
             
        r.append(tem1)
        r.append(tem2)
        r.append(tem3)
        r.append(tem4)
    pd.DataFrame(r).to_csv('results_FC.csv')
    
        


figfc()
#results1()