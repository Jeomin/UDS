# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# %%
import numpy as np
import SWMM_ENV

env_params={
        'orf':'chaohu',
        'advance_seconds':300
    }
env=SWMM_ENV.SWMM_ENV(env_params)

# %%
import DQN


agent_params={
    'state_dim':len(env.config['states']),
    'action_dim':2**len(env.config['action_assets']),
    'evalnet_layer_A':3,
    'evalnet_A':[{'num':30},{'num':30},{'num':30}],
    'evalnet_layer_V':3,
    'evalnet_V':[{'num':30},{'num':30},{'num':30}],
    'targetnet_layer_A':3,
    'targetnet_A':[{'num':30},{'num':30},{'num':30}],
    'targetnet_layer_V':3,
    'targetnet_V':[{'num':30},{'num':30},{'num':30}],
    'num_rain':300,
    
    'training_step':20,
    'gamma':0.3,
    'epsilon':0.1,
    'ep_min':0.01,
    'ep_decay':0.1
}
agent = DQN.DQN(agent_params,env)
agent.load_model()

# %% [markdown]
# # Apply SHAP

# %%
import shap
shap.initjs()

# %%
def Getdata_pd():
    
    config = yaml.load(open("chaohu.yaml"), yaml.FullLoader)
    title=[]
    for it in config['states']:
        title.append(it[0])
    title.append('action')

    
    dqn_data=np.load('./Results/0.npy',allow_pickle=True).tolist()
    dqn_s=np.array(dqn_data['state'])[1:,:]
    
    for idx in range(5):
        dqn_data=np.load('./Results/'+str(idx)+'.npy',allow_pickle=True).tolist()
        dqn_s=np.concatenate((dqn_s,np.array(dqn_data['state'])[1:,:]),axis=0)
    
    return dqn_s

# %%
dqn_s=Getdata_pd()

# %%
explainer = shap.DeepExplainer(agent.model,dqn_s)

# %%
shap_values = explainer.shap_values(dqn_s)

# %%
len(shap_values)

# %%
shap.force_plot(explainer.expected_value[1], shap_values[1], dqn_s)

# %%
shap.summary_plot(shap_values, dqn_s)

# %%



