# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 11:22:28 2023

@author: chongtm
"""
def train(model,params,memory_buffer):   
    #sampling and upgrading
    
    model.compile(loss='mse', optimizer=Adam(1e-3))
    memory_buffer = deque(maxlen=1000)
    for j in range(params['training_step']):
        history['episode'].append(j)
        count = 0
        Q_value, Q_loss = 0, 0
        
        # 更新target_model
        model.target_model.set_weights(model.model.get_weights())
        
        history['Episode_reward'].append(Q_value)
        history['Loss'].append(Q_loss)
        print('Episode: {} | Episode reward: {} | loss: {:.3f} | e:{:.2f}'.format(j, Q_value, Q_loss, params['epsilon']))              
        memory_buffer = deque(maxlen=1000)        
        model.model.save_weights('./model/dqn.h5')
        if j % 30 == 0:
            np.save('./Results/Train.npy',history)
    return history





def test(env,model,rain):
    # simulation on given rainfall
    test_history = {'time':[] ,'state': [], 'action': [], 'reward': [], 'F':[], 'C':[]}
    s = env.reset(rain)
    done, t= False, 0
    test_history['time'].append(t)
    test_history['state'].append(s)
    while not done:
        a = model.choose_action(s,False)
        action = model.action_table[a,:].tolist()
        snext,reward,F,C,done = env.step(action)
        s = snext
        t +=1
        
        test_history['time'].append(t)
        test_history['state'].append(s)
        test_history['action'].append(action)
        test_history['reward'].append(reward)
        test_history['F'].append(F)
        test_history['C'].append(C)

    return test_history




history = agent.train(raindata[0:])
np.save('./Results/Train.npy',history)

raindata = np.load('test_raindata.npy').tolist()
agent.load_model()
for i in range(len(raindata)):
    test_his = agent.test(raindata[i])
    np.save('./Results/'+str(i)+'.npy',test_his)
