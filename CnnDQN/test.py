# -*- encoding: utf-8 -*-
'''
File    :   test_CatchPigs.py
Time    :   2020/08/26 12:12:48
Author  :   Chao Wang 
Version :   1.0
Contact :   374494067@qq.com
@Desc    :   None
'''



import gym
from CnnDQN_catchpigs import DQNAgent
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
from envs.env_SingleCatchPigs.env_SingleCatchPigs import EnvSingleCatchPigs
from envs.env_CatchPigs.env_CatchPigs import EnvCatchPigs
from tqdm import *

max_episode = 2000
eval_max_episode = 500
max_steps = 50
eps_end=0.01
eps_decay=0.995
record_freq = 20
n_actions = 5
if __name__ == '__main__':

    # env = EnvSingleCatchPigs(7)
    env = EnvCatchPigs(7,False)

    dqn = DQNAgent(
            # state_size=n_states,
            action_size=n_actions,
            seed=0,
            buffer_size=int(1e5),
            batch_size=64,
            lr=0.01,
            gamma=0.99,
            tau=1e-3,
            epsilon=1,
            update_freq=4
        )


    train_curve1 = []
    train_curve2 = []
    train_mean_score1 = deque(maxlen=record_freq) 
    train_mean_score2 = deque(maxlen=record_freq) 

    # for episode in tqdm(range(max_episode)):
    for episode in range(max_episode):
        env.reset()
        episode_r1 = 0
        episode_r2 = 0
        for t in range(max_steps):
            # env.render()
            obs1,obs2 = env.get_obs()
            action1 = dqn.choose_action(obs1)
            action2 = dqn.choose_action(obs2)

            [reward_1, reward_2], done = env.step([action1, action2])
            next_obs1,next_obs2 = env.get_obs()

            dqn.step(obs1, action1, reward_1, next_obs1, done)
            dqn.step(obs2, action2, reward_2, next_obs2, done)

            obs1 = next_obs1
            obs2 = next_obs2

            episode_r1 += reward_1
            episode_r2 += reward_2

            if done:
                break
        train_mean_score1.append(episode_r1)
        train_mean_score2.append(episode_r2)
        if episode%record_freq == 0:
            train_curve1.append(np.mean(train_mean_score1))
            train_curve2.append(np.mean(train_mean_score2))
            print('\rEpisode {}\tAverage reward_1: {:.2f}\tAverage reward_2: {:.2f}'.format(episode, np.mean(train_mean_score1),np.mean(train_mean_score2)),'   epsilon ',dqn.epsilon)
        dqn.epsilon = max(eps_end, eps_decay*dqn.epsilon) # decrease epsilon

    print('save success!')

    plt.plot(train_curve1, linewidth=1, label='DQN_agent1_train')
    plt.plot(train_curve2, linewidth=1, label='DQN_agent2_train')
    plt.savefig("test_dqn_catchpigs.png")
    # plt.show()
    #################################
    # eval
    plt.clf()
    print('#########################')
    print('start eval')
    print('load success!')

    dqn.epsilon = 0
    eval_curve1 = []
    eval_curve2 = []
    for episode in tqdm(range(eval_max_episode)):
        obs = env.reset()
        episode_r1 = 0
        episode_r2 = 0
        for t in range(max_steps):
            obs1,obs2 = env.get_obs()
            action1 = dqn.choose_action(obs1)
            action2 = dqn.choose_action(obs2)

            [reward_1, reward_2], done = env.step([action1, action2])
            next_obs1,next_obs2 = env.get_obs()

            dqn.step(obs1, action1, reward_1, next_obs1, done)
            dqn.step(obs2, action2, reward_2, next_obs2, done)

            obs1 = next_obs1
            obs2 = next_obs2

            episode_r1 += reward_1
            episode_r2 += reward_2

            if done:
                break
        eval_curve1.append(episode_r1)
        eval_curve2.append(episode_r2)

    plt.plot(eval_curve1, linewidth=1, label='DQN_agent1_eval')
    plt.plot(eval_curve2, linewidth=1, label='DQN_agent2_eval')
    plt.legend()
    plt.savefig("test_dqn_catchpigs_eval_1.png")


    '''
    test save and load model
    '''
   # eval
    plt.clf()
    print('#########################')
    print('start save and load eval')
    dqn.save('ckpt/dqn.ph')
    dqn.load('ckpt/dqn.ph')
    print('load success!')

    dqn.epsilon = 0
    eval_curve1 = []
    eval_curve2 = []
    for episode in tqdm(range(eval_max_episode)):
        obs = env.reset()
        episode_r1 = 0
        episode_r2 = 0
        for t in range(max_steps):
            obs1,obs2 = env.get_obs()
            action1 = dqn.choose_action(obs1)
            action2 = dqn.choose_action(obs2)

            [reward_1, reward_2], done = env.step([action1, action2])
            next_obs1,next_obs2 = env.get_obs()

            dqn.step(obs1, action1, reward_1, next_obs1, done)
            dqn.step(obs2, action2, reward_2, next_obs2, done)

            obs1 = next_obs1
            obs2 = next_obs2

            episode_r1 += reward_1
            episode_r2 += reward_2

            if done:
                break
        eval_curve1.append(episode_r1)
        eval_curve2.append(episode_r2)

    plt.plot(eval_curve1, linewidth=1, label='DQN_agent1_eval')
    plt.plot(eval_curve2, linewidth=1, label='DQN_agent2_eval')
    plt.legend()
    plt.savefig("test_dqn_catchpigs_eval_save&load.png")