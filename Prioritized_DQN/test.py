import gym
from Prioritized_DQN import DQNAgent
import matplotlib.pyplot as plt
from collections import deque
import numpy as np




max_episode = 5000
eval_max_episode = 200
max_steps = 1000
eps_end=0.01
eps_decay=0.995
record_freq = 20

beta_start = 0.4
beta_frames = 1000 
beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

if __name__ == '__main__':
    # env = gym.make('LunarLander-v2')
    env = gym.make('CartPole-v0')
    env.seed(0)
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)

    dqn = DQNAgent(
            state_size=n_states,
            action_size=n_actions,
            seed=0,
            buffer_size=int(1e5),
            batch_size=64,
            lr=0.01,
            gamma=0.99,
            tau=1e-3,
            epsilon=1,
            update_freq=10,
            beta = 0.4
        )

    train_curve = []
    train_mean_score = deque(maxlen=record_freq) 

    for episode in range(max_episode):
        obs = env.reset()
        episode_r = 0
        for t in range(max_steps):
            # env.render()
            action = dqn.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            dqn.step(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_r += reward
            if done:
                break
        train_mean_score.append(episode_r)
        if episode%record_freq == 0:
            train_curve.append(np.mean(train_mean_score))
            print('\rEpisode {}\tAverage reward: {:.2f}'.format(episode, np.mean(train_mean_score)),'epsilon ',dqn.epsilon)
        dqn.epsilon = max(eps_end, eps_decay*dqn.epsilon) # decrease epsilon
        dqn.beta =  beta_by_frame(episode)



    plt.plot(train_curve, linewidth=1, label='DQN_train')
    # plt.savefig("test_dqn.png")
    # plt.show()
    # plt.clf()
    #################################
    # eval
    print('#########################')
    print('start eval')
    dqn.epsilon = 0
    eval_curve = []
    for episode in range(eval_max_episode):
        obs = env.reset()
        episode_r = 0
        for t in range(max_steps):
            # env.render()
            action = dqn.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            episode_r += reward
            if done:
                break
        eval_curve.append(episode_r)
        print('Eval Ep: ', episode,'| Ep_r: ', round(episode_r, 2),'epsilon ',dqn.epsilon)

    plt.plot(eval_curve, linewidth=1, label='DQN_eval')
    plt.legend()
    plt.savefig("result_dqn.png")