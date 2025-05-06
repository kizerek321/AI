import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import sailor_funct as sf

number_of_episodes = 4000  # number of training epizodes DO NOT INCREASE!
gamma = 1.0  # discount factor

# Training data parameters - these may be time-dependent functions:
alpha = 0.03  # training speed factor
epsilon = 0.2  # exploration factor
T0 = 1.0  # initial temperature
T_min = 0.015
decay_rate = 0.00005
# another exploration method e.g.softmax

# file_name = 'map_small.txt'
#file_name = 'map_easy.txt'
# file_name = 'map_simple.txt'
# file_name = 'map_middle.txt'
# file_name = 'map_mid.txt'
file_name = 'map_big.txt'
# file_name = 'map_spiral.txt'

reward_map = sf.load_data(file_name)
num_of_rows, num_of_columns = reward_map.shape

num_of_steps_max = int(2.5 * (num_of_rows + num_of_columns))  # maximum number of steps in an episode
Q = np.zeros([num_of_rows, num_of_columns, 4], dtype=float)  # trained usability table of <state,action> pairs
sum_of_rewards = np.zeros([number_of_episodes], dtype=float)

for episode in range(number_of_episodes):
    T = max(0.01, 1.0 * np.exp(-0.001 * episode))
    state = np.zeros([2], dtype=int)  # initial state here [1 1] but rather random due to exploration
    # state = ....................................

    # print('initial state = ' + str(state) )
    the_end = False
    nr_pos = 0
    # reward_map_curr = reward_map
    while not the_end:
        nr_pos += 1  # move number

        # ε-greedy action selection
        # if np.random.rand() < epsilon:
        #    action = np.random.randint(1, 5)  # Random action (1-right, 2-up, 3-left, 4-down)
        # else:
        #    action = 1 + np.argmax(Q[state[0], state[1], :])  # Best action (greedy)
        T = max(T_min, T0 * np.exp(-decay_rate * episode))  # exponential decay of T

        q_values = Q[state[0], state[1], :]
        exp_q = np.exp(q_values / T)
        probabilities = exp_q / np.sum(exp_q)
        action = 1 + np.random.choice(4, p=probabilities)

        state_next, reward = sf.environment(state, action, reward_map)

        # State-action usability modifcication:
        best_next_action = np.argmax(Q[state_next[0], state_next[1], :])
        Q[state[0], state[1], action - 1] = (1 - alpha) * Q[state[0], state[1], action - 1] + \
                                            alpha * (reward + gamma * Q[state_next[0], state_next[1], best_next_action])

        # print('state = ' + str(state) + ' action = ' + str(action) +  ' -> next state = ' + str(state_next) + ' reward = ' + str(reward))

        state = state_next  # going to the next state

        # end of episode if maximum number of steps is reached or last column
        # is reached
        if (nr_pos == num_of_steps_max) or (state[1] >= num_of_columns - 1):
            the_end = True
        alpha = max(0.02, 0.1 * (1 - episode / number_of_episodes))
        sum_of_rewards[episode] += reward
    if episode % 500 == 0:
        print('episode = ' + str(episode) + ' average sum of rewards = ' + str(np.mean(sum_of_rewards)))
# print('average sum of rewards = ' + str(np.mean(sum_of_rewards)))

sf.sailor_test(reward_map, Q, 1000)
sf.draw(reward_map, Q, file_name)
