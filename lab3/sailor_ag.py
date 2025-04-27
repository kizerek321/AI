"""
    Genetic algoritm for sailor problem. 
    Sailor sails from left to right side of the map bypassing dangerous places (with negative rewards).
    The environment is nondeterministic due to the wind and waves which means that the boat moves to chosen
    places only with constant probabilities. If you want to evaluate sailor's strategy in the reliable way,
    you need to use many episodes for strategy evaluation.
    
    In this task you need to invent crossover and mutation operations, invent the evaluation method, choose   
    the best parameters values of the evolution, maybe add more parameters and operations. Your solutions 
    should work for different maps and different gamma values.
"""
import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import sailor_funct as sf


def reproduction(Popul,fitnesses):      # new population based on fittness values
   fit_cum = np.copy(fitnesses)
   for i in range(fitnesses.size-1):    # cumulative sum
      fit_cum[i+1] += fit_cum[i]

   max_cum_value = fit_cum[fitnesses.size-1]
   Popul_new = np.copy(Popul)

   for i in range(fitnesses.size):
      rand_value = np.random.random()*max_cum_value
      for j in range(fit_cum.size):
         prev_val = 0
         if j>0:
            prev_val = fit_cum[j-1]
         if (rand_value > prev_val) & (rand_value <= fit_cum[j]):
            parent_index = j
            break
      Popul_new[i] = np.copy(Popul[j])
   return Popul_new
# end of reproduction

# Action identifiers (1 - right, 2 - up, 3 - left, 4 - bottom):
def mutate(strategy):
    if np.random.rand() < p_mut:
        idx = np.random.randint(len(strategy))
        strategy[idx] = np.random.choice([2, 4, 3, 1])  # Change a random move
    return strategy

def crossover(parent1, parent2):
    if np.random.rand() < p_cross:
        point = np.random.randint(1, len(parent1) - 1)  # Random split point
        return np.concatenate((parent1[:point], parent2[point:])), \
               np.concatenate((parent2[:point], parent1[point:]))
    return parent1, parent2  # No crossover


# Evolution parameters ..... (find the best values)
number_of_sumulations = 1000                 # do not change (time complexity parameter)
number_of_individuals = 20                 # number of individuals in population (each individual conatains sailor strategy)
number_of_episodes_for_eval = 10            # number of epizodes for strategy evaluation


p_cross = 0.8                              # crossover probability
p_mut = 0.1                                # mutation probability
if_elitism = True                           # the best individual goes to the next population unchanged
selection_pressure = 1.5                   # if higher -> more copies of the best individuals in new population expense of worst individuals


# Task definition parameters ............ (begin from easy one, and try mid one later)
#file_name = 'map_small.txt' 
file_name = 'map_easy.txt'  
#file_name = 'map_mid.txt'
#file_name = 'map_big.txt'
#file_name = 'map_spiral.txt'
gamma = 1.0                                 # discount factor (part of a task). If gamma < 1, rewards with longer time distance are less important
                                            # (it pays an agent to get positive rewards as soon as possible and penalties as long as possible)

# number fo epochs of evolution with time complexity preservation:             
number_of_epochs = number_of_sumulations //  (number_of_individuals*number_of_episodes_for_eval)            
reward_map = sf.load_data(file_name)                 # load map of rewards from file
num_of_rows, num_of_columns = reward_map.shape

num_of_steps_max = int(2.5*(num_of_rows + num_of_columns))    # maximum number of steps in an episode
Popul = np.random.randint(1,5,(number_of_individuals, num_of_rows, num_of_columns))  # population of strategies

print('Initial population = ' + str(Popul))

maximum_mean_sum_of_rewards = -1000000000


for epoch in range(number_of_epochs):
    # evaluation of individuals:
    mean_sums_of_rewards = np.zeros([number_of_individuals])
    minimum_mean_sum_of_rewords = 1000000000
    fitnesses = np.zeros([number_of_individuals])
    for individual in range(number_of_individuals):
        mean_sums_of_rewards[individual] = sf.sailor_test(reward_map, Popul[individual],number_of_episodes_for_eval,gamma)
        if minimum_mean_sum_of_rewords > mean_sums_of_rewards[individual]:
            minimum_mean_sum_of_rewords = mean_sums_of_rewards[individual]
        if maximum_mean_sum_of_rewards < mean_sums_of_rewards[individual]:
            maximum_mean_sum_of_rewards = mean_sums_of_rewards[individual]
            best_strategy = Popul[individual]
            print('new best individual = ' + str(maximum_mean_sum_of_rewards) + ' in epoch ' + str(epoch))

    # Fittness values must be >= 0 and higher as individual better. You can use selection_factor as an exponent of
    # mean_sums_of_rewards to adjust selection pressure - expected number of copies of the best individuals against the
    # existence of the worst.
    for individual in range(number_of_individuals):
        ranks = np.argsort(mean_sums_of_rewards)  # Rank individuals
        fitnesses = np.linspace(1, selection_pressure, len(ranks))  # Assign probabilities
        # selection pressure can be used ....
        # rank reproduction can be used ...

    print('epoch = ' + str(epoch) + ' avarage sum of rewards over population = ' + str(np.mean(mean_sums_of_rewards)))

    # Reproduction ...... can be rank or roulette version
    Popul = reproduction(Popul,fitnesses)    # roulette version
   
    # Crossover .......
    new_population = []
    for i in range(0, number_of_individuals, 2):  # Pair up individuals
        parent1, parent2 = Popul[i], Popul[i + 1]  # Select parents
        child1, child2 = crossover(parent1, parent2)  # Apply crossover
        new_population.extend([child1, child2])  # Add to new population

    Popul = new_population  # Replace old population with new one

    # Mutation .......
    for i in range(number_of_individuals):
        Popul[i] = mutate(Popul[i])  # Mutate each individual

    # Other operations (elitism, niches, parameter changing functions etc.) ...

# end of evolution loop

mean_sum_of_rewards = sf.sailor_test(reward_map, best_strategy, 1000, gamma)
print('Average sum of rewards for best strategy = ' + str(mean_sum_of_rewards))
sf.draw(reward_map,best_strategy,mean_sum_of_rewards)
print('Final population = ' + str(Popul))
