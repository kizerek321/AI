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
import numpy as np
import matplotlib.pyplot as plt
import sailor_funct as sf

# Parameters
number_of_sumulations = 1000
number_of_individuals = 20
number_of_episodes_for_eval = 50

p_cross = 0.9
p_mut = 0.2  # stronger mutation
if_elitism = True
elite_fraction = 0.15
selection_pressure = 4.0

#file_name = 'map_small.txt'
file_name = 'map_easy.txt'
#file_name = 'map_mid.txt'
#file_name = 'map_big.txt'
#file_name = 'map_spiral.txt'

gamma = 1.05

number_of_epochs = number_of_sumulations // (number_of_individuals * number_of_episodes_for_eval)
reward_map = sf.load_data(file_name)
num_of_rows, num_of_columns = reward_map.shape
num_of_steps_max = int(2.5 * (num_of_rows + num_of_columns))

# Initialize population: prefer going right and down
Popul = np.random.choice([1, 2,3, 4], size=(number_of_individuals, num_of_rows, num_of_columns), p=[0.4, 0.2,0.1, 0.3])


def reproduction(Popul, fitnesses):
    fit_cum = np.cumsum(fitnesses)
    max_cum_value = fit_cum[-1]
    Popul_new = np.copy(Popul)
    for i in range(len(Popul)):
        rand_value = np.random.random() * max_cum_value
        parent_index = np.searchsorted(fit_cum, rand_value)
        Popul_new[i] = np.copy(Popul[parent_index])
    return Popul_new


def mutate(strategy):
    strategy = np.copy(strategy)
    for _ in range(5):  # mutate multiple cells
        if np.random.rand() < p_mut:
            i, j = np.random.randint(strategy.shape[0]), np.random.randint(strategy.shape[1])
            strategy[i, j] = np.random.choice([1, 2, 4], p=[0.5, 0.2, 0.3])
    return strategy


def crossover(parent1, parent2):
    if np.random.rand() < p_cross:
        row = np.random.randint(parent1.shape[0])
        child1 = np.vstack((parent1[:row], parent2[row:]))
        child2 = np.vstack((parent2[:row], parent1[row:]))
        return child1, child2
    return np.copy(parent1), np.copy(parent2)


# Evolution
maximum_mean_sum_of_rewards = -1000000000

for epoch in range(number_of_epochs):
    mean_sums_of_rewards = np.zeros(number_of_individuals)
    for i in range(number_of_individuals):
        mean_sums_of_rewards[i] = sf.sailor_test(reward_map, Popul[i], number_of_episodes_for_eval, gamma)

    # Track best strategy
    best_index = np.argmax(mean_sums_of_rewards)
    if mean_sums_of_rewards[best_index] > maximum_mean_sum_of_rewards:
        maximum_mean_sum_of_rewards = mean_sums_of_rewards[best_index]
        best_strategy = np.copy(Popul[best_index])
        print(f'New best individual = {maximum_mean_sum_of_rewards:.2f} at epoch {epoch}')

    # Fitness: higher rewards mean higher fitness
    fitnesses = mean_sums_of_rewards - np.min(mean_sums_of_rewards) + 1e-6
    fitnesses = fitnesses ** selection_pressure
    fitnesses /= np.sum(fitnesses)

    # Elitism
    if if_elitism:
        num_elites = max(1, int(elite_fraction * number_of_individuals))
        elite_indices = np.argsort(mean_sums_of_rewards)[-num_elites:]
        elite_individuals = Popul[elite_indices]

    # Reproduction
    new_population = reproduction(Popul, fitnesses)

    # Crossover + Mutation
    offspring = []
    for i in range(0, len(new_population), 2):
        p1 = new_population[i]
        p2 = new_population[(i + 1) % len(new_population)]
        c1, c2 = crossover(p1, p2)
        offspring.append(mutate(c1))
        offspring.append(mutate(c2))

    Popul = np.array(offspring[:number_of_individuals])

    # Add elites
    if if_elitism:
        worst_indices = np.argsort(mean_sums_of_rewards)[:num_elites]
        for i in range(num_elites):
            Popul[worst_indices[i]] = np.copy(elite_individuals[i % num_elites])

    if epoch % 10 == 0:
        print(f'Epoch {epoch}: average reward = {np.mean(mean_sums_of_rewards):.2f}')

# Final evaluation
mean_sum_of_rewards = sf.sailor_test(reward_map, best_strategy, 1000, gamma)
print('Final average reward of best strategy:', mean_sum_of_rewards)
sf.draw(reward_map, best_strategy, mean_sum_of_rewards)
print('Finished')

