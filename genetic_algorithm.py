from bisect import bisect_left
import numpy as np
import sys
import time
import os
import matplotlib.pyplot as plt

path = ''
os.chdir(path)

# importing a model built by a colleague from TenorCell project for fitness calculation
from variational_model import VariationalKerasModelWithScalers
model = VariationalKerasModelWithScalers.get_from_experiment_name(path+'/test1')


# Data representation - converting intiger to binary vector and vice versa
def int_to_bin(Population, bin_length):
    length = len(Population[: , 1])
    width = len(Population[1, :])
    PopulationBin = np.empty([length, width*bin_length])
    for i in range(length):
        for j in range(width):
            for k in range(bin_length):
                PopulationBin[i, j*bin_length + k] = '{0:07b}'.format(Population[i, j])[k]
    return PopulationBin
                
def bin_to_int(PopulationBin, bin_length):
    length = len(PopulationBin[: , 1])
    width = len(PopulationBin[1, :])
    Population = np.empty([length, int(width/bin_length)])
    for i in range(length):
        for j in range(int(width/bin_length)):
            setup = 0
            for k in range(bin_length):
                setup += PopulationBin[i, j*bin_length +k] * np.power(2, bin_length-k-1)
            Population[i, j] = setup%120
    return np.int_(Population)


# Fitness calculation
def calculate_fitness(Population, Results, Best_solutions):
    PopFitness = model.predict(Population)
    Results.append(min(PopFitness))
    PopFitness = Results[-1]/PopFitness # Relative fitness in order to find minimum
    Best_solutions.append(Population[np.argmax(PopFitness) , :])
    return PopFitness


# Selection Probability Calculation methods
def roulette_wheel(PopFitness):
    SelectionProb = np.cumsum(PopFitness/sum(PopFitness))
    return SelectionProb    


# Parents population selection methods
def simple_selection(Population, Select_N, N_best, Results, Best_solutions):
    PopFitness = calculate_fitness(Population, Results, Best_solutions)
    SelectionProb = roulette_wheel(PopFitness)
    SelectedPool = np.empty( [Select_N, len(Population[1, :])] )
    # Take N_best chromosomes with the best fitness
    if N_best != 0:
        best_selection = np.argpartition(PopFitness.reshape(len(Population[:, 1]),), -N_best)[-N_best:]
    else:
        best_selection = []
    for i in range(len(best_selection)):
        SelectedPool[i] = Population[best_selection[i] , :]
    # Take Select_N - N_best random chromosomes with calculated probability
    for i in range(N_best, Select_N):
        p = np.random.uniform(0,1)
        SelectedPool[i] = Population[bisect_left(SelectionProb, p) , :]
    return SelectedPool

def distance_based_selection(Population, Select_N, N_best, Results, Best_solutions):
    PopFitness = calculate_fitness(Population, Results, Best_solutions)
    SelectedPool = np.empty( [Select_N, len(Population[1, :])] )
    SelectedPool[0, :] = Population[np.argmax(PopFitness), :]
    i = 1
    while i < Select_N:
        Distance = np.zeros( [len(Population[: , 1])] )
        for j in range(len(Population[: , 1])):
            d = 0
            for k in range(i):
                d += np.sqrt(np.sum((Population[j, :] - SelectedPool[k, :])**2))
            Distance[j] = d
        Distance = Distance/max(Distance)
        DistanceFitness = np.multiply(PopFitness**(1/2), (Distance**(1/2)).reshape(len(Distance), 1))
        SelectedPool[i, :] = Population[np.argmax(DistanceFitness), :]
        i += 1
    return SelectedPool


# Create next generation
def create_offspring(Cross_type, SelectedPool, Offspring_pairs, include_selected):
    length = len(SelectedPool[: , 1])
    NextGeneration = np.empty([(length*Offspring_pairs) , len(SelectedPool[1, :])])
    for i in range(Offspring_pairs):
        np.random.shuffle(SelectedPool)
        for j in range(int(length/2)):
            x1 = SelectedPool[2*j]
            x2 = SelectedPool[2*j + 1]
            NextGeneration[length*i + 2*j], NextGeneration[length*i + 2*j + 1] = eval('Cross_type(x1,x2)')
    # Add parentage population to next generation
    if include_selected == 1:
        return np.concatenate((NextGeneration, SelectedPool), axis=0)
    else:
        return NextGeneration


# Crossover methods
def cross_uniform(x1, x2):
    new_x1 = []
    new_x2 = []
    for i in range(len(x1)):
        p = np.random.uniform()
        if p > 0.5:
            new_x1.append(x1[i])
            new_x2.append(x2[i])
        else:
            new_x1.append(x2[i])
            new_x2.append(x1[i])
    return new_x1, new_x2

def cross_one_point(x1, x2):
    new_x1 = []
    new_x2 = []
    length = len(x1)
    cross_point = np.random.randint(1, length)
    new_x1[0 : cross_point] = x1[0 : cross_point]
    new_x1[cross_point : length] = x2[cross_point : length]
    new_x2[0 : cross_point] = x1[0 : cross_point]
    new_x2[cross_point : length] = x2[cross_point : length]
    return new_x1, new_x2


# Mutation methods
def mutate_swap(NextGeneration, mutate):
    length = len(NextGeneration[: , 1])
    width = len(NextGeneration[1 , :])
    for i in range(length):
        for j in range(width):
            p = np.random.uniform()
            if p < mutate:
                NextGeneration[i, j], NextGeneration[i, j-1] = NextGeneration[i, j-1], NextGeneration[i, j]

def mutate_random_change(NextGeneration, mutate, max_value):
    length = len(NextGeneration[: , 1])
    width = len(NextGeneration[1 , :])
    for i in range(length):
        for j in range(width):
            p = np.random.uniform()
            if p < mutate:
                NextGeneration[i, j] = np.random.randint(max_value+1)
           
            
# Algorithm
def one_generation(Results, Best_solutions, Population, Select_N, N_best, Offspring_pairs, include_selected, Cross_type, mutate_swap, mutate_random, binary, bin_length, max_value, selection_type, proportion, max_iter, i):
    # Selection
    if selection_type == 'simple_selection' or i >= proportion*max_iter:
        SelectedPool = simple_selection(Population, Select_N, N_best, Results, Best_solutions)
    elif selection_type == 'distance_based_selection' and i < proportion*max_iter:
        SelectedPool = distance_based_selection(Population, Select_N, N_best, Results, Best_solutions)
    else:
        sys.exit("Selection error")
    if binary == 1:
        SelectedPool = int_to_bin(np.int_(SelectedPool), bin_length)
    # Crossover
    if Select_N * Offspring_pairs + include_selected * Select_N == len(Population[: , 1]):
        NextGeneration = create_offspring(Cross_type, SelectedPool, Offspring_pairs, include_selected)
    else:
        sys.exit("Unstable population")
    # Mutation
    mutate_swap(NextGeneration, mutate_swap)
    if binary == 1:
        mutate_random_change(NextGeneration, mutate_random, 2)
        NextGeneration = bin_to_int(NextGeneration, bin_length)
    else:
        mutate_random_change(NextGeneration, mutate_random, max_value)
    return NextGeneration


def genetic_algorithm(Population, Select_N, N_best, Offspring_pairs, Cross_type, mutate_swap, mutate_random, max_iter, selection_type, proportion, include_selected = 0, binary = 0, bin_length = 7, max_value = 119):
    # Results gathering initialization
    Results = []
    Best_solutions = []
    Time = [0]
    # Generation loop
    i = 0
    while i < max_iter:
        start = time.time()
        Population = one_generation(Results, Best_solutions, Population, Select_N, N_best, Offspring_pairs, include_selected, Cross_type, mutate_swap, mutate_random, binary, bin_length, max_value, selection_type, proportion, max_iter, i)
        stop = time.time()
        Time.append(stop - start + Time[i])
        i += 1
    # Last generation results gathering
    calculate_fitness(Population, Results, Best_solutions)
    return Results, Time, Best_solutions


# Run algotithm
Population = np.random.randint(120, size = (40,21) )

Results, Time, Best_solutions = genetic_algorithm(# Initialization
                                                  Population = Population, # Initial population
                                                  max_iter = 150, # Number of generations
                                                  
                                                  # Chromosomes representation
                                                  binary = 0, # 0 - values of parameters, 1 - binary representation
                                                  bin_length = 7, # if binary = 1: number of bits in binary representation (default = 7)
                                                  max_value = 119, # if binary = 0: maximum value of parameter
                                                  
                                                  # Selection
                                                  Select_N = 20, # Number of chromosomes to be selected to parenatage population
                                                  N_best = 10, # Number of the best chromosomes to be deterministically selected to parenatage population
                                                  selection_type = 'distance_based_selection', # 'distance_based_selection' or 'simple_selection'
                                                  proportion = 0.7, # if 'distance_based_selection': distance based selection works well when after several iteration it is replaced by simple selection. Proportion variable indicates the share between distance based and simple selection.
                                                  
                                                  # Crossover
                                                  Cross_type = cross_uniform, # cross_uniform or cross_one_point
                                                  Offspring_pairs = 1, # How many pairs of offspring will by created by every parent
                                                  include_selected = 1, # if include_selected = 1 the parents will be added to next generation
                                                  
                                                  #Mutation
                                                  mutate_swap = 0.01, # Probability of swap mutation
                                                  mutate_random = 0.01 # Propability of random change mutation
                                                  )

plt.plot(range(len(Results)), Results, color="C0")
plt.show()
