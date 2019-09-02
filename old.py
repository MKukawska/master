from __future__ import print_function
from bisect import bisect_left
import numpy as np
import sys
import time
from scipy.stats import rankdata
from sklearn.utils import shuffle
from math import floor
np.random.seed(0)
'''
Data representation - converting integer to binary vector and vice versa
'''


def int_to_bin(Population, bin_length):
    length = len(Population[:, 1])
    width = len(Population[1, :])
    PopulationBin = np.empty([length, width * bin_length])
    for i in range(length):
        for j in range(width):
            for k in range(bin_length):
                PopulationBin[i, j * bin_length + k] = '{0:07b}'.format(
                    Population[i, j])[k]
    return PopulationBin


def bin_to_int(PopulationBin, bin_length, max_value):
    length = len(PopulationBin[:, 1])
    width = len(PopulationBin[1, :])
    Population = np.empty([length, int(width / bin_length)])
    for i in range(length):
        for j in range(int(width / bin_length)):
            setup = 0
            for k in range(bin_length):
                setup += PopulationBin[i, j * bin_length + k] * np.power(
                    2, bin_length - k - 1)
            Population[i, j] = setup % (max_value + 1)
    return np.int_(Population)


'''
Fitness calculation
'''


def calculate_fitness(Population, model, Results, Best_solutions):
    PopFitness = model.predict(Population)
    Results.append([min(PopFitness)[0], np.mean(PopFitness)])
    PopFitness = Results[-1][
        0] / PopFitness  # Relative fitness in order to find minimum
    Best_solutions.append(Population[np.argmax(PopFitness), :])
    return PopFitness


'''
Selection Probability Calculation methods
'''


def roulette_wheel(PopFitness):
    SelectionProb = np.cumsum(PopFitness / sum(PopFitness))
    return SelectionProb


def ranking(PopFitness, Ps):
    Rank = rankdata(-PopFitness, method='ordinal')
    SelectionProb = np.array([((1 - Ps)**(rank - 1)) * Ps for rank in Rank])
    SelectionProb[np.argmax(Rank)] = (1 - Ps)**(np.max(Rank) - 1)
    SelectionProb = np.cumsum(SelectionProb)
    return SelectionProb


'''
Parents population selection methods
'''


def simple_selection(Population, model, Select_N, N_best, Results,
                     Best_solutions, Ps, Ranking_pr):
    PopFitness = calculate_fitness(Population, model, Results, Best_solutions)
    if Ranking_pr == True:
        SelectionProb = ranking(PopFitness, Ps)
    else:
        SelectionProb = roulette_wheel(PopFitness)
    SelectedPool = np.empty([Select_N, len(Population[1, :])])
    # Take N_best chromosomes with the best fitness
    if N_best != 0:
        best_selection = np.argpartition(
            PopFitness.reshape(len(Population[:, 1]), ), -N_best)[-N_best:]
    else:
        best_selection = []
    # print(N_best, SelectedPool.shape, best_selection[15])

    for i in range(len(best_selection)):
        SelectedPool[i] = Population[best_selection[i], :]
    # Take Select_N - N_best random chromosomes with calculated probability
    for i in range(N_best, Select_N):
        try:
            p = np.random.uniform(0, 1)
            # we don't want to go after the size of the array
            index = bisect_left(SelectionProb, p)
            index = Population.shape[0] - 1 if index >= Population.shape[
                0] else index
            SelectedPool[i] = Population[index, :]
        except Exception as inst:
            output = [
                'A very specific bad thing happened.', inst, 'p: ', p,
                'SelectedPool:', SelectedPool, 'i', i, 'Population:',
                Population, 'SelectionProb:', SelectionProb
            ]
            import pickle
            with open('error.pkl', 'wb') as fw:
                pickle.dump(output, fw, pickle.HIGHEST_PROTOCOL)
            raise ValueError('A very specific bad thing happened.')
    return SelectedPool


def distance_based_selection(Population, model, Select_N, N_best, Results,
                             Best_solutions):
    PopFitness = calculate_fitness(Population, model, Results, Best_solutions)
    SelectedPool = np.empty([Select_N, len(Population[1, :])])
    SelectedPool[0, :] = Population[np.argmax(PopFitness), :]
    i = 1
    while i < Select_N:
        Distance = np.zeros([len(Population[:, 1])])
        for j in range(len(Population[:, 1])):
            d = 0
            for k in range(i):
                d += np.sqrt(
                    np.sum((Population[j, :] - SelectedPool[k, :])**2))
            Distance[j] = d
        Distance = Distance / max(Distance)
        DistanceFitness = np.multiply(PopFitness**(1 / 2),
                                      (Distance**(1 / 2)).reshape(
                                          len(Distance), 1))
        SelectedPool[i, :] = Population[np.argmax(DistanceFitness), :]
        i += 1
    return SelectedPool


def tournament_selection(Population, model, Tournament_size, Ps, Ranking_pr,
                         Results, Best_solutions):
    PopFitness = calculate_fitness(Population, model, Results, Best_solutions)
    Select_N = int(np.ceil(len(Population[:, 1]) / Tournament_size))
    SelectedPool = np.zeros([Select_N, len(Population[1, :])])
    PopFitness, Population = shuffle(PopFitness, Population)
    for i in range(Select_N):
        Tournament = Population[i * Tournament_size:i * Tournament_size +
                                Tournament_size, :]
        T_PopFitness = PopFitness[i * Tournament_size:i * Tournament_size +
                                  Tournament_size, :]
        if Ranking_pr == True:
            SelectionProb = ranking(T_PopFitness, Ps)
        else:
            SelectionProb = roulette_wheel(T_PopFitness)
        p = np.random.uniform(0, 1)
        index = bisect_left(SelectionProb, p)
        # we don't want to go after the size of the array
        index = Tournament.shape[0] - 1 if index >= Tournament.shape[
            0] else index
        SelectedPool[i] = Tournament[index, :]
    return SelectedPool


def create_offspring(Cross_type, SelectedPool, Offspring_pairs,
                     include_selected):
    """
    Creating next generation
    :param Cross_type:
    :param SelectedPool:
    :param Offspring_pairs:
    :param include_selected:
    :return:
    """
    length = len(SelectedPool[:, 1])
    NextGeneration = np.empty([(length * Offspring_pairs),
                               len(SelectedPool[1, :])])
    for i in range(Offspring_pairs):
        np.random.shuffle(SelectedPool)
        for j in range(int(np.ceil(length / 2))):
            x1 = SelectedPool[2 * j]
            x2 = SelectedPool[(2 * j + 1) % length]
            if Cross_type == 'cross_one_point':
                NextGeneration[length*i + 2*j], NextGeneration[(length*i + 2*j + 1)%(Offspring_pairs*length)] =\
                    cross_one_point(x1, x2)
            elif Cross_type == 'cross_uniform':
                NextGeneration[length*i + 2*j], NextGeneration[(length*i + 2*j + 1)%(Offspring_pairs*length)] =\
                    cross_uniform(x1, x2)
            elif Cross_type == 'cross_two_points':
                NextGeneration[length*i + 2*j], NextGeneration[(length*i + 2*j + 1)%(Offspring_pairs*length)] = \
                    cross_two_points(x1, x2)
            elif Cross_type == 'cross_map_one_group':
                NextGeneration[length*i + 2*j], NextGeneration[(length*i + 2*j + 1)%(Offspring_pairs*length)] = \
                    cross_map_one_group(x1, x2)
            elif Cross_type == 'cross_map_two_groups':
                NextGeneration[length*i + 2*j], NextGeneration[(length*i + 2*j + 1)%(Offspring_pairs*length)] =\
                    cross_map_two_groups(x1, x2)
            elif Cross_type == 'cross_map_neighbors':
                NextGeneration[length*i + 2*j], NextGeneration[(length*i + 2*j + 1)%(Offspring_pairs*length)] = \
                    cross_map_neighbors(x1, x2)
            else:
                print('Error - no crossover')
    # Add parentage population to next generation
    if include_selected:
        return np.int_(np.concatenate((NextGeneration, SelectedPool), axis=0))
    else:
        return np.int_(NextGeneration)


"""
Crossover methods
"""


def cross_uniform(x1, x2):
    new_x1 = []
    new_x2 = []
    for i in range(len(x1)):
        p = np.random.uniform()
        if p < 0.5:
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
    new_x1[0:cross_point] = x1[0:cross_point]
    new_x1[cross_point:length] = x2[cross_point:length]
    new_x2[0:cross_point] = x2[0:cross_point]
    new_x2[cross_point:length] = x1[cross_point:length]
    return new_x1, new_x2


def cross_two_points(x1, x2):
    new_x1 = []
    new_x2 = []
    length = len(x1)
    cross_point_1 = np.random.randint(1, length - 1)
    cross_point_2 = np.random.randint(cross_point_1, length)
    new_x1[0:cross_point_1] = x1[0:cross_point_1]
    new_x1[cross_point_1:cross_point_2] = x2[cross_point_1:cross_point_2]
    new_x1[cross_point_2:length] = x1[cross_point_2:length]
    new_x2[0:cross_point_1] = x2[0:cross_point_1]
    new_x2[cross_point_1:cross_point_2] = x1[cross_point_1:cross_point_2]
    new_x2[cross_point_2:length] = x2[cross_point_2:length]
    return new_x1, new_x2


def crossroads_grouping():
    # crossroads grouping according to the map
    # the groups are disjoint!!!
    group_1 = [6, 7, 8, 5, 20, 19, 0, 17]
    group_2 = [16, 1, 13, 9]
    group_3 = [10, 2, 3, 18]
    group_4 = [4, 15, 14, 11, 12]
    return [group_1, group_2, group_3, group_4]


def extract_neighbors(x, nearest_neighbors):
    x_neighbor = []
    x_left = []
    for i in range(len(x)):
        if i in nearest_neighbors:
            x_neighbor.append(x[i])
        else:
            x_left.append(x[i])
    return x_neighbor, x_left


def adjust_length(x, aux_x, length):
    """
    :param x: given vector
    :param aux_x: auxiliary vector from which we get elements, if necessary
    :param length: required length of x
    :return: x cut/prolonged to len(x)=length
    """
    new_x = []
    new_aux_x = []
    if len(x) == length:
        return x, aux_x
    if len(x) < length:
        difference = length - len(x)
        new_x[:len(x)] = x
        new_x[len(x):length] = aux_x[0:difference]
        new_aux_x = aux_x[difference:]
        return new_x, new_aux_x
    if len(x) > length:
        difference = len(x) - length
        new_x = x[:length]
        new_aux_x[0:difference] = x[length:len(x)]
        new_aux_x[difference:] = aux_x
        return new_x, new_aux_x


def cross_map_one_group(x1, x2):
    """
    Chooses randomly one group from 4 groups defined by crossroads_grouping
    New x_1 takes elements from x_1 that belonging to group, and elements from x2 that does not belong, same for new_x2
    Adjusts len of x2_left if necessary
    :param x1:
    :param x2:
    :return: new_x1, new_x2
    """
    new_x1 = []
    new_x2 = []
    g_index = np.random.randint(0, 4)
    groups = crossroads_grouping()
    x1_g, x1_left = extract_neighbors(x1, groups[g_index])
    x2_g, x2_left = extract_neighbors(x2, groups[g_index])
    new_length_g, new_length_left, tot_length = len(x1_g), len(x1_left), len(x1)
    tot_length = len(x1)
    new_x1[0:new_length_g], new_x1[new_length_g:tot_length] = x1_g, x2_left
    new_x2[0:new_length_left], new_x2[new_length_left:tot_length] = x2_g, x1_left
    return new_x1, new_x2


def cross_map_two_groups(x1, x2):
    """
    Chooses randomly two groups from 4 groups defined by crossroads_grouping
    New x_1 takes elements from x_1 that belonging to groups, and elements from x2 that does not belong, same for new_x2
    Adjusts len of x2_left if necessary
    :param x1:
    :param x2:
    :return: new_x1, new_x2
    """
    new_x1 = []
    new_x2 = []
    g1_index = np.random.randint(1, 4)
    g2_index = np.random.randint(0, g1_index)
    groups = crossroads_grouping()
    x1_g, x1_left = extract_neighbors(x1, groups[g1_index] + groups[g2_index])
    x2_g, x2_left = extract_neighbors(x2, groups[g1_index] + groups[g2_index])
    new_length_g = len(x1_g)
    new_length_g2 = len(x2_g)
    tot_length = len(x1)
    new_x1[0:new_length_g], new_x1[new_length_g:tot_length] = x1_g, x2_left
    new_x2[0:new_length_g2], new_x2[new_length_g2:tot_length] = x2_g, x1_left
    return new_x1, new_x2


def cross_map_neighbors(x1, x2, d=3):
    """
    Chooses randomly one group from 4 groups defined by crossroads_grouping
    New x_1 takes elements from x_1 that belonging to group, and elements from x2 that does not belong, same for new_x2
    Adjusts len of x2_left if necessary
    :param x1:
    :param x2:
    :param d: distance
    :return: new_x1, new_x2
    """
    new_x1 = []
    new_x2 = []
    cross_point = np.random.randint(1, len(x1) - 1)
    groups = crossroads_grouping()
    group = [group for group in groups if cross_point in group][0]
    position_in_group = group.index(cross_point)
    nearest_neighbors = []

    for i in range(d):
        if position_in_group + i < len(group):
            nearest_neighbors.append(group[position_in_group + i])
        if position_in_group - i >= 0:
            nearest_neighbors.append(group[position_in_group - i])

    x1_n, x1_left = extract_neighbors(x1, nearest_neighbors)
    x2_n, x2_left = extract_neighbors(x2, nearest_neighbors)
    new_length_g, new_length_left, tot_length = len(x1_n), len(x1_left), len(
        x1)
    tot_length = len(x1)
    new_x1[0:new_length_g], new_x1[new_length_g:tot_length] = x1_n, x2_left
    new_x2[0:new_length_left], new_x2[new_length_left:
                                      tot_length] = x2_n, x1_left
    return new_x1, new_x2


"""
Mutation methods
"""


def mutate_swap(NextGeneration, mutate):
    length = len(NextGeneration[:, 1])
    width = len(NextGeneration[1, :])
    for i in range(length):
        for j in range(width):
            p = np.random.uniform()
            if p < mutate:
                NextGeneration[i, j], NextGeneration[
                    i, j - 1] = NextGeneration[i, j - 1], NextGeneration[i, j]


def mutate_random_change(NextGeneration, mutate, max_value, base=1):
    length = len(NextGeneration[:, 1])
    width = len(NextGeneration[1, :])
    for i in range(length):
        for j in range(width):
            p = np.random.uniform()
            if p < mutate:
                NextGeneration[i, j] = roundint(
                    np.random.randint(max_value + 1), base)


def roundint(x, base):
    return int(base * round(float(x) / base))


def granulate_population(Population, base=5):
    return np.array([[roundint(x, base) for x in p] for p in Population])


def one_generation(Results,
                   Best_solutions,
                   Population,
                   model,
                   Select_N,
                   N_best,
                   Offspring_pairs,
                   include_selected,
                   Cross_type,
                   mut_swap,
                   mut_random,
                   binary,
                   bin_length,
                   max_value,
                   selection_type,
                   proportion,
                   max_iter,
                   i,
                   Ps,
                   Ranking_pr,
                   Tournament_size,
                   base=1):
    """
    Realization of the algorithm
    :param Results:
    :param Best_solutions:
    :param Population:
    :param model:
    :param Select_N:
    :param N_best:
    :param Offspring_pairs:
    :param include_selected:
    :param Cross_type:
    :param mut_swap:
    :param mut_random:
    :param binary:
    :param bin_length:
    :param max_value:
    :param selection_type:
    :param proportion:
    :param max_iter:
    :param i:
    :param Ps:
    :param Ranking_pr:
    :param Tournament_size:
    :param base:
    :return:
    """
    # Selection
    if selection_type == 'tournament_selection':
        SelectedPool = tournament_selection(Population, model, Tournament_size,
                                            Ps, Ranking_pr, Results,
                                            Best_solutions)
    elif selection_type == 'simple_selection' or i >= proportion * max_iter:
        SelectedPool = simple_selection(Population, model, Select_N, N_best,
                                        Results, Best_solutions, Ps,
                                        Ranking_pr)
    elif selection_type == 'distance_based_selection' and i < proportion * max_iter:
        SelectedPool = distance_based_selection(
            Population, model, Select_N, N_best, Results, Best_solutions)
    else:
        sys.exit("Selection error")
    if binary == True:
        SelectedPool = int_to_bin(np.int_(SelectedPool), bin_length)
    # Crossover
    NextGeneration = create_offspring(Cross_type, SelectedPool,
                                      Offspring_pairs, include_selected)
    # Mutation
    mutate_swap(NextGeneration, mut_swap)
    if binary == True:
        mutate_random_change(NextGeneration, mut_random, 2)
        NextGeneration = granulate_population(
            bin_to_int(NextGeneration, bin_length, max_value), base)
    else:
        mutate_random_change(NextGeneration, mut_random, max_value, base)
    return NextGeneration


# def writeoutput(writefile):


def genetic_algorithm(Population,
                      model,
                      Cross_type,
                      mut_swap,
                      mut_random,
                      max_iter,
                      selection_type,
                      proportion=0.7,
                      Select_N=0, 
                      N_best=0, # can be 0.5, max 1
                      include_selected=False,
                      binary=False,
                      bin_length=7,
                      max_value=119,
                      Ranking_pr=False,
                      Ps=0.9,
                      Tournament_size=3,
                      granularity=[5, 3, 1],
                      file=''):

    import csv
    gridwriter = ''
    if file != '':
        csv_file = open(file, 'w', newline='')
        gridwriter = csv.writer(csv_file, delimiter=',')
        gridwriter.writerow(
            ('Iteration', 'Num', 'Population_size', 'Population', 'Cross_type',
             'mut_swap', 'mut_random', 'max_iter', 'selection_type',
             'proportion', 'Select_N', 'N_best', 'include_selected', 'binary',
             'bin_length', 'max_value', 'Ranking_pr', 'Ps', 'Tournament_size',
             'Result', 'Simulation', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7',
             'L8', 'L9', 'L10', 'L11', 'L12', 'L13', 'L14', 'L15', 'L16',
             'L17', 'L18', 'L19', 'L20', 'L21'))

    # Results gathering initialization
    Results = []
    Best_solutions = []
    Time = [0]
    # Calculating offsping pairs
    if selection_type == 'tournament_selection':
        Offspring_pairs = int(Tournament_size - include_selected)
    else:
        Offspring_pairs = int(round(
            len(Population[:, 1]) / Select_N - include_selected))

    # Select_N = int(round(Select_N*len(Population)))
    # N_best = int(round(N_best*len(Population)))
    
    # Generation loop
    i = 0
    while i < max_iter:
        base = granularity[floor(i / (max_iter / len(granularity)))]
        Population = granulate_population(Population, base)
        start = time.time()

        
        # print(Select_N,N_best,len(Population))

        Population = one_generation(
            Results, Best_solutions, Population, model, Select_N, N_best,
            Offspring_pairs, include_selected, Cross_type, mut_swap,
            mut_random, binary, bin_length, max_value, selection_type,
            proportion, max_iter, i, Ps, Ranking_pr, Tournament_size, base)

        PopFitness = model.predict(Population)

        if file != '':
            gridwriter.writerows(
                [(i, ii, len(Population), 'Random_pop', Cross_type, mut_swap,
                  mut_random, max_iter, selection_type, proportion, Select_N,
                  N_best, include_selected, binary, bin_length, max_value,
                  Ranking_pr, Ps, Tournament_size, PopFitness[ii][0], '',
                  Population[ii][0], Population[ii][1], Population[ii][2],
                  Population[ii][3], Population[ii][4], Population[ii][5],
                  Population[ii][6], Population[ii][7], Population[ii][8],
                  Population[ii][9], Population[ii][10], Population[ii][11],
                  Population[ii][12], Population[ii][13], Population[ii][14],
                  Population[ii][15], Population[ii][16], Population[ii][17],
                  Population[ii][18], Population[ii][19], Population[ii][20])
                 for ii in range(len(Population))])

        stop = time.time()
        Time.append(stop - start + Time[i])
        i += 1
    # Last generation results gathering
    calculate_fitness(Population, model, Results, Best_solutions)

    # if file!='':
    # close(csv_file)
    return Results, Time, Best_solutions
