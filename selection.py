from typing import Optional, Callable
import numpy as np
from generation import Generation


class Selection:

    def __init__(self, selection_size: int, take_best_ratio: Optional[int]=0):
        self.selection_pool = selection_pool
        self.take_best = np.floor(self.selection_pool * take_best_ratio)
        self.draw_n = self.selection_pool - self.take_best

    def pass_best(self, fitness):
        selected = np.empty((self.selection_pool))
        if self.take_best:
            selected[0:self.take_best] = np.argpartition(
                fitness, -self.take_best)[-self.take_best:]

        return selected

    def __call__(self, gen: Generation):
        return gen



class SimpleSelection(Selection):

    def __init__(self, prob_func: Callable, selection_size: int, take_best_ratio: Optional[int]=0):
        super().__init__(selection_size, take_best_ratio)
        self.prob_func = prob_func

    def __call__(self, gen: Generation, **kwargs):
        selected = self.pass_best(gen.fitness)
        try:
            prob = self.prob_func(gen.fitness, **kwargs)
        except:
            pass

        drawn_p = np.random.uniform(0, 1, size=self.draw_n)
        for idx in range(self.draw_n):
            index = min(bisect_left(prob, drawn_p), len(gen))
            selected[self.take_best + idx] = index
        gen.selected(selected)

        return gen


 

        




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