from typing import Callable, List
import numpy as np


class Generation:

    def __init__(self, population: np.array):
        self.pop = population
        self.features = self.pop.shape[1]

    def __len__(self):
        return len(self.pop)

    def calculate_fitness(self, fitness_func: Callable):
        self.fitness = self.fitness_func(self.pop)
        return self.fitness

    def selected(self, indices: List[int]):
        self.selected = self.pop[indices]