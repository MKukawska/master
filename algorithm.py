from typing import Optional, List, Callable, Union
import numpy as np
from generation import Generation


class GA:

    def __init__(self, fitness_func: Callable, pipe: Optional[List[Calable]]=None):
        self.fitness_func = fitness_func
        self.pipe = pipe

    def iteration(self):
        

    def run(self, population: np.array, max_iter: int, callback: List[Callable]):
        gen = Generation(population)
        gen.calculate_fitness(self.fitness_func)


