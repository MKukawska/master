import numpy as np
from scipy.stats import rankdata


def ranking(fitness: np.array, decay: float, **kwargs):
    assert decay < 1.0
    ranking = rankdata(fitness, method='ordinal')
    prob = np.array([((1 - decay)**(rank - 1)) * decay for rank in ranking])
    prob = np.cumsum(prob)
    prob = prob/max(prob)

    return prob


def roulette_wheel(fitness: np.array, **kwargs):
    return = np.cumsum(fitness / sum(fitness))