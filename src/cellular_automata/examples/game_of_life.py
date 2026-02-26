import numpy as np
from scipy.signal import convolve2d

from cellular_automata import CellularAutomaton, CellValueType, ComputeMode


class GameOfLife(CellularAutomaton):
    def __init__(self, width, height):
        super().__init__(width, height, CellValueType.DISCRETE, [0, 1], random_seed=42)

    @property
    def compute_mode(self):
        return ComputeMode.CONCURRENT

    def criteria(self) -> np.ndarray:
        kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
        live_neighbors = convolve2d(self.grid, kernel, mode='same', boundary='wrap')
        birth    = (self.grid == 0) & (live_neighbors == 3)
        survival = (self.grid == 1) & np.isin(live_neighbors, [2, 3])
        return (birth | survival).astype(self.grid.dtype)
