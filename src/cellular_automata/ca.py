import concurrent.futures
import os
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Iterable, Sequence

import numpy as np


class CellValueType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1
    OBJECT = 2

class SimulationLogMode(Enum):
    NONE = 0
    MINIMAL = 1
    FULL = 2

class ComputeMode(Enum):
    SINGLE = 0
    CONCURRENT = 1

class CellularAutomaton(ABC):
    def __init__(
        self,
        width: int,
        height: int,
        value_type: CellValueType,
        value_options: Sequence,
        random_seed: int = 42,
        max_thread_workers: int | None = None,
        **kwargs,
    ):
        self._validate_value_type_options(value_type, value_options)

        self.width = width
        self.height = height
        self.value_options = value_options
        self.value_type = value_type

        self.random_seed = random_seed
        self.max_thread_workers = max_thread_workers
 
        self.grid = np.empty(shape=(self.height, self.width))

        self.history = {}
        self.performance = {
                "num_iterations": None,
                "total_time": None,
                "time_per_iteration": None,
                "total_compute": None,
                "iteration_data": {}
                }

        for k, v in kwargs.items():
            setattr(self, k, v)

    def _validate_value_type_options(self, value_type, value_options):
        if not value_options:
            raise ValueError("values_options cannot be empty.")

        if len(value_options) < 2:
            raise ValueError("values_options must have at least 2 values.")

        if value_type == CellValueType.CONTINUOUS and len(value_options) != 2:
            raise ValueError(f"CellValueType.CONTINUOUS requires exactly 2 value_options but {len(value_options)} were given.")

    @property
    @abstractmethod
    def compute_mode(self) -> ComputeMode:
        ...

    def initialize(self, from_array: np.ndarray | Iterable | None = None) -> "CellularAutomaton":
        """
        Initialize the cellular automata grid. If an array is given,
        it may be used to initialize the grid. Otherwise the grid
        will be randomly initialized.

        Args:
            from_array(np.ndarray | Iterable | None): The array to use to initialize the grid.

        Returns:
            CellularAutomata: The initialized cellular automata.

        Raises:
            TypeError: If the input array is not convertible to np.ndarray.
            ValueError: If the input array does not have the same shape as the grid.
                or if the input array values are not in the previously configured values options.
        """
        if from_array is None:
            np.random.seed(self.random_seed)
            if self.value_type == CellValueType.CONTINUOUS:
                self.grid = np.random.random(size=(self.height, self.width))
                self.grid = (
                        self.grid *
                        (self.value_options[1] - self.value_options[0]) 
                        + self.value_options[0]
                        )
            if self.value_type == CellValueType.DISCRETE:
                self.grid = np.random.choice(self.value_options, size=(self.height, self.width))
        else:
            if not isinstance(from_array, np.ndarray):
                try:
                    from_array = np.asarray(from_array)
                except Exception as e:
                    raise TypeError(f"{type(from_array)} is not castable to np.ndarray. {e}")

            #  check array shape to match the grid
            array_shape = from_array.shape
            if array_shape != self.grid.shape:
                raise ValueError(f"Input array must have the same shape as the cellular automata \
                    configuration {self.grid.shape}. {array_shape} was given")

            #  check array value options or range
            array_values = np.unique(from_array)
            if self.value_type == CellValueType.DISCRETE:
                if not np.all(np.isin(array_values, self.value_options)):
                    raise ValueError("Input array values must be in configured values.")
            if self.value_type == CellValueType.CONTINUOUS:
                array_min = np.min(array_values)
                array_max = np.max(array_values)
                if array_min < self.value_options[0] or array_max > self.value_options[1]:
                    raise ValueError(f"Input array values must be between {self.value_options[0]} \
                            and {self.value_options[1]}")


            self.grid = from_array

        return self
    
    @abstractmethod
    def criteria(self, *args, **kwargs) -> np.ndarray | Any:
        ...

    def step(self) -> np.ndarray | Any:
        if self.compute_mode == ComputeMode.CONCURRENT:
            next_state = self.criteria()
            if not isinstance(next_state, np.ndarray):
                raise TypeError(f"When using ComputeMode.GRID, criteria must return np.ndarray. {type(next_state)} was returned.")
            if next_state.shape != self.grid.shape:
                raise ValueError(f"Expected shape {self.grid.shape} but got array with shape {next_state.shape}.")
        elif self.compute_mode == ComputeMode.SINGLE:
            coords = [(i, j) for i in range(self.height) for j in range(self.width)]

            def criteria_with_coords(coord):
                return self.criteria(*coord)

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread_workers or 10) as executor:
                next_state = tuple(executor.map(criteria_with_coords, coords))

            next_state = np.array(next_state).reshape(self.grid.shape)
        else:
            raise ValueError(f"Compute mode {self.compute_mode} is not supported. Must be one of {ComputeMode}.")

        return next_state

    def update(self, next_state):
        self.grid = next_state

    def run(self, max_iterations: int = 100, log_mode: SimulationLogMode = SimulationLogMode.MINIMAL, keep_history: bool = True) -> dict[str, Any]:
        if log_mode != SimulationLogMode.NONE:
            sim_start = time.perf_counter()

        for i in range(max_iterations):
            if log_mode == SimulationLogMode.FULL:
                iter_start = time.perf_counter()
                next_state = self.step()
                iter_end = time.perf_counter()
                self.save_snapshot(i, next_state, iter_end - iter_start)
            else:
                next_state = self.step()

            if keep_history:
                self.history[i] = np.copy(next_state)

            self.update(next_state)

        if log_mode != SimulationLogMode.NONE:
            sim_end = time.perf_counter()
            self.performance["total_time"] = sim_end - sim_start
            self.performance["num_iterations"] = i + 1
            self.performance["time_per_iteration"] = (sim_end - sim_start) / (i + 1)

        return self.performance

    def save_snapshot(self, iteration: int, state: Any, time_: float | None = None):
        self.performance['iteration_data'][iteration] = {
            "time": time_,
            "grid": state,
        }

    def display(self, iteration: int | None = None):
        for row in range(self.height):
            for col in range(self.width):
                if iteration is None:
                    grid_to_print = self.grid
                elif iteration in self.history:
                    grid_to_print = self.history[iteration]
                else:
                    raise IndexError(f"History does not contain iteration {iteration}.")
                print(grid_to_print[row, col], end=" ")
            print()

    def replay(self, iterations: int | None = None, speed: float = 1.0):
        if not self.history:
            print("No history to replay. Make sure to run the simulation first or set keep_history=True.")
            return

        for i in range(iterations or len(self.history)):
            os.system("cls" if os.name == "nt" else "clear")
            self.display(i)
            time.sleep(1 / speed)

    def plot_performance(self):
        if not self.performance["iteration_data"]:
            print("No data to plot. Make sure to run the simulation first or set log_mode=SimulationLogMode.FULL.")
            return

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 7))
        x_data = list(self.performance["iteration_data"].keys())
        y_data = list(v['time'] for v in self.performance["iteration_data"].values())
        ax.plot(x_data, y_data)
        plt.legend('auto')

        plt.show()

    def export_performance(self, filename: str | None = None):
        if not self.performance['iteration_data']:
            print("No data to export. Make sure to run the simulation first and set log_mode=SimulationLogMode.FULL.")
            return

        import gzip
        import json
        if not filename:
            filename = f"{self.__class__.__name__.lower()}_performance_{time.time()}.json.gz"
        if not filename.endswith('.gz'):
            filename += ".gz"
        

        data = {}
        for k,v in self.performance.items():
            if k == 'iteration_data':
                data[k] = {
                    i: {
                        'time': v[i]['time'], 
                        'grid': [[round(x, 4) for x in row] for row in v[i]['grid'].tolist()]
                    } for i in v
                    }
            else:
                data[k] = v
        with gzip.open(filename, 'wt', encoding='utf-8') as f:
            json.dump(data, f, separators=(',', ':'))

        print(f"Performance data exported to {filename}")
