import numpy as np
from typing import Any

from cellular_automata.ca import CellValueType, ComputeMode, SimulationLogMode
from cellular_automata.gca import GraphCellularAutomaton
from graph import Graph

class GraphGameOfLife(GraphCellularAutomaton):
    """
    Conway's Game of Life on an arbitrary undirected graph.
    Rules are identical to the classic version:
      - Birth: dead cell with exactly 3 live neighbors → alive
      - Survival: live cell with 2 or 3 live neighbors → stays alive
      - Otherwise → dies
    """

    def __init__(
        self,
        num_nodes: int,
        random_seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(
            num_nodes=num_nodes,
            value_type=CellValueType.DISCRETE,
            value_options=[0, 1],
            random_seed=random_seed,
            **kwargs,
        )
        self.node_states: np.ndarray | None = None

    @property
    def compute_mode(self) -> ComputeMode:
        return ComputeMode.CONCURRENT

    def criteria(self) -> Graph:
        new_labels = np.zeros_like(self.graph.node_labels)

        for node in self.graph.nodes:
            neighbors = self.graph.neighbors(node)
            live_nbrs = np.sum(
                1 for nbr in neighbors
                if self.graph.node_labels[nbr] == 1
            )
            if len(neighbors) == 0:
                live_percentage = 0
            else:
                live_percentage = live_nbrs / len(neighbors)

            current = self.graph.node_labels[node]

            if current == 1:
                if 0.125 < live_percentage < 0.5:
                    new_labels[node] = 1
            else:
                if 0.2 < live_percentage < 0.5:
                    new_labels[node] = 1

        return new_labels

    def update(self, next_state: np.ndarray) -> None:
        self.graph.node_labels= next_state

    def save_snapshot(
        self,
        iteration: int,
        state: np.ndarray,
        time_: float | None = None
    ) -> None:
        self.performance["iteration_data"][iteration] = {
            "time": time_,
            "grid": self.grid,
            "node_states": np.copy(state),
        }
