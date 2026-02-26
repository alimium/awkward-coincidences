import gzip
import json
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from cellular_automata.ca import CellularAutomaton, CellValueType
from graph import Graph, GraphEdgeType


class GraphCellularAutomaton(CellularAutomaton):
    def __init__(
            self, 
            num_nodes: int, 
            value_type: CellValueType, 
            value_options: Sequence,
            random_seed: int = 42,
            max_thread_workers: int | None = None,
            **kwargs,
        ):
        super().__init__(
                num_nodes,
                num_nodes,
                value_type,
                value_options,
                random_seed,
                max_thread_workers,
                **kwargs,
                )

        self.graph = None


    def initialize(
            self, 
            from_adj_matrix: np.ndarray | Sequence | None = None, 
            node_labels: Sequence | None = None,
            edge_type: GraphEdgeType | None = None
        ):
        super().initialize(from_array=from_adj_matrix)
        if edge_type is None:
            if np.array_equal(self.grid, self.grid.T):
                edge_type = GraphEdgeType.UNDIRECTED
            else:
                edge_type = GraphEdgeType.DIRECTED

        self.graph = Graph(
                num_nodes=self.width,
                adjacency_matrix=self.grid,
                node_labels=node_labels,
                edge_type=edge_type,
                )
        return self

    def update(self, next_state):
        self.graph = next_state

    def save_snapshot(
        self,
        iteration: int,
        state: Any,
        time_: float | None = None,
    ) -> None:
        self.performance["iteration_data"][iteration] = {
            "time": time_,
            "grid": np.copy(state.adjacency_matrix),
            "node_states": np.copy(state.node_labels),
        }

    def export_performance(self, filename: str | None = None) -> None:
        if not self.performance['iteration_data']:
            print(
                "No data to export. Run the simulation first with "
                "log_mode=SimulationLogMode.FULL."
            )
            return

        if not filename:
            filename = (
                f"{self.__class__.__name__.lower()}_graph_performance_{time.time()}.json.gz"
            )
        if not filename.endswith('.gz'):
            filename += '.gz'

        data: dict[str, Any] = {
            'type': 'graph',
            'num_iterations': self.performance.get('num_iterations'),
            'total_time':     self.performance.get('total_time'),
            'time_per_iteration': self.performance.get('time_per_iteration'),
            'iteration_data': {},
        }

        for i, entry in self.performance['iteration_data'].items():
            adj = entry['grid']
            node_st = entry.get('node_states', np.zeros(self.width))

            snap_graph = Graph(num_nodes=self.width, adjacency_matrix=adj)

            frame: dict[str, Any] = {
                'num_nodes':   self.width,
                'node_states': [round(float(v), 6) for v in node_st],   # float() handles int64
                'edges':       [
                    [int(u), int(v), round(float(w), 6)]                 # cast u, v too
                    for u, v, w in snap_graph.to_edge_list()
                ],
            }

            data['iteration_data'][str(int(i))] = {   # int(i) kills np.int64 keys
                'time':  float(entry['time']) if entry['time'] is not None else None,
                'graph': frame,
            }
        with gzip.open(filename, 'wt', encoding='utf-8') as f:
            json.dump(data, f, separators=(',', ':'))

        print(f"Graph performance data exported to {filename}")
