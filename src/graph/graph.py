from collections import defaultdict
from collections.abc import Sequence
from enum import Enum
from typing import Any

import numpy as np


class GraphEdgeType(Enum):
    DIRECTED = 0
    UNDIRECTED = 1

class Graph:
    def __init__(
        self,
        num_nodes: int,
        adjacency_matrix: np.ndarray | Sequence,
        node_labels: Sequence | None = None,
        edge_type: GraphEdgeType = GraphEdgeType.UNDIRECTED,
    ) -> None:
        if not isinstance(adjacency_matrix, np.ndarray):
            try:
                adjacency_matrix = np.asarray(adjacency_matrix)
            except Exception:
                raise TypeError(
                    f"{type(adjacency_matrix)} is not convertible to np.ndarray."
                )
        if adjacency_matrix.shape != (num_nodes, num_nodes):
            raise ValueError(
                "Adjacency matrix must be square with side length equal to num_nodes."
            )
        if edge_type == GraphEdgeType.UNDIRECTED and not np.array_equal(
            adjacency_matrix, adjacency_matrix.T
        ):
            raise ValueError(
                "Undirected graph requires a symmetric adjacency matrix."
            )
        if node_labels is not None and len(node_labels) != num_nodes:
            raise ValueError(
                "Length of node_labels must equal num_nodes."
            )

        self.num_nodes = num_nodes
        self.nodes = range(num_nodes)
        self.edge_type = edge_type

        self.node_labels: dict[int, Any] | None = (
            {i: node_labels[i] for i in self.nodes} if node_labels is not None else None
        )

        self._graph: dict[int, dict[int, float]] = self._build_graph(adjacency_matrix)

    def _check_node_exists(self, node: int):
        if node not in set(self.nodes):
            raise ValueError(f"Node {node} is not in the graph.")

    def _graph_index(self, node: int, inner_node: int | None = None):
        node = self._graph.get(node, {})
        if inner_node is not None:
            node = node.get(inner_node, None)
        return node

    def _build_graph(self, adjacency_matrix: np.ndarray) -> dict[int, dict[int, float]]:
        """
        BUild internal graph representation as dict of dicts.
        The values of the inner dicts are the edge weights defined
        in the adjacency matrix.

        Args:
            adjacency_matrix (np.ndarray): A square adjacency matrix.

        Returns:
            dict[int, dict[int, float]]: The internal graph representation.
        """
        rows, cols = np.nonzero(adjacency_matrix)
        graph: dict[int, dict[int, float]] = defaultdict(dict)

        if self.edge_type == GraphEdgeType.UNDIRECTED:
            for i, j in zip(rows, cols):
                if i <= j:
                    graph[i][j] = float(adjacency_matrix[i, j])
                    if i != j:
                        graph[j][i] = float(adjacency_matrix[j, i])
        else:
            for i, j in zip(rows, cols):
                graph[i][j] = float(adjacency_matrix[i, j])

        return graph

    def neighbors(self, node: int, level: int = 1) -> set[int]:
        """
        Return the list of reachable nodes from a node.
        
        Arge:
            node (int): The node to start from.
            level (int): The number of steps to reach neighbors.

        Returns:
            set[int]: The set of reachable nodes.

        Raises:
            ValueError: If level is less than 1 or the node is not in the graph.
        """
        if level < 1:
            raise ValueError("Level must be non-negative.")
        self._check_node_exists(node)

        neighbors = self._graph_index(node)
        has_loop = node in neighbors
        neighbors = set(neighbors.keys())
        #  NOTE: we need level 1 neighbors to be fast
        if level > 1:
            for _ in range(level):
                neighbors = set().union(*[self._graph_index(node) for n in neighbors])
            if not has_loop:
                neighbors.discard(node)

        return neighbors

    def weight(self, u: int, v: int) -> float | None:
        """
        Return the weight of an edge, or None if it does not exist.
        
        Args:
            u (int): The source node.
            v (int): The destination node.

        Returns:
            float: The weight of the edge.
            None: If the edge does not exist.

        Raises:
            ValueError: If either node is not in the graph.
        """
        self._check_node_exists(u)
        self._check_node_exists(v)

        return self._graph_index(u, v)

    def in_degree(self, node: int) -> int:
        """
        Return the in-degree of a node. This operation 
        is more expensive compared to the out-degree.
        
        Args:
            node (int): The node to check.

        Returns:
            int: The in-degree of the node.

        Raises:
            ValueError: If the graph is undirected or the node does not exist.
        """
        if self.edge_type == GraphEdgeType.UNDIRECTED:
            raise ValueError("Use Graph.degree for undirected graphs.")
        self._check_node_exists(node)

        return len({i for i in self._graph if node in self._graph_index(i)})

    def out_degree(self, node: int) -> int:
        """
        Return the out-degree of a node.

        Args:
            node (int): The node to check.

        Returns:
            int: The out-degree of the node.

        Raises:
            ValueError: If the graph is undirected or the node does not exist.
        """
        if self.edge_type == GraphEdgeType.UNDIRECTED:
            raise ValueError("Use Graph.degree for undirected graphs.")
        self._check_node_exists(node)

        return len(self._graph_index(node))

    def degree(self, node: int) -> int:
        """
        Return the degree of a node.

        Args:
            node (int): The node to check.

        Returns:
            int: The degree of the node.

        Raises:
            ValueError: If the graph is directed or the node does not exist.
        """
        if self.edge_type == GraphEdgeType.DIRECTED:
            raise ValueError("Use in_degree or out_degree for directed graphs.")
        self._check_node_exists(node)

        return len(self._graph_index(node))

    def add_edge(self, u: int, v: int, weight: float = 1.0) -> None:
        """
        Add or update an edge from u to v.
        The reverse edge is handled for undirected graphs.

        Args:
            u (int): The source node.
            v (int): The destination node.
            weight (float): The weight of the edge.

        Returns:
            None

        Raises:
            ValueError: If either node is not in the graph.
        """
        self._check_node_exists(u)
        self._check_node_exists(v)

        self._graph[u][v] = weight
        if self.edge_type == GraphEdgeType.UNDIRECTED and u != v:
            self._graph[v][u] = weight

    def remove_edge(self, u: int, v: int) -> None:
        """
        Remove an edge. The reverse edge is handled for undirected graphs.

        Args:
            u (int): The source node.
            v (int): The destination node.

        Returns:
            None

        Raises:
            ValueError: If either node is not in the graph.
        """

        self._check_node_exists(u)
        self._check_node_exists(v)

        if self._graph_index(u, v):
            del self._graph[u][v]
            if self.edge_type == GraphEdgeType.UNDIRECTED and u != v:
                del self._graph[v][u]
 
    def to_edge_list(self) -> list[tuple[int, int, float]]:
        """
        Return all edges as a list of (i, j, weight) tuples.
        This should ideally not be called frequently as it 
        takes O(num_nudes^2) time.

        Returns:
            list[tuple[int, int, float]]: The list of edges.
        """
        edges = []
        seen: set[tuple[int, int]] = set()
        for i, nbrs in self._graph.items():
            for j, w in nbrs.items():
                key = (
                        (min(i, j), max(i, j)) 
                        if self.edge_type == GraphEdgeType.UNDIRECTED
                        else (i, j)
                    )
                if key not in seen:
                    seen.add(key)
                    edges.append((i, j, w))
        return edges

    def __len__(self) -> int:
        return self.num_nodes

    def __repr__(self) -> str:
        prefix = (
                "Directed"
                if self.edge_type == GraphEdgeType.DIRECTED
                else "Undirected"
                if self.edge_type == GraphEdgeType.UNDIRECTED
                else ""
            )
        num_nodes = self.num_nodes
        num_edges = len(self.to_edge_list())
        return f"{prefix}Graph(num_nodes={num_nodes}, num_edges={num_edges})"

