from concurrent.futures import Executor
from copy import deepcopy
from itertools import groupby
from typing import Any, Callable, List, Optional, Protocol, Tuple
import networkx as nx


Cluster = set[str]


class PSection(Protocol):
    def get_name(self) -> str:
        ...

    def __lt__(self, other: "PSection") -> bool:
        ...

    def __gt__(self, other: "PSection") -> bool:
        ...

    def __eq__(self, other: Any) -> bool:
        ...


class HierarchyFactory:
    def __init__(self):
        self.main_graph = nx.DiGraph()
        self.temp_graph = nx.DiGraph()
        self._delimiter = "/"  # also acts as the root name
        self.current_path = self._delimiter
        self.main_graph.add_node(
            self._delimiter,
            section=None,  # TODO: allow for this to be set
        )  # Root node with empty section list

    # path formatting
    def _split_path(self, path: str) -> list[str]:
        """Split the path into its parts."""
        return [node for node in path.split(self._delimiter) if node]

    def _formalize_path(self, path: str) -> str:
        """Formalize the path by adding the delimiter to the beginning and end, as well as pruning any redundant delimiters."""
        return self._delimiter.join(["", *self._split_path(path), ""])

    def _to_absolute_path(self, relative_path: str) -> str:
        """Converts `relative_path` to an absolute path."""
        return self._formalize_path(self.current_path + relative_path)

    def _seg_to_graph(self, path_segments: list[str]) -> nx.DiGraph:
        """Translate `path_segments` (as produced by `_split_path`) to its graph counter version."""
        if len(path_segments) == 0:
            path_segments = [self._delimiter]
        right_shifted = [self._delimiter] + path_segments[:-1]
        return nx.DiGraph().add_edges_from(
            [(n, s) for n, s in zip(path_segments, right_shifted)]
        )

    # setters
    def set_root_section(self, section_handler: PSection) -> None:
        """Set the root node to `section_handler`."""
        self.main_graph.nodes[self._delimiter]["section"] = section_handler

    def include_section(self, section_handler: PSection, name: Optional[str]) -> None:
        """Add `section_handler` to the temporary graph for later construction."""
        if name is None:
            name = section_handler.get_name()
        self.temp_graph.add_node(name, section=section_handler)

    def add_to_path(
        self,
        section_handler: PSection,
        parent_path: str,
        given_name: Optional[str] = None,
    ) -> None:
        """Add `section_handler` directly to the main graph under the `parent_path` (relative)."""
        if given_name is None:
            given_name = section_handler.get_name()

        path_segments = self._split_path(self._to_absolute_path(parent_path))
        if self._seg_to_graph(path_segments) in self.main_graph:
            self.main_graph.add_node(given_name, section=section_handler)
            self.main_graph.add_edge(given_name, path_segments[-1])

    # (automated) construction
    def _check_node_consistency(self) -> Optional[ValueError]:
        for node in self.main_graph.nodes():
            for operator in ["__lt__", "__gt__", "__eq__"]:
                if not hasattr(self.main_graph.nodes[node]["section"], operator):
                    raise ValueError(
                        f"Node {node} does not implement the comparison methods."
                    )  # Does not check for interoperability between different node types

    def _check_edge_consistency(self) -> Optional[ValueError]:
        for edge in self.main_graph.edges():
            if not (
                self.main_graph.nodes[edge[0]]["section"]
                < self.main_graph.nodes[edge[1]]["section"]
            ):
                raise ValueError(
                    f"Graph is inconsistently ordered: {edge[0]} is not less than or equal {edge[1]}."
                )

    # TODO: consider handler for cycles: splitting them

    def _get_extremities(self, head: bool = True) -> list[str]:
        if head:
            return [
                node for node, degree in self.main_graph.out_degree() if degree == 0
            ]
        return [node for node, degree in self.main_graph.in_degree() if degree == 0]

    def _move_along_branch(self, branch_node_name: str, up: bool = True) -> list[str]:
        if up:
            return list(self.main_graph.predecessors(branch_node_name))
        return list(self.main_graph.successors(branch_node_name))

    def _dispatch_attach(
        self, attach_branch_node_name: str, target_parent_node_name: str
    ) -> list[Callable]:
        futures: list[Callable] = []  # acts as a trampoline for the attach function?
        for target_node_name in self._move_along_branch(
            target_parent_node_name, up=False
        ):
            futures.append(
                Executor.submit(
                    self._check_attach, attach_branch_node_name, target_node_name
                )
            )
        return futures

    def _check_attach(
        self,
        attach_branch_node_name: str,
        target_node_name: str,
        target_parent_node_name: str,
    ) -> Optional[list[Callable]]:
        if (
            self.main_graph.nodes[attach_branch_node_name]["section"]
            > self.main_graph.nodes[target_node_name]["section"]
        ):
            return self.main_graph.add_edge(attach_branch_node_name, target_node_name)
        elif (
            self.main_graph.nodes[attach_branch_node_name]["section"]
            == self.main_graph.nodes[target_node_name]["section"]
        ):
            return self.main_graph.add_edge(
                attach_branch_node_name, target_parent_node_name
            )
        return self._dispatch_attach(
            attach_branch_node_name,
            target_node_name,
        )

    def _sort_heads(self, heads: list[str]) -> list[list[str]]:
        comparison_key = lambda head: self.main_graph.nodes[head]["section"]
        return [
            list(group)
            for _, group in groupby(sorted(heads, comparison_key), key=comparison_key)
        ]

    def _automated_construct(self) -> None:
        """Automatically constructs the temporary graph into the main graph.
        This procedure relies on the `PSection` comparison methods."""
        sorted_heads = self._sort_heads(self._get_extremities(self.main_graph, head=True))
        for smaller_branch_head, larger_branch_head in zip(
            sorted_heads[:-1], sorted_heads[1:]
        ):
            self._dispatch_attach(smaller_branch_head, larger_branch_head)

    def construct(self) -> None:
        """Constructs the temporary graph into the main graph."""
        # migrate the temporary graph to the main graph if so
        self.main_graph = deepcopy(self.temp_graph)
        if not nx.is_weakly_connected(self.main_graph):
            # check if the temporary graph is a valid tree
            self._check_graph_consistency(self.main_graph)
            # rely on `self._automated_construct` to do the heavy lifting
            return self._automated_construct()

    # navigation
    def cd(self, path: Optional[str]) -> None:
        """Change the current path to the target path.
        All further path specifications are relative to the current path."""
        if path is None:
            self.current_path = self._delimiter
            return

        path = self._to_absolute_path(path)
        if self._seg_to_graph(self._split_path(path)) in self.main_graph:
            self.current_path = path
            return
        raise ValueError(f"Path {path} does not exist in `main_graph`.")

    def pwd(self) -> str:
        """Print the current path."""
        return self.current_path

    def ls(self, path: Optional[str] = None) -> List[str]:
        """List the contents of the target directory (relative path)."""
        path = self._to_absolute_path(path)
        if self._seg_to_graph(self._split_path(path)) in self.main_graph:
            return list(self.main_graph.successors(path))

    # searching
    def _match_pattern(self, node: str, pattern_parts: List[str], index: int) -> bool:
        """Recursively checks if the node path matches the pattern."""
        if index == len(pattern_parts):
            return True
        if pattern_parts[index] == "**":
            if index + 1 == len(pattern_parts):
                return True
            for descendant in nx.descendants(self.main_graph, node):
                if self._match_pattern(descendant, pattern_parts, index + 1):
                    return True
        elif pattern_parts[index] == "*":
            for successor in self.main_graph.successors(node):
                if self._match_pattern(successor, pattern_parts, index + 1):
                    return True
        else:
            next_node = f"{node}/{pattern_parts[index]}".replace("//", "/")
            if self.main_graph.has_node(next_node) and self._match_pattern(
                next_node, pattern_parts, index + 1
            ):
                return True
        return False

    def find(self, query: str) -> List[str]:
        """Finds paths matching the query with support for wildcards."""
        pattern_parts = query.strip("/").split("/")
        matches = []
        for node in self.main_graph.nodes():
            if self._match_pattern(node, pattern_parts, 0):
                matches.append(node)
        return matches

    def __str__(self) -> str:
        return "\n".join(sorted(nx.lexicographical_topological_sort(self.graph)))
