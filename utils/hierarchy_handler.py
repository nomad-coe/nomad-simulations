from typing import List, Optional, Protocol, Tuple
import networkx as nx


class PSection(Protocol):
    def get_name(self) -> str:
        ...

    def __lt__(self, other: "PSection") -> bool:
        ...

    def __gt__(self, other: "PSection") -> bool:
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
    def _check_branch_consistency(self, branch: str) -> bool:
        # branches are expected to be internally ordered
        # raise an error when __lt__ or __gt__ is not implemented
        pass

    def _get_extremity(self, branch: str, head: bool=True) -> str:
        pass

    def _move_along_branch(self, branch: str, up: bool=True) -> str:
        # yield the next node (name) along the branch
        pass

    def _attach_branch(self, branch_1: str, branch_2_head: str) -> str:
        # create an edge from branch_2 head to branch_1
        pass

    def _insert_branch(self, branch_1: str, branch_2_head: str) -> str:
        # remove the edge between branch_1 and [branch_1 + dn]
        # link branch_2_head to branch_1
        # link branch_2_tail to [branch_1 + dn]
        pass

    def _merge_branch(self, branch_1: str, branch_2: str) -> str:
        # compare their tail_1 / head_2 (node with no incoming/outgoing edges)
        # if head_2 < tail_1, add an edge from head_2 to tail_1
        # if not, repeat the process with head_2 and [tail_1 + 1 * up]
        # if a hit, check whether head_2 > [tail_1 + 1 * down], decide on attaching or inserting
        pass

    def _automated_construct(self) -> None:
        """Automatically constructs the temporary graph into the main graph.
        This procedure relies on the `PSection` comparison method `__lt__()` and `__gt__()`."""
        # split the branch space into 2 parts
        # stop when there is only 1 branch left
        pass

    def construct(self) -> None:
        """Constructs the temporary graph into the main graph."""
        # check if the temporary graph is a valid tree
        # migrate the temporary graph to the main graph if so
        # rely on `self._automated_construct` to do the heavy lifting
        # raise an error indicating missing links and/or cycles if not
        pass

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
