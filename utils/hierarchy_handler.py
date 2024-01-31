from typing import List, Optional, Protocol, Tuple
import networkx as nx


class ISectionInterface(Protocol):
    def get_name(self) -> str:
        ...

    def __lt__(self, other: "ISectionInterface") -> bool:
        ...

    def __gt__(self, other: "ISectionInterface") -> bool:
        ...


class SectionHandler(ISectionInterface):
    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        return self.name

    def __lt__(self, other: "SectionHandler") -> bool:
        return self.name < other.name

    def __gt__(self, other: "SectionHandler") -> bool:
        return self.name > other.name


class HierarchyFactory:
    def __init__(self):
        self.main_graph = nx.DiGraph()
        self.temp_graph = nx.DiGraph()
        self.current_path = "/"
        self.main_graph.add_node("/", data=[])  # Root node with empty section list

    def add_node(self, section_handler: ISectionInterface) -> None:
        """Add `ISectionInterface` to the temporary graph for later construction."""
        node_name = section_handler.get_name()
        self.temp_graph.add_node(node_name, data=section_handler)

    def add_path(self, path: str, section_handler: ISectionInterface) -> None:
        """Add `ISectionInterface` directly to the main graph under the path."""
        parts = path.strip("/").split("/")  # clean path and split into parts
        parent = "/"
        for part in parts[:-1]:
            parent = f"{parent}{part}/" if parent != "/" else f"/{part}"
            if not self.main_graph.has_node(parent):
                self.main_graph.add_node(parent, data=[])

        # Add the final part with the section_handler
        final_path = f"{parent}{parts[-1]}" if parent != "/" else f"/{parts[-1]}"
        if not self.main_graph.has_node(final_path):
            self.main_graph.add_node(final_path, data=[section_handler])
        else:
            self.main_graph.nodes[final_path]["data"].append(section_handler)

    def get_node(self, path: str) -> Optional[nx.DiGraph]:
        if path in self.graph:
            descendants = nx.descendants(self.graph, path) | {path}
            return self.graph.subgraph(descendants).copy()
        else:
            print(f"Path {path} does not exist.")
            return None

    def cd(self, path: str) -> None:
        """Change the current path to the target path.
        All further path specifications are relative to the current path."""
        if path == "/":
            self.current_path = path
            return
        elif path.startswith("/"):
            target_path = path
        else:
            target_path = f"{self.current_path}/{path}".replace("//", "/")

        if target_path in self.main_graph:
            self.current_path = target_path
        else:
            print(f"Path {target_path} does not exist.")

    def pwd(self) -> str:
        """Print the current path."""
        return self.current_path

    def ls(self, path: Optional[str] = None) -> List[str]:
        """List the contents of the target directory (relative path)."""
        target_path = path or self.current_path
        if target_path in self.main_graph:
            # List the names of sections in the target directory
            sections = [
                handler.get_name()
                for handler in self.main_graph.nodes[target_path]["data"]
            ]
            # List subdirectories
            subdirs = [
                node.rsplit("/", 1)[-1]
                for node in self.main_graph.successors(target_path)
            ]
            return sorted(sections + subdirs)
        else:
            print(f"Path {target_path} does not exist.")
            return []

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
