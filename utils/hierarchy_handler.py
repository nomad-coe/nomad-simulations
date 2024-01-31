from typing import List, Optional
import networkx as nx

class HierarchyFactory:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.current_path = "/"
        self.graph.add_node(self.current_path)  # Root node

    def add_path(self, path: str) -> None:
        parts = path.strip("/").split("/")
        for i in range(len(parts)):
            parent = "/" + "/".join(parts[:i])
            child = parent + "/" + parts[i] if parent != "/" else "/" + parts[i]
            if not self.graph.has_node(child):
                self.graph.add_node(child)
            if not self.graph.has_edge(parent, child):
                self.graph.add_edge(parent, child)

    def get_node(self, path: str) -> Optional[nx.DiGraph]:
        if path in self.graph:
            descendants = nx.descendants(self.graph, path) | {path}
            return self.graph.subgraph(descendants).copy()
        else:
            print(f"Path {path} does not exist.")
            return None

    def ls(self, path: Optional[str] = None) -> List[str]:
        target_path = path or self.current_path
        if target_path in self.graph:
            return list(self.graph.successors(target_path))
        else:
            print(f"Path {target_path} does not exist.")
            return []

    def cd(self, path: str) -> None:
        if path == "/":
            self.current_path = path
            return
        elif path.startswith("/"):
            target_path = path
        else:
            target_path = f"{self.current_path}/{path}".replace("//", "/")
        
        if target_path in self.graph:
            self.current_path = target_path
        else:
            print(f"Path {target_path} does not exist.")

    def pwd(self) -> str:
        return self.current_path

    def __str__(self) -> str:
        return "\n".join(sorted(nx.lexicographical_topological_sort(self.graph)))

    def _match_pattern(self, node: str, pattern_parts: List[str], index: int) -> bool:
        """Recursively checks if the node path matches the pattern."""
        if index == len(pattern_parts):
            return True
        if pattern_parts[index] == "**":
            if index + 1 == len(pattern_parts):
                return True
            for descendant in nx.descendants(self.graph, node):
                if self._match_pattern(descendant, pattern_parts, index + 1):
                    return True
        elif pattern_parts[index] == "*":
            for successor in self.graph.successors(node):
                if self._match_pattern(successor, pattern_parts, index + 1):
                    return True
        else:
            next_node = f"{node}/{pattern_parts[index]}".replace("//", "/")
            if self.graph.has_node(next_node) and self._match_pattern(next_node, pattern_parts, index + 1):
                return True
        return False

    def find(self, query: str) -> List[str]:
        """Finds paths matching the query with support for wildcards."""
        pattern_parts = query.strip("/").split("/")
        matches = []
        for node in self.graph.nodes():
            if self._match_pattern(node, pattern_parts, 0):
                matches.append(node)
        return matches
