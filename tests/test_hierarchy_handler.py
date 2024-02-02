from copy import deepcopy
import pytest
import networkx as nx
from ..utils.hierarchy_handler import HierarchyFactory, PSection


class MockSection(PSection):
    def __init__(self, name: str, formula: str):
        self.name = name
        self.formula = formula

    def get_name(self) -> str:
        return self.name

    def __lt__(self, other: "MockSection") -> bool:
        return self.name < other.name

    def __gt__(self, other: "MockSection") -> bool:
        return self.name > other.name


@pytest.fixture
def mock_hierarchy() -> HierarchyFactory:
    """Forbidden construction: only to be used for testing construction mechanisms.
    The testing system we construct is:
    - polymer
    -- monomer
    - solvent
    -- (possible extension to solvent molecules)
    """
    graph = HierarchyFactory()
    root_name = graph._delimiter
    graph.main_graph.add_node("polymer", section=MockSection("", "C20H42"))
    graph.main_graph.add_node("monomer", section=MockSection("", "CH2"))
    graph.main_graph.add_node("solvent", section=MockSection("", "H100O50"))
    graph.main_graph.add_edge("polymer", root_name)
    graph.main_graph.add_edge("monomer", "polymer")
    graph.main_graph.add_edge("solvent", root_name)


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/polymer", nx.DiGraph().add_edges_from([("polymer", "/")])),
        ("/solvent", nx.DiGraph().add_edges_from([("solvent", "/")])),
        ("/polymer/monomer", nx.DiGraph().add_edges_from([("polymer", "/")])),
    ],
)
def test_path_to_graph(mock_hierarchy, path: str, expected: list):
    assert mock_hierarchy._seg_to_graph(mock_hierarchy._split_path(path)) == expected


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/polymer", "C20H42"),
        ("/solvent", "H100O50"),
    ],
)
def test_get_section(mock_hierarchy, path: str, expected: str):
    assert mock_hierarchy.get_section(path).formula == expected


@pytest.mark.parametrize(
    "parent_path, name",
    [
        ("/solvent", "water"),
    ],
)
def test_add_path(mock_hierarchy, parent_path, name: str):
    mock_copy = deepcopy(mock_hierarchy)
    mock_copy.add_to_path(parent_path, MockSection(name, "H2O"))
    assert name in mock_copy.main_graph.nodes[name]
