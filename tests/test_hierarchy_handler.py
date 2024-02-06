from ..simulationdataschema.system import AtomicCell
from ..utils.hierarchy_handler import HierarchyFactory, PSection


class MockSection(PSection):
    def __init__(self, name: str, labels: list[str]):
        self.name = name
        self.section = AtomicCell(labels=labels)

    def get_name(self) -> str:
        return self.name

    def __lt__(self, other: "MockSection") -> bool:
        return set(self.labels) < (set(other.name))

    def __gt__(self, other: "MockSection") -> bool:
        return set(self.labels) > (set(other.name))

    def __eq__(self, other: "MockSection") -> bool:
        return set(self.labels) < (set(other.name))


def test_hierarchy_factory():
    graph = HierarchyFactory()
    graph.include_section(MockSection("polymer", ["C"]*20 + ["H"]*42), "polymer")
    graph.include_section(MockSection("monomer", ["C"] + ["H"]*2), "monomer")
    graph.include_section(MockSection("solvent", ["H"]*100 + ["O"]*50), "solvent")
    graph.construct()

    assert graph.main_graph.edges == {
        ("/", "polymer"),
        ("polymer", "monomer"),
        ("/", "solvent"),
    }
