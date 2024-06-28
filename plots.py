import graphviz as gr
import numpy as np

from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.metainfo.basesections import Activity, Entity
from nomad.metainfo import Quantity, SubSection

from nomad_simulations.schema_packages.atoms_state import AtomsState, OrbitalsState
from nomad_simulations.schema_packages.general import (
    Simulation,
    Program,
    BaseSimulation,
)


# class Program(Entity):
#     name = Quantity(type=str)


# class Simulation(Activity):
#     cpu1_start = Quantity(type=np.float64, unit='second')
#     program = SubSection(Program.m_def, repeat=False)


# simulation = Simulation()
# print(m_package)

# m_package.all_definitions.get('Program').quantities[0].type

nomad_simulations_classes = {
    # 'BaseSimulation': BaseSimulation(),
    'Simulation': Simulation(),
    # 'Program': Program(),
}


def format_quantities(class_def):
    quantities = class_def.all_quantities
    formatted_quantities = []
    for name, quantity in quantities.items():
        #     q_type = quantity.type.__name__
        #     q_unit = quantity.unit if quantity.unit else ''
        #     formatted_quantities.append(f"{name}: {q_type}, {q_unit}")
        formatted_quantities.append(f'{name}')
    return '\\n'.join(formatted_quantities)


def generate_inheritance_tree(class_def, graph: gr.Digraph, added_edges: set):
    prev_name = ''
    for i, inh in enumerate(class_def.inherited_sections):
        node_properties = {
            'name': inh.name,
            'shape': 'box',
            'fontname': 'Titillium-Web',
            'style': 'filled',
        }
        if inh.name in nomad_simulations_classes.keys():
            node_properties.update({'fontcolor': 'white', 'fillcolor': '#0097a7'})
        else:
            node_properties.update({'fontcolor': 'black', 'fillcolor': 'white'})
        graph.node(**node_properties)

        if i > 0:
            if inh.name == 'ArchiveSection':
                prev_name = inh.name
                continue
            edge = (inh.name, prev_name)
            if edge not in added_edges:
                graph.edge(inh.name, prev_name, arrowhead='empty')
                added_edges.add(edge)

        # sub-sections
        for subsection in class_def.sub_sections:
            graph.edge()

        prev_name = inh.name
    return graph


for name, section in nomad_simulations_classes.items():
    graph = gr.Digraph(
        name=f'{name.lower()}-uml-diagram', comment=f'{name} UML diagram', format='pdf'
    )
    graph.attr(rankdir='BT')
    # Set to keep track of added edges to avoid duplication
    added_edges = set()
    graph = generate_inheritance_tree(
        class_def=section.m_def, graph=graph, added_edges=added_edges
    )
    if name == 'Simulation':
        graph.edge(name, 'BaseSimulation', arrowhead='empty')
    print(graph)
    graph.render(directory='umlgraphs').replace('\\', '/')
