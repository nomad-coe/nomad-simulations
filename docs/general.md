# `Simulation` base section

<!--
Improve these paragraphs once `Program` and `BaseSimulation` are integrated in `basesections.py`
--->
In NOMAD, all the simulation metadata is defined in the `Simulation` section. You can find its Python schema definition in [src/nomad_simulations/general.py](https://github.com/nomad-coe/nomad-simulations/blob/develop/src/nomad_simulations/general.py). This section will appear under the `data` section for the [*archive*](https://nomad-lab.eu/prod/v1/staging/docs/reference/glossary.html#archive) metadata structure of each [*entry*](https://nomad-lab.eu/prod/v1/staging/docs/reference/glossary.html#entry).

The `Simulation` section inherits from a _base section_ `BaseSimulation`. In NOMAD, a set of [base sections](https://nomad-lab.eu/prod/v1/staging/docs/howto/customization/base_sections.html) derived from the [Basic Formal Ontology (BFO)](https://basic-formal-ontology.org/) are defined. We used them to define `BaseSimulation` as an [`Activity`](http://purl.obolibrary.org/obo/BFO_0000015). The UML diagram is:

<div class="click-zoom">
    <label>
        <input type="checkbox">
        <img src="../assets/simulation_base.png" alt="Simulation base section diagram." width="80%" title="Click to zoom in">
    </label>
</div>

`BaseSimulation` contains the general information about the `Program` used, as well as general times of the simulation, e.g., the datetime at which it started (`datetime`) and ended (`datetime_end`). `Simulation` contains further information about the specific input and output sections ([see below](#sub-sections-in-simulation)) The detailed UML diagram of quantities and functions defined for `Simulation` is thus:

<div class="click-zoom">
    <label>
        <input type="checkbox">
        <img src="../assets/simulation.png" alt="Simulation quantities and functions UML diagram." width="50%" title="Click to zoom in">
    </label>
</div>

??? question "Notation for the section attributes in the UML diagram"
    We included the information of each attributes / quantities after its definition. The notation is:

        <name-of-quantity>: <type-of-quantity>, <units-of-quantity>

    Thus, `cpu1_start: np.float64, s` means that there is a quantity named `'cpu1_start'` of type `numpy.float64` and whose units are `'s'` (seconds).
    We also include the existance of sub-sections by bolding the name, i.e.:

        <name-of-sub-section>: <sub-section-definition>

    E.g., there is a sub-section under `Simulation` named `'model_method'` whose section defintion can be found in the `ModelMethod` section. We will represent this sub-section containment in more complex UML diagrams in the future using the containment arrow (see below for [an example using `Program`](#program)).

We use double inheritance from `EntryData` in order to populate the `data` section in the NOMAD archive. All of the base sections discussed here are subject to the [public normalize function](normalize.md) in NOMAD. The private function `set_system_branch_depth()` is related with the [ModelSystem base section](model_system/model_system.md).

## Main sub-sections in `Simulation` {#sub-sections-in-simulation}

The `Simulation` base section is composed of 4 main sub-sections:

1. `Program`: contains all the program information, e.g., `name` of the program, `version`, etc.
2. `ModelSystem`: contains all the system information about geometrical positions of atoms, their states, simulation cells, symmetry information, etc.
3. `ModelMethod`: contains all the methodological information, and it is divided in two main aspects: the mathematical model or approximation used in the simulation (e.g., `DFT`, `GW`, `ForceFields`, etc.) and the numerical settings used to compute the properties (e.g., meshes, self-consistent parameters, basis sets settings, etc.).
4. `Outputs`: contains all the output properties, as well as references to the `ModelSystem` used to obtain such properties. It might also contain information which will populate `ModelSystem` (e.g., atomic occupations, atomic moments, crystal field energies, etc.).

!!! note "Self-consistent steps, SinglePoint entries, and more complex workflows."
    The minimal unit for storing data in the NOMAD archive is an [*entry*](https://nomad-lab.eu/prod/v1/staging/docs/reference/glossary.html#entry). In the context of simulation data, an entry may contain data from a calculation on an individual system configuration (e.g., a single-point DFT calculation) using **only** the above-mentioned sections of the `Simulation` section. Information from self-consistent iterations to converge properties for this configuration are also contained within these sections.

    More complex calculations that involve multiple configurations require the definition of a *workflow* section within the archive. Depending on the situation, the information from individual workflow steps may be stored within a single or multiple entries. For example, for efficiency, the data from workflows involving a large amount of configurations, e.g., molecular dynamics trajectories, are stored within a single entry. Other standard workflows store the single-point data in separate entries, e.g.,  a `GW` calculation is composed of a `DFT SinglePoint` entry and a `GW SinglePoint` entry. Higher-level workflows, which simply connect a series of standard or custom workflows, are typically stored as a separate entry. You can check the [NOMAD simulations workflow schema](https://github.com/nomad-coe/nomad-schema-plugin-simulation-workflow) for more information.

The following schematic represents a simplified representation of the `Simulation` section (note that the arrows here are a simple way of visually defining _inputs_ and _outputs_):

<div class="click-zoom">
    <label>
        <input type="checkbox">
        <img src="../assets/simulation_composition.png" alt="Simulation composition diagram." width="90%" title="Click to zoom in">
    </label>
</div>

### `Program` {#program}

The `Program` base section contains all the information about the program / software / code used to perform the simulation. We consider it to be a [`(Continuant) Entity`](http://purl.obolibrary.org/obo/BFO_0000002) and contained within `BaseSimulation` as a sub-section. The detailed UML diagram is:

<div class="click-zoom">
    <label>
        <input type="checkbox">
        <img src="../assets/program.png" alt="Program quantities and functions UML diagram." width="75%" title="Click to zoom in">
    </label>
</div>


When [writing a parser](https://nomad-lab.eu/prod/v1/staging/docs/howto/customization/parsers.html), we recommend to start by instantiating the `Program` section and populating its quantities, in order to get acquainted with the NOMAD parsing infrastructure.

For example, imagine we have a file which we want to parse with the following information:
```txt
! * * * * * * *
! Welcome to SUPERCODE, version 7.0
...
```

We can parse the program `name` and `version` by matching the texts (see, e.g., [Wikipedia page for Regular expressions, also called _regex_](https://en.wikipedia.org/wiki/Regular_expression)):

```python
from nomad.parsing.file_parser import TextParser, Quantity
from nomad_simulations import Simulation, Program


class SUPERCODEParser:
    """
    Class responsible to populate the NOMAD `archive` from the files given by a
    SUPERCODE simulation.
    """

    def parse(self, filepath, archive, logger):
        output_parser = TextParser(
            quantities=[
                Quantity('program_version', r'version *([\d\.]+) *', repeats=False)
            ]
        )
        output_parser.mainfile = filepath

        simulation = Simulation()
        simulation.program = Program(
            name='SUPERCODE',
            version=output_parser.get('program_version'),
        )
        # append `Simulation` as an `archive.data` section
        archive.data.append(simulation)
```
