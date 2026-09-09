<!-- generated-by: Claude Opus 4.8 | last_updated: 2026-09-09 -->
# runschema → nomad-simulations schema mapping

This document maps every attribute of the legacy **runschema** (`Run` / `System` / `Method` /
`Calculation`) onto the modern **nomad-simulations** schema (`Simulation` → `Program` /
`ModelSystem` / `ModelMethod` / `Outputs`). It is the schema-to-schema bridge that lets the
legacy parser coverage reports (which target runschema) be compared against the new simulation
parsers (which target nomad-simulations).

Direction is runschema → nomad-simulations: the left column is the legacy attribute path, the
right column its modern equivalent. **Status** is one of:

- **Mapped** — a direct 1:1 equivalent exists.
- **Partial** — an equivalent exists but is renamed, restructured, or lossy (see Notes).
- **Unmapped** — no nomad-simulations equivalent yet; the *Unmapped rows are a coverage-gap
  audit* of the modern schema.

**Overall:** 189 mapped, 210 partial, 185 unmapped of 584 attributes
(32.4% mapped, 68.3% mapped-or-partial).

This mapping was authored by reading both schema sources; each section was self-checked so that
every left path exists in runschema and every cited target exists in nomad-simulations. It is a
review-based reference, not a machine-generated artifact — treat the Partial/Unmapped judgements
as a starting point for schema-coverage discussion.


## Run → Simulation

**Summary:** 11 mapped, 3 partial, 6 unmapped (of 20; 55.00% mapped).

| runschema path | Status | nomad-simulations target | Notes |
| --- | --- | --- | --- |
| `run.calculation_file_uri` | Unmapped | — | raw-file URI bookkeeping; no equivalent on `Simulation` |
| `run.clean_end` | Mapped | `Simulation.finished_without_errors` | inherited from `BaseSimulation`; same boolean semantics (proper termination) |
| `run.raw_id` | Partial | `Simulation.lab_id` | `lab_id` (inherited from `BaseSection`) is the nearest external-id slot; not a semantic exact match |
| `run.starting_run_ref` | Unmapped | — | run-to-run reference for chained runs; no cross-simulation reference field in `general.py` |
| `run.n_references` | Unmapped | — | counter for `runs_ref`; no equivalent |
| `run.runs_ref` | Unmapped | — | links to other `Run` sections; no analogous simulation-to-simulation reference |
| `run.program` | Mapped | `Simulation.program` | `SubSection` of `Program`; inherited from `BaseSimulation` (`repeats=False`) |
| `run.program.name` | Mapped | `Simulation.program.name` | |
| `run.program.version` | Mapped | `Simulation.program.version` | |
| `run.program.version_internal` | Mapped | `Simulation.program.version_internal` | |
| `run.program.compilation_datetime` | Unmapped | — | modern `Program` has no compilation datetime field |
| `run.program.compilation_host` | Mapped | `Simulation.program.compilation_host` | |
| `run.time_run` | Partial | `Simulation` (self) | no dedicated `TimeRun` sub-section; time fields are flattened onto `Simulation` via `SimulationTime` mixin and `BaseSection.datetime` |
| `run.time_run.date_start` | Partial | `Simulation.datetime` | `datetime` (Datetime, inherited from `BaseSection`) vs legacy Unix-epoch float; nearest start-of-run timestamp |
| `run.time_run.date_end` | Mapped | `Simulation.datetime_end` | inherited from `SimulationTime`; legacy is float seconds, modern is `Datetime` (type change) |
| `run.time_run.cpu1_start` | Mapped | `Simulation.cpu1_start` | inherited from `SimulationTime`; same unit (second) |
| `run.time_run.cpu1_end` | Mapped | `Simulation.cpu1_end` | inherited from `SimulationTime`; same unit (second) |
| `run.time_run.wall_start` | Mapped | `Simulation.wall_start` | inherited from `SimulationTime`; same unit (second) |
| `run.time_run.wall_end` | Mapped | `Simulation.wall_end` | inherited from `SimulationTime`; same unit (second) |
| `run.message` | Unmapped | — | `MessageRun` sub-section (type/value) has no equivalent; `Program.warnings` covers only warnings, not general run messages |

## System → ModelSystem

**Summary:** 31 mapped, 14 partial, 28 unmapped (of 73; 42.47% mapped).

| runschema path | Status | nomad-simulations target | Notes |
| --- | --- | --- | --- |
| `system.name` | Mapped | `ModelSystem.name` | |
| `system.type` | Mapped | `ModelSystem.type` | modern is an `MEnum` (atom, bulk, surface, 2D, ...); runschema is free `str` |
| `system.configuration_raw_gid` | Unmapped | — | geometry checksum; no modern equivalent (modern uses `HashedPositions` internally, not a stored quantity) |
| `system.is_representative` | Mapped | `ModelSystem.is_representative` | |
| `system.n_references` | Unmapped | — | reference-bookkeeping quantity dropped in modern schema |
| `system.sub_system_ref` | Partial | `ModelSystem.sub_systems[]` | modern nests actual sub-systems (SectionProxy) instead of storing a `Reference` |
| `system.systems_ref` | Unmapped | — | cross-system reference links (e.g. supercell↔unit-cell) have no modern quantity |
| `system.atoms` | Partial | `ModelSystem` (self) + `ModelSystem.particle_states[]` | `Atoms` sub-section flattened: cell/positions inherited from `Representation`, per-atom data in `particle_states[]` |
| `system.atoms.n_atoms` | Partial | `ModelSystem.n_particles` | modern renames to particle-generic; also derivable from `len(positions)`/`len(particle_states)` |
| `system.atoms.atomic_numbers` | Mapped | `ModelSystem.particle_states[].atomic_number` (`AtomsState.atomic_number`) | per-atom array → per-particle scalar on each `AtomsState` |
| `system.atoms.equivalent_atoms` | Mapped | `ModelSystem.local_symmetry.equivalent_atoms` (`LocalCrystalSymmetry`/`LocalSymmetry.equivalent_atoms`) | moved to per-representation `local_symmetry` |
| `system.atoms.wyckoff_letters` | Mapped | `ModelSystem.local_symmetry.wyckoff_letters` (`LocalCrystalSymmetry.wyckoff_letters`) | |
| `system.atoms.concentrations` | Unmapped | — | alloy site concentrations have no modern equivalent |
| `system.atoms.species` | Partial | `ModelSystem.particle_states[].atomic_number` | no dedicated `species` quantity; species identity carried by `AtomsState.atomic_number`/`chemical_symbol` |
| `system.atoms.labels` | Mapped | `ModelSystem.particle_states[].chemical_symbol` (`AtomsState.chemical_symbol`); site tokens → `AtomsState.label` | chemical labels → `chemical_symbol`; non-symbol site names → `label` |
| `system.atoms.positions` | Mapped | `ModelSystem.positions` | positions held at `ModelSystem` top level, not in a representation |
| `system.atoms.velocities` | Mapped | `ModelSystem.velocities` | |
| `system.atoms.lattice_vectors` | Mapped | `ModelSystem.lattice_vectors` (`Representation.lattice_vectors`) | |
| `system.atoms.lattice_vectors_reciprocal` | Unmapped | — | no reciprocal-lattice quantity in modern schema |
| `system.atoms.local_rotations` | Unmapped | — | per-atom orientation matrices dropped |
| `system.atoms.periodic` | Mapped | `ModelSystem.periodic_boundary_conditions` (`Representation.periodic_boundary_conditions`) | |
| `system.atoms.supercell_matrix` | Mapped | `ModelSystem.representations[].supercell_matrix` (`AlternativeRepresentation.supercell_matrix`) | now lives on the alternative representation |
| `system.atoms.symmorphic` | Unmapped | — | no `symmorphic` flag in modern symmetry sections |
| `system.atoms.bond_list` | Mapped | `ModelSystem.bond_list` | |
| `system.atoms_group` | Partial | `ModelSystem.sub_systems[]` | molecule/fragment grouping restructured into recursive `sub_systems` + `branch_label`/`branch_depth`/`particle_indices` |
| `system.atoms_group.label` | Mapped | `ModelSystem.sub_systems[].branch_label` | |
| `system.atoms_group.type` | Partial | `ModelSystem.sub_systems[].type` | modern `type` is a constrained `MEnum`; group `type` was free `str` |
| `system.atoms_group.index` | Partial | `ModelSystem.sub_systems[].branch_depth` | not an exact match: `branch_depth` is tree depth, runschema `index` is index within parent group |
| `system.atoms_group.composition_formula` | Mapped | `ModelSystem.sub_systems[].composition_formula` | |
| `system.atoms_group.n_atoms` | Partial | `ModelSystem.sub_systems[].n_particles` | particle-generic rename; also derivable from `len(particle_indices)` |
| `system.atoms_group.atom_indices` | Mapped | `ModelSystem.sub_systems[].particle_indices` | |
| `system.atoms_group.is_molecule` | Unmapped | — | no stored quantity; modern computes it via `ModelSystem.is_molecule()` method |
| `system.atoms_group.bond_list` | Mapped | `ModelSystem.sub_systems[].bond_list` | |
| `system.atoms_group.atoms_group` | Partial | `ModelSystem.sub_systems[].sub_systems[]` | recursive nesting preserved via `SectionProxy('ModelSystem')` |
| `system.chemical_composition` | Unmapped | — | full (non-reduced) composition string; modern `ChemicalFormula` has no plain-composition field (`descriptive` is closest but not equivalent) |
| `system.chemical_composition_hill` | Mapped | `ModelSystem.chemical_formula.hill` (`ChemicalFormula.hill`) | |
| `system.chemical_composition_reduced` | Mapped | `ModelSystem.chemical_formula.reduced` (`ChemicalFormula.reduced`) | |
| `system.chemical_composition_anonymous` | Mapped | `ModelSystem.chemical_formula.anonymous` (`ChemicalFormula.anonymous`) | |
| `system.constraint` | Unmapped | — | `Constraint` sub-section not ported to nomad-simulations |
| `system.constraint.kind` | Unmapped | — | no `Constraint` section in modern schema |
| `system.constraint.n_constraints` | Unmapped | — | no `Constraint` section in modern schema |
| `system.constraint.n_atoms` | Unmapped | — | no `Constraint` section in modern schema |
| `system.constraint.atom_indices` | Unmapped | — | no `Constraint` section in modern schema |
| `system.constraint.parameters` | Unmapped | — | no `Constraint` section in modern schema |
| `system.prototype` | Partial | `ModelSystem.symmetry` (`GlobalCrystalSymmetry`) | prototype info folded into the crystal-symmetry section (see fields below) |
| `system.prototype.aflow_id` | Mapped | `ModelSystem.symmetry.prototype_aflow_id` (`GlobalCrystalSymmetry.prototype_aflow_id`) | |
| `system.prototype.aflow_url` | Unmapped | — | no AFLOW URL quantity in modern schema |
| `system.prototype.assignment_method` | Unmapped | — | prototype assignment-method provenance dropped |
| `system.prototype.label` | Partial | `ModelSystem.symmetry.strukturbericht_designation` / `prototype_formula` | runschema `<sg>-<name>-<Pearson>` label has no exact modern field; closest are strukturbericht + prototype_formula |
| `system.springer_material` | Unmapped | — | `SpringerMaterial` classification section not ported |
| `system.springer_material.id` | Unmapped | — | no `SpringerMaterial` section in modern schema |
| `system.springer_material.alphabetical_formula` | Unmapped | — | no `SpringerMaterial` section in modern schema |
| `system.springer_material.url` | Unmapped | — | no `SpringerMaterial` section in modern schema |
| `system.springer_material.compound_class` | Unmapped | — | no `SpringerMaterial` section in modern schema |
| `system.springer_material.classification` | Unmapped | — | no `SpringerMaterial` section in modern schema |
| `system.symmetry` | Mapped | `ModelSystem.symmetry` (`GlobalSymmetry` base; instantiated as `GlobalCrystalSymmetry`) | repeating `Symmetry` sub-section → single `symmetry` sub-section |
| `system.symmetry.bravais_lattice` | Partial | `ModelSystem.symmetry.bravais_lattice` (property) + `lattice_type` + `lattice_centering` | Pearson string split into `lattice_type`/`lattice_centering` MEnums; `bravais_lattice` is a reconstructed read-only property |
| `system.symmetry.choice` | Unmapped | — | spglib centering/origin choice string has no modern quantity |
| `system.symmetry.crystal_system` | Partial | `ModelSystem.symmetry.lattice_type` (`GlobalCrystalSymmetry.lattice_type`) | crystal family encoded in `lattice_type` MEnum (e.g. 'c - cubic'); no standalone `crystal_system` |
| `system.symmetry.hall_number` | Mapped | `ModelSystem.symmetry.hall_number` (`GlobalCrystalSymmetry.hall_number`) | |
| `system.symmetry.hall_symbol` | Mapped | `ModelSystem.symmetry.hall_symbol` (`GlobalCrystalSymmetry.hall_symbol`) | |
| `system.symmetry.international_short_symbol` | Mapped | `ModelSystem.symmetry.space_group_symbol` (`GlobalCrystalSymmetry.space_group_symbol`) | renamed; IUC short symbol → `space_group_symbol` |
| `system.symmetry.origin_shift` | Mapped | `ModelSystem.symmetry.analysis_origin_shift` (`GlobalCrystalSymmetry.analysis_origin_shift`) | standardization origin shift from spglib |
| `system.symmetry.point_group` | Mapped | `ModelSystem.symmetry.point_group_symbol` (`GlobalCrystalSymmetry.point_group_symbol`) | |
| `system.symmetry.space_group_number` | Mapped | `ModelSystem.symmetry.space_group_number` (`GlobalCrystalSymmetry.space_group_number`) | |
| `system.symmetry.symmetry_method` | Unmapped | — | symmetry-source provenance flag dropped |
| `system.symmetry.transformation_matrix` | Mapped | `ModelSystem.symmetry.analysis_transformation_matrix` (`GlobalCrystalSymmetry.analysis_transformation_matrix`) | standardization transform matrix from spglib |
| `system.symmetry.system_original` | Partial | `ModelSystem` (original/top-level representation) | original cell is the `ModelSystem` itself, not a nested `Atoms` sub-section |
| `system.symmetry.system_primitive` | Mapped | `ModelSystem.representations[]` (name='primitive', `AlternativeRepresentation`) | primitive cell added to `representations` during normalization |
| `system.symmetry.system_std` | Mapped | `ModelSystem.representations[]` (name='conventional', `AlternativeRepresentation`) | standardized/conventional cell added to `representations` |
| `system.descriptors` | Unmapped | — | `Descriptors` (SOAP/MACE) section not ported to nomad-simulations |
| `system.descriptors.soap` | Unmapped | — | no `SOAP` descriptor section in modern schema |
| `system.descriptors.mace` | Unmapped | — | no `MACE` descriptor section in modern schema |

## Method (core) → ModelMethod

**Summary:** 60 mapped, 73 partial, 38 unmapped (of 171; 35.09% mapped).

| runschema path | Status | nomad-simulations target | Notes |
| --- | --- | --- | --- |
| `method.label` | Partial | `ModelMethod.name` | Legacy MEnum(DFT/TB/GW/DMFT/BSE/kMC/NMR); modern `name` is free str typically holding these standard names. |
| `method.stress_tensor_method` | Unmapped | — | No stress-tensor-method quantity in nomad-simulations method sections. |
| `method.starting_method_ref` | Unmapped | — | No inter-method reference quantity (starting/core/methods_ref) in `ModelMethod`. |
| `method.core_method_ref` | Unmapped | — | No inter-method reference quantity in `ModelMethod`. |
| `method.n_references` | Unmapped | — | Reference-bookkeeping quantity dropped. |
| `method.methods_ref` | Unmapped | — | No `methods_ref` array in `ModelMethod`. |
| `method.dft` | Mapped | `DFT` (ModelMethodElectronic) | `Method.dft` subsection → `DFT` model section (repeats in a `Simulation.model_method[]` list). |
| `method.dft.self_interaction_correction_method` | Partial | `SelfInteractionCorrection.method` | Modern SIC is a `BaseModelMethod` subclass (contribution), not a str on DFT; different MEnum set. |
| `method.dft.xc_functional` | Mapped | `DFT.xc` (`XCFunctional`) | Single (non-repeating) `xc` subsection. |
| `method.dft.xc_functional.name` | Partial | `XCFunctional.functional_key` | Legacy inherits `Model.name`; modern `functional_key` is the canonical alias. |
| `method.dft.xc_functional.reference` | Partial | `BaseModelMethod.external_reference` | Legacy inherits `Model.reference` (str); modern `external_reference` (URL) lives on the parent model, not `XCFunctional` (an `ArchiveSection`). |
| `method.dft.xc_functional.exchange` | Partial | `XCFunctional.components[]` | Merged into single `components[]` list; `XCComponent.kind='exchange'`. |
| `method.dft.xc_functional.exchange.name` | Partial | `XCFunctional.components[].canonical_label` | LibXC label; `display_name` for human-readable. |
| `method.dft.xc_functional.exchange.parameters` | Unmapped | — | Free-form dict; modern uses structured hybrid params (`fraction_exact_exchange`, `range_separation_parameter`), no generic dict. |
| `method.dft.xc_functional.exchange.weight` | Mapped | `XCFunctional.components[].weight` | |
| `method.dft.xc_functional.correlation` | Partial | `XCFunctional.components[]` | `XCComponent.kind='correlation'`. |
| `method.dft.xc_functional.correlation.name` | Partial | `XCFunctional.components[].canonical_label` | |
| `method.dft.xc_functional.correlation.parameters` | Unmapped | — | No generic parameters dict. |
| `method.dft.xc_functional.correlation.weight` | Mapped | `XCFunctional.components[].weight` | |
| `method.dft.xc_functional.hybrid` | Partial | `XCFunctional.components[]` | `kind` MEnum has no 'hybrid'; hybrid captured via `family=hybrid-*` and `fraction_exact_exchange`/`global_exact_exchange`. |
| `method.dft.xc_functional.hybrid.name` | Partial | `XCFunctional.components[].canonical_label` | |
| `method.dft.xc_functional.hybrid.parameters` | Partial | `XCFunctional.global_exact_exchange` / `XCComponent.fraction_exact_exchange` | Legacy stored `exact_exchange_mixing_factor` in dict; modern has structured fields but drops arbitrary params. |
| `method.dft.xc_functional.hybrid.weight` | Mapped | `XCFunctional.components[].weight` | |
| `method.dft.xc_functional.contributions` | Partial | `XCFunctional.components[]` | Legacy generic contributions of `Functional`; modern `XCComponent` (kind may be 'xc'/'k'). |
| `method.dft.xc_functional.contributions.name` | Partial | `XCFunctional.components[].canonical_label` | |
| `method.dft.xc_functional.contributions.parameters` | Unmapped | — | No generic parameters dict. |
| `method.dft.xc_functional.contributions.weight` | Mapped | `XCFunctional.components[].weight` | |
| `method.k_mesh` | Mapped | `KSpace.k_mesh[]` (`KMesh`) | Modern `KMesh` sits under `numerical_settings[KSpace].k_mesh[]`, not directly on the model. |
| `method.k_mesh.dimensionality` | Mapped | `KMesh.dimensionality` | Inherited from `Mesh`. |
| `method.k_mesh.sampling_method` | Partial | `KMesh.center` + `Mesh.spacing` + `Mesh.quadrature` | Legacy single MEnum split across three modern quantities (Gamma/MP → `center`; Equidistant/Log/Tan → `spacing`; Gauss-* → `quadrature`). |
| `method.k_mesh.n_points` | Mapped | `KMesh.n_points` | Inherited from `Mesh`. |
| `method.k_mesh.grid` | Mapped | `KMesh.grid` | Inherited from `Mesh`. |
| `method.k_mesh.points` | Mapped | `KMesh.points` | Legacy complex128; modern float64. |
| `method.k_mesh.multiplicities` | Mapped | `KMesh.multiplicities` | |
| `method.k_mesh.weights` | Mapped | `KMesh.weights` | |
| `method.k_mesh.offset` | Mapped | `KMesh.offset` | |
| `method.k_mesh.all_points` | Mapped | `KMesh.all_points` | |
| `method.k_mesh.high_symmetry_points` | Partial | `KMesh.high_symmetry_points` | Legacy str[] of labels; modern JSON dict {label: coords}. |
| `method.k_mesh.line_path_segments` | Partial | `KSpace.k_line_path` (`KLinePath`) | Restructured: legacy `LinePathSegment[]` → single `KLinePath` with high_symmetry_path names/values. |
| `method.k_mesh.line_path_segments.start_point` | Partial | `KLinePath.high_symmetry_path_names[]` | Per-segment start/end folded into an ordered name list. |
| `method.k_mesh.line_path_segments.end_point` | Partial | `KLinePath.high_symmetry_path_names[]` | Idem. |
| `method.k_mesh.line_path_segments.n_points` | Partial | `KLinePath.n_line_points` | Whole-path count, not per-segment. |
| `method.k_mesh.line_path_segments.points` | Partial | `KLinePath.points` | Whole-path points. |
| `method.electronic` | Partial | `ModelMethodElectronic` | No dedicated `Electronic` container; fields spread across model/system/output sections. |
| `method.electronic.spin_target` | Unmapped | — | Target spin multiplicity not represented on `ModelMethodElectronic` (see `AtomsState.spin` for per-atom S, not a target). |
| `method.electronic.charge` | Unmapped | — | Total electronic charge not on `ModelMethodElectronic`; `AtomsState.charge` is per-atom formal charge. |
| `method.electronic.n_bands` | Unmapped | — | Band count not stored on method; belongs to outputs. |
| `method.electronic.n_spin_channels` | Partial | `ModelMethodElectronic.is_spin_polarized` | Modern stores boolean spin polarization, not channel count. |
| `method.electronic.n_electrons` | Unmapped | — | No electron-count quantity on `ModelMethodElectronic`. |
| `method.electronic.method` | Partial | `ModelMethod.name` / `type` | Free-form method identifier; overlaps with model name/type. |
| `method.electronic.relativity_method` | Partial | `RelativityModel.level` / `.approximation` | Modern relativity is a `BaseModelMethod` subclass with MEnum `level`/`approximation`, not a str on Electronic. |
| `method.electronic.van_der_waals_method` | Partial | `EmpiricalDispersionModel.model` | Modern dispersion/vdW is a `BaseModelMethod` subclass (contribution), not a str. |
| `method.electronic.smearing` | Mapped | `Smearing` (NumericalSettings) | Modern `Smearing` under `numerical_settings[]`. |
| `method.electronic.smearing.kind` | Partial | `Smearing.name` | Modern MEnum(Fermi-Dirac/Gaussian/Methfessel-Paxton); legacy free str with more values. |
| `method.electronic.smearing.width` | Unmapped | — | No smearing width quantity on modern `Smearing`. |
| `method.scf` | Mapped | `SelfConsistency` (NumericalSettings) | Under `numerical_settings[]`. |
| `method.scf.native_tier` | Unmapped | — | No code-specific precision tier on modern `SelfConsistency` (cf. `BasisSetContainer.native_tier`). |
| `method.scf.n_max_iteration` | Mapped | `SelfConsistency.n_max_iterations` | |
| `method.scf.threshold_energy_change` | Partial | `SelfConsistency.threshold_change` | Modern unifies energy/density thresholds into one flexible-unit `threshold_change`. |
| `method.scf.threshold_density_change` | Partial | `SelfConsistency.threshold_change` | Same unified quantity as energy change. |
| `method.scf.minimization_algorithm` | Mapped | `SelfConsistency.scf_minimization_algorithm` | |
| `method.atom_parameters` | Partial | `AtomsState` (+ `Pseudopotential`, `ParticleParameters`) | Legacy per-kind method container split across `AtomsState` (intrinsic), `Pseudopotential`/`ParticleParameters` (numerical settings). |
| `method.atom_parameters.atom_number` | Mapped | `AtomsState.atomic_number` | |
| `method.atom_parameters.atom_index` | Unmapped | — | No atom-index back-reference on `AtomsState`. |
| `method.atom_parameters.n_valence_electrons` | Partial | `Pseudopotential.n_valence_electrons` | Only meaningful when a pseudopotential is present. |
| `method.atom_parameters.n_core_electrons` | Unmapped | — | No core-electron count quantity in nomad-simulations. |
| `method.atom_parameters.label` | Mapped | `AtomsState.label` | |
| `method.atom_parameters.mass` | Partial | `ParticleState.mass` / `ParticleParameters.effective_mass` | Intrinsic mass on `ParticleState`; FF-adjusted mass on `ParticleParameters`. |
| `method.atom_parameters.pseudopotential_name` | Partial | `Pseudopotential.name` | Legacy deprecated flat str; modern on the `Pseudopotential` section. |
| `method.atom_parameters.pseudopotential` | Mapped | `Pseudopotential` (NumericalSettings) | |
| `method.atom_parameters.pseudopotential.name` | Mapped | `Pseudopotential.name` | |
| `method.atom_parameters.pseudopotential.type` | Partial | `Pseudopotential.type` | Legacy MEnum('US V','US MBK','PAW'); modern MEnum('NC','US','PAW','NC-PAW','NC-PAW-GW') — different taxonomy. |
| `method.atom_parameters.pseudopotential.norm_conserving` | Mapped | `Pseudopotential.is_norm_conserving` | |
| `method.atom_parameters.pseudopotential.cutoff` | Partial | `Pseudopotential.cutoffs[].value` (`PPCutoff`) | Modern uses repeating `PPCutoff` with `cutoff_kind`/`cutoff_role` context. |
| `method.atom_parameters.pseudopotential.xc_functional_name` | Partial | `Pseudopotential.xc_functional` (`XCFunctional`) | Legacy str[]; modern a full `XCFunctional` subsection (`functional_key`/components). |
| `method.atom_parameters.pseudopotential.l_max` | Mapped | `Pseudopotential.l_max` | |
| `method.atom_parameters.pseudopotential.lm_max` | Mapped | `Pseudopotential.lm_max` | |
| `method.atom_parameters.core_hole` | Partial | `AtomsState.electronic_state` → `CoreHole` | Modern `CoreHole` extends `ElectronicState`, referenced via `electronic_state`. |
| `method.atom_parameters.core_hole.n_quantum_number` | Mapped | `SphericalSymmetryState.n_quantum_number` | Via `ElectronicState.spin_orbit_state`. |
| `method.atom_parameters.core_hole.l_quantum_number` | Mapped | `SphericalSymmetryState.l_quantum_number` | |
| `method.atom_parameters.core_hole.ml_quantum_number` | Mapped | `SphericalSymmetryState.ml_quantum_number` | |
| `method.atom_parameters.core_hole.j_quantum_number` | Mapped | `SphericalSymmetryState.j_quantum_number` | |
| `method.atom_parameters.core_hole.mj_quantum_number` | Mapped | `SphericalSymmetryState.mj_quantum_number` | |
| `method.atom_parameters.core_hole.ms_quantum_bool` | Partial | `SphericalSymmetryState.ms_quantum_number` | Legacy bool; modern signed value (`s_quantum_number`/`ms_quantum_number`). |
| `method.atom_parameters.core_hole.degeneracy` | Mapped | `ElectronicState.degeneracy` | |
| `method.atom_parameters.core_hole.n_electrons_excited` | Partial | `CoreHole.n_excited_electrons` | Renamed. |
| `method.atom_parameters.core_hole.occupation` | Mapped | `ElectronicState.occupation` | |
| `method.atom_parameters.core_hole.dscf_state` | Mapped | `CoreHole.dscf_state` | |
| `method.atom_parameters.n_orbitals` | Unmapped | — | No active-orbital-count quantity on `AtomsState`. |
| `method.atom_parameters.orbitals` | Unmapped | — | Active-orbital labels not represented (cf. TB `orbitals_ref`, out of scope). |
| `method.atom_parameters.onsite_energies` | Unmapped | — | Per-orbital onsite energies not on `AtomsState` (TB-specific, out of scope). |
| `method.atom_parameters.charge` | Partial | `AtomsState.charge` / `ParticleParameters.partial_charge` | Legacy total atom charge (coulomb); modern formal integer charge or FF partial charge. |
| `method.atom_parameters.charges` | Unmapped | — | Per-orbital charge array not represented. |
| `method.atom_parameters.hubbard_kanamori_model` | Unmapped | — | Hubbard/many-body handled by the correlated-methods agent (`HubbardInteractions`). |
| `method.molecule_parameters` | Unmapped | — | No `MoleculeParameters` container in nomad-simulations. |
| `method.molecule_parameters.label` | Unmapped | — | No equivalent. |
| `method.molecule_parameters.n_atoms` | Unmapped | — | No equivalent. |
| `method.molecule_parameters.atom_parameters` | Unmapped | — | No molecule-grouped atom parameters. |
| `method.electrons_representation` | Mapped | `BasisSetContainer` (NumericalSettings) | Under `numerical_settings[]`. |
| `method.electrons_representation.native_tier` | Mapped | `BasisSetContainer.native_tier` | |
| `method.electrons_representation.type` | Partial | `BasisSetContainer` derived name / component classes | Legacy MEnum of container types; modern derives name from component classes (PlaneWave/AtomCentered/MuffinTin), no single `type` MEnum. |
| `method.electrons_representation.scope` | Partial | `BasisSetComponent.hamiltonian_scope` | Legacy free str[]; modern a reference to `BaseModelMethod`. |
| `method.electrons_representation.basis_set` | Mapped | `BasisSetContainer.basis_set_components[]` (`BasisSetComponent`) | |
| `method.electrons_representation.basis_set.type` | Partial | component subclass (PlaneWaveBasisSet / AtomCenteredBasisSet / …) | Legacy MEnum; modern encodes type via subclass identity. |
| `method.electrons_representation.basis_set.scope` | Partial | `BasisSetComponent.hamiltonian_scope` | str[] → reference. |
| `method.electrons_representation.basis_set.cutoff` | Mapped | `PlaneWaveBasisSet.cutoff_energy` | |
| `method.electrons_representation.basis_set.cutoff_fractional` | Mapped | `APWPlaneWaveBasisSet.cutoff_fractional` | |
| `method.electrons_representation.basis_set.frozen_core` | Partial | `FrozenCore` (NumericalSettings) | Modern separate `FrozenCore` section, not a bool on the basis set. |
| `method.electrons_representation.basis_set.spherical_harmonics_cutoff` | Partial | `MuffinTinRegion.l_max` | Modern per-muffin-tin `l_max`. |
| `method.electrons_representation.basis_set.atom_parameters` | Partial | `BasisSetComponent.species_scope` | Legacy reference to `AtomParameters`; modern reference to `AtomsState`. |
| `method.electrons_representation.basis_set.atom_centered` | Partial | `AtomCenteredBasisSet` | Restructured (component subclass with `functional_compositions`/`atomic_orbitals`). |
| `method.electrons_representation.basis_set.atom_centered.name` | Mapped | `AtomCenteredBasisSet.basis_set` | Basis set family name. |
| `method.electrons_representation.basis_set.atom_centered.formula` | Partial | `AtomCenteredBasisSet.canonical_basis_set` | Generalized/canonical name. |
| `method.electrons_representation.basis_set.atom_centered.atom_number` | Partial | `BasisSetComponent.species_scope` → `AtomsState.atomic_number` | Via species reference, not a direct int. |
| `method.electrons_representation.basis_set.atom_centered.n_basis_functions` | Mapped | `AtomCenteredBasisSet.n_total_basis_functions` | |
| `method.electrons_representation.basis_set.atom_centered.gaussian_basis_group` | Partial | `AtomCenteredFunction` (via `functional_compositions`) | Restructured contraction representation. |
| `method.electrons_representation.basis_set.atom_centered.gaussian_basis_group.n_contractions` | Unmapped | — | No explicit contraction count; implicit in `AtomCenteredFunction` arrays. |
| `method.electrons_representation.basis_set.atom_centered.gaussian_basis_group.n_exponents` | Partial | `AtomCenteredFunction.n_primitive` | Number of primitives ≈ exponents. |
| `method.electrons_representation.basis_set.atom_centered.gaussian_basis_group.contractions` | Partial | `AtomCenteredFunction.contraction_coefficients` | Shape/grouping differs. |
| `method.electrons_representation.basis_set.atom_centered.gaussian_basis_group.exponents` | Mapped | `AtomCenteredFunction.exponents` | |
| `method.electrons_representation.basis_set.atom_centered.gaussian_basis_group.ls` | Partial | `AtomCenteredFunction.angular_momentum` | Per-function l vs legacy per-contraction array. |
| `method.electrons_representation.basis_set.orbital` | Partial | `APWLChannel.orbitals[]` (`APWOrbital`/`APWLocalOrbital`) | Restructured under `MuffinTinRegion.l_channels`. |
| `method.electrons_representation.basis_set.orbital.type` | Partial | `APWOrbital.type` | Different MEnum grouping (APW/LAPW vs APW/LAPW/SLAPW split across subclasses). |
| `method.electrons_representation.basis_set.orbital.n_quantum_number` | Unmapped | — | Not retained on APW orbital sections. |
| `method.electrons_representation.basis_set.orbital.l_quantum_number` | Partial | `APWLChannel.name` | l encoded on the channel, not the orbital. |
| `method.electrons_representation.basis_set.orbital.j_quantum_number` | Unmapped | — | Not retained on APW orbital sections. |
| `method.electrons_representation.basis_set.orbital.kappa_quantum_number` | Unmapped | — | Not retained. |
| `method.electrons_representation.basis_set.orbital.occupation` | Unmapped | — | Not retained on APW orbital sections. |
| `method.electrons_representation.basis_set.orbital.core_level` | Unmapped | — | Not retained. |
| `method.electrons_representation.basis_set.orbital.energy_parameter` | Mapped | `APWBaseOrbital.energy_parameter` | |
| `method.electrons_representation.basis_set.orbital.energy_parameter_n` | Mapped | `APWBaseOrbital.energy_parameter_n` | |
| `method.electrons_representation.basis_set.orbital.order` | Partial | `APWBaseOrbital.differential_order` | Renamed. |
| `method.electrons_representation.basis_set.orbital.boundary_condition_order` | Unmapped | — | No boundary-condition-order quantity on APW orbital sections. |
| `method.electrons_representation.basis_set.orbital.update` | Partial | `APWBaseOrbital.energy_status` | Update/updated flags folded into an `energy_status` MEnum. |
| `method.electrons_representation.basis_set.orbital.updated` | Partial | `APWBaseOrbital.energy_status` | Same `energy_status`. |
| `method.electrons_representation.basis_set.shape` | Partial | `MuffinTinRegion` / `Mesh` geometry | Legacy `BasisSetMesh.shape` MEnum; modern mesh geometry implicit in component subclass. |
| `method.electrons_representation.basis_set.box_lengths` | Unmapped | — | Basis-set-mesh box geometry not represented. |
| `method.electrons_representation.basis_set.radius` | Partial | `MuffinTinRegion.radius` | Only for muffin-tin case. |
| `method.electrons_representation.basis_set.grid_spacing` | Partial | `Mesh.spacing` | Legacy metric spacing array; modern `spacing` is an MEnum type, not values. |
| `method.electrons_representation.basis_set.radius_lin_spacing` | Unmapped | — | No radial-grid linear spacing quantity. |
| `method.electrons_representation.basis_set.radius_log_spacing` | Unmapped | — | No radial-grid log spacing quantity. |
| `method.electrons_representation.basis_set.n_grid_points` | Partial | `Mesh.n_points` | Generic mesh point count. |
| `method.electrons_representation.basis_set.n_radial_grid_points` | Unmapped | — | No dedicated radial-grid point count. |
| `method.electrons_representation.basis_set.n_spherical_grid_points` | Unmapped | — | No dedicated spherical-grid point count. |
| `method.force_field` | Mapped | `ForceField` (ModelMethod) | Modern `ForceField` is a top-level model section. |
| `method.force_field.model` | Partial | `ForceField.contributions[]` (`Potential`) | Legacy `Model` container of `Interaction`; modern flattens to `Potential` contributions. |
| `method.force_field.model.name` | Partial | `ForceField.name` / `Potential.name` | `Model.name` → model or potential name. |
| `method.force_field.model.reference` | Partial | `ForceField.external_reference` / `.kimid` | str → URL (`external_reference`) or OpenKIM `kimid`. |
| `method.force_field.model.contributions` | Mapped | `ForceField.contributions[]` (`Potential`) | Interaction terms. |
| `method.force_field.model.contributions.type` | Partial | `Potential.type` | Legacy free str; modern MEnum(bond/angle/dihedral/…). |
| `method.force_field.model.contributions.name` | Mapped | `Potential.name` | |
| `method.force_field.model.contributions.n_interactions` | Mapped | `Potential.n_interactions` | |
| `method.force_field.model.contributions.n_atoms` | Partial | `Potential.n_particles` | Renamed (atoms → particles). |
| `method.force_field.model.contributions.atom_labels` | Partial | `Potential.particle_labels` | Renamed. |
| `method.force_field.model.contributions.atom_indices` | Partial | `Potential.particle_indices` | Renamed. |
| `method.force_field.model.contributions.functional_form` | Mapped | `Potential.functional_form` | |
| `method.force_field.model.contributions.n_parameters` | Unmapped | — | No parameter-count quantity; modern `parameters` is a `ParameterEntry[]`. |
| `method.force_field.model.contributions.parameters` | Partial | `Potential.parameters[]` (`ParameterEntry`) | Legacy dict → structured name/value/unit entries. |
| `method.force_field.model.contributions.contributions` | Mapped | `Potential.contributions[]` (`BaseModelMethod`) | Nested contributions via inherited `contributions`. |
| `method.force_field.force_calculations` | Mapped | `ForceCalculations` (NumericalSettings) | |
| `method.force_field.force_calculations.vdw_cutoff` | Mapped | `ForceCalculations.vdw_cutoff` | |
| `method.force_field.force_calculations.coulomb_type` | Mapped | `ForceCalculations.coulomb_type` | Same MEnum. |
| `method.force_field.force_calculations.coulomb_cutoff` | Mapped | `ForceCalculations.coulomb_cutoff` | |
| `method.force_field.force_calculations.neighbor_searching` | Partial | `ForceCalculations` (fields inlined) | Legacy `NeighborSearching` subsection; modern inlines its fields onto `ForceCalculations`. |
| `method.force_field.force_calculations.neighbor_searching.neighbor_update_frequency` | Mapped | `ForceCalculations.neighbor_update_frequency` | Inlined. |
| `method.force_field.force_calculations.neighbor_searching.neighbor_update_cutoff` | Mapped | `ForceCalculations.neighbor_update_cutoff` | Inlined. |
| `method.photon` | Partial | `Photon` (ArchiveSection) | Modern `Photon` exists but is used by excited-state methods (BSE), out of core scope. |
| `method.photon.multipole_type` | Mapped | `Photon.multipole_type` | |
| `method.photon.polarization` | Mapped | `Photon.polarization` | |
| `method.photon.energy` | Mapped | `Photon.energy` | |
| `method.photon.momentum_transfer` | Mapped | `Photon.momentum_transfer` | |

## Method (correlated / beyond-DFT) → ModelMethod

**Summary:** 51 mapped, 21 partial, 39 unmapped (of 111; 45.95% mapped).

| runschema path | Status | nomad-simulations target | Notes |
| --- | --- | --- | --- |
| `method.hubbard_kanamori_model.orbital` | Partial | `ModelMethod.contributions[HubbardInteractions].orbitals_ref` | Legacy is a single orbital-label string; modern is a list of `ElectronicState` references. |
| `method.hubbard_kanamori_model.n_orbital` | Mapped | `ModelMethod.contributions[HubbardInteractions].n_orbitals` | |
| `method.hubbard_kanamori_model.u` | Mapped | `ModelMethod.contributions[HubbardInteractions].u_interaction` | |
| `method.hubbard_kanamori_model.jh` | Mapped | `ModelMethod.contributions[HubbardInteractions].j_hunds_coupling` | |
| `method.hubbard_kanamori_model.up` | Mapped | `ModelMethod.contributions[HubbardInteractions].u_interorbital_interaction` | |
| `method.hubbard_kanamori_model.j` | Mapped | `ModelMethod.contributions[HubbardInteractions].j_local_exchange_interaction` | Legacy `j` (exchange) → `j_local_exchange_interaction`. |
| `method.hubbard_kanamori_model.u_effective` | Mapped | `ModelMethod.contributions[HubbardInteractions].u_effective` | |
| `method.hubbard_kanamori_model.slater_integrals` | Mapped | `ModelMethod.contributions[HubbardInteractions].slater_integrals` | |
| `method.hubbard_kanamori_model.umn` | Mapped | `ModelMethod.contributions[HubbardInteractions].u_matrix` | Local Coulomb interaction matrix. |
| `method.hubbard_kanamori_model.double_counting_correction` | Mapped | `ModelMethod.contributions[HubbardInteractions].double_counting_correction` | |
| `method.tb.slater_koster` | Partial | `SlaterKoster` (TB subclass) | Legacy is a SubSection; modern is a `TB` subclass, not a nested SubSection. |
| `method.tb.xtb` | Partial | `xTB` (TB subclass) | Legacy SubSection; modern is a `TB` subclass. |
| `method.tb.wannier` | Partial | `Wannier` (TB subclass) | Legacy SubSection; modern is a `TB` subclass. |
| `method.tb.slater_koster.orbitals` | Partial | `SlaterKoster.bonds[SlaterKosterBond].orbital_1/orbital_2` | Legacy holds a list of `TightBindingOrbital`; modern references `ElectronicState` from within bonds; no standalone orbital list. |
| `method.tb.slater_koster.bonds` | Mapped | `SlaterKoster.bonds[SlaterKosterBond]` | |
| `method.tb.slater_koster.overlaps` | Mapped | `SlaterKoster.overlaps[SlaterKosterBond]` | |
| `method.tb.slater_koster.bonds.bond_label` | Partial | `SlaterKoster.bonds[SlaterKosterBond].name` | Modern `name` is an MEnum ('sss','sps','sds') resolved from orbital references, not a free string. |
| `method.tb.slater_koster.bonds.center1` | Partial | `SlaterKoster.bonds[SlaterKosterBond].orbital_1` | Legacy nests a `TightBindingOrbital`; modern is an `ElectronicState` reference. |
| `method.tb.slater_koster.bonds.center2` | Partial | `SlaterKoster.bonds[SlaterKosterBond].orbital_2` | Legacy nests a `TightBindingOrbital`; modern is an `ElectronicState` reference. |
| `method.tb.slater_koster.bonds.sss` | Partial | `SlaterKoster.bonds[SlaterKosterBond].integral_value` | Modern collapses all per-type SK integrals (sss/sps/…/fff) into one `integral_value` + `name`. |
| `method.tb.slater_koster.bonds.sps` | Partial | `SlaterKoster.bonds[SlaterKosterBond].integral_value` | See `sss`. |
| `method.tb.slater_koster.bonds.sds` | Partial | `SlaterKoster.bonds[SlaterKosterBond].integral_value` | See `sss`. |
| `method.tb.slater_koster.bonds.sfs` | Unmapped | — | No `sfs` bond type in modern SK MEnum; `integral_value`/`name` do not cover f-integrals. |
| `method.tb.slater_koster.bonds.pps` | Unmapped | — | Modern SK `name` MEnum only covers sss/sps/sds. |
| `method.tb.slater_koster.bonds.ppp` | Unmapped | — | See `pps`. |
| `method.tb.slater_koster.bonds.pds` | Unmapped | — | See `pps`. |
| `method.tb.slater_koster.bonds.pdp` | Unmapped | — | See `pps`. |
| `method.tb.slater_koster.bonds.pfs` | Unmapped | — | See `pps`. |
| `method.tb.slater_koster.bonds.pfp` | Unmapped | — | See `pps`. |
| `method.tb.slater_koster.bonds.dds` | Unmapped | — | See `pps`. |
| `method.tb.slater_koster.bonds.ddp` | Unmapped | — | See `pps`. |
| `method.tb.slater_koster.bonds.ddd` | Unmapped | — | See `pps`. |
| `method.tb.slater_koster.bonds.dfs` | Unmapped | — | See `pps`. |
| `method.tb.slater_koster.bonds.dfp` | Unmapped | — | See `pps`. |
| `method.tb.slater_koster.bonds.dfd` | Unmapped | — | See `pps`. |
| `method.tb.slater_koster.bonds.ffs` | Unmapped | — | See `pps`. |
| `method.tb.slater_koster.bonds.ffp` | Unmapped | — | See `pps`. |
| `method.tb.slater_koster.bonds.ffd` | Unmapped | — | See `pps`. |
| `method.tb.slater_koster.bonds.fff` | Unmapped | — | See `pps`. |
| `method.tb.xtb.hamiltonian` | Unmapped | — | Modern `xTB(TB)` has no `hamiltonian`/`overlap`/`repulsion`/`magnetic`/`coulomb` `Interaction` subsections; the legacy `xTB(Model)` term structure was dropped. |
| `method.tb.xtb.overlap` | Unmapped | — | See `method.tb.xtb.hamiltonian`. |
| `method.tb.xtb.repulsion` | Unmapped | — | See `method.tb.xtb.hamiltonian`. |
| `method.tb.xtb.magnetic` | Unmapped | — | See `method.tb.xtb.hamiltonian`. |
| `method.tb.xtb.coulomb` | Unmapped | — | See `method.tb.xtb.hamiltonian`. |
| `method.tb.wannier.n_projected_orbitals` | Partial | `Wannier.n_orbitals_per_atom` (inherited from `TB`) | Modern has no dedicated Wannier projected-orbital count; closest is TB's per-atom orbital count. |
| `method.tb.wannier.n_bands` | Mapped | `Wannier.n_bloch_bands` | |
| `method.tb.wannier.is_maximally_localized` | Mapped | `Wannier.is_maximally_localized` | Modern also derives `localization_type` from this. |
| `method.tb.wannier.convergence_tolerance_max_localization` | Unmapped | — | No convergence-tolerance quantity on modern `Wannier`. |
| `method.tb.wannier.energy_window_outer` | Mapped | `Wannier.energy_window_outer` | |
| `method.tb.wannier.energy_window_inner` | Mapped | `Wannier.energy_window_inner` | |
| `method.lattice_model_hamiltonian.hubbard_kanamori_model` | Mapped | `ModelMethod.contributions[HubbardInteractions]` | Same target as `method.hubbard_kanamori_model`. |
| `method.core_hole.solver` | Unmapped | — | Modern `CoreHoleSpectra` `solver` is commented out; no equivalent quantity. |
| `method.core_hole.edge` | Mapped | `CoreHoleSpectra.edge` | Same MEnum edge labels; modern resolves from `core_hole_ref`. |
| `method.core_hole.mode` | Mapped | `CoreHoleSpectra.type` | Legacy `mode` ('absorption'/'emission') → modern `type`. |
| `method.core_hole.broadening` | Partial | `ExcitedStateMethodology.broadening` (via `CoreHoleSpectra` inheritance chain) | Modern `CoreHoleSpectra(ModelMethodElectronic)` does not inherit `ExcitedStateMethodology`, so no `broadening`; nearest is generic excited-state broadening on other sections. |
| `method.photon.multipole_type` | Partial | `Photon.multipole_type` | Legacy is free `str`; modern is MEnum ('dipolar','quadrupolar','NRIXS','Raman'). |
| `method.photon.polarization` | Mapped | `Photon.polarization` | |
| `method.photon.energy` | Mapped | `Photon.energy` | |
| `method.photon.momentum_transfer` | Mapped | `Photon.momentum_transfer` | |
| `method.gw.type` | Mapped | `GW.type` | Same MEnum values. |
| `method.gw.analytical_continuation` | Mapped | `GW.analytical_continuation` | Same MEnum values. |
| `method.gw.interval_qp_corrections` | Mapped | `GW.interval_qp_corrections` | |
| `method.gw.screening` | Partial | `GW.screening_ref` | Legacy nests a `Screening` SubSection; modern holds a `Screening` reference instead. |
| `method.gw.n_states` | Mapped | `GW.n_states` (from `ExcitedStateMethodology`) | |
| `method.gw.n_empty_states` | Mapped | `GW.n_empty_states` (from `ExcitedStateMethodology`) | |
| `method.gw.broadening` | Mapped | `GW.broadening` (from `ExcitedStateMethodology`) | |
| `method.gw.type (ExcitedStateMethodology.type)` | Unmapped | — | Base `ExcitedStateMethodology.type` (generic string) is overridden by `GW.type`; no separate generic type slot in modern. |
| `method.gw.k_mesh` | Partial | `BaseModelMethod.numerical_settings[KSpace.k_mesh[KMesh]]` | Modern excited-state classes carry no `k_mesh` SubSection; k-meshes live under `numerical_settings[KSpace]`. |
| `method.gw.q_mesh` | Unmapped | — | No `q_mesh` on modern `ExcitedStateMethodology`/`GW`; `KSpace` models only `k_mesh`/`k_line_path`. |
| `method.gw.frequency_mesh` | Unmapped | — | `FrequencyMesh` is not implemented in nomad-simulations (only named in a docstring enum). |
| `method.bse.type` | Mapped | `BSE.type` | Same MEnum values. |
| `method.bse.solver` | Mapped | `BSE.solver` | Same MEnum values. |
| `method.bse.screening` | Partial | `BSE.screening_ref` | Legacy nests `Screening`; modern holds a `Screening` reference. |
| `method.bse.core_hole` | Partial | `CoreHoleSpectra` | Legacy nests `CoreHoleSpectra` under BSE; modern `CoreHoleSpectra` is a standalone `ModelMethodElectronic` linked via `excited_state_method_ref`, not a BSE SubSection. |
| `method.bse.n_states` | Mapped | `BSE.n_states` (from `ExcitedStateMethodology`) | |
| `method.bse.n_empty_states` | Mapped | `BSE.n_empty_states` (from `ExcitedStateMethodology`) | |
| `method.bse.broadening` | Mapped | `BSE.broadening` (from `ExcitedStateMethodology`) | |
| `method.bse.k_mesh` | Partial | `BaseModelMethod.numerical_settings[KSpace.k_mesh[KMesh]]` | See `method.gw.k_mesh`. |
| `method.bse.q_mesh` | Unmapped | — | See `method.gw.q_mesh`. |
| `method.bse.frequency_mesh` | Unmapped | — | `FrequencyMesh` not implemented in nomad-simulations. |
| `method.dmft.n_impurities` | Mapped | `DMFT.n_impurities` | |
| `method.dmft.n_correlated_orbitals` | Mapped | `DMFT.n_orbitals` | Renamed; shape `['n_impurities']`. |
| `method.dmft.n_electrons` | Mapped | `DMFT.n_electrons` | |
| `method.dmft.inverse_temperature` | Mapped | `DMFT.inverse_temperature` | |
| `method.dmft.magnetic_state` | Mapped | `DMFT.magnetic_state` | Same MEnum values. |
| `method.dmft.impurity_solver` | Mapped | `DMFT.impurity_solver` | Same MEnum values. |
| `method.gw.screening.dielectric_infinity` | Mapped | `Screening.dielectric_infinity` | |
| `method.gw.screening.n_states` | Mapped | `Screening.n_states` (from `ExcitedStateMethodology`) | |
| `method.gw.screening.n_empty_states` | Mapped | `Screening.n_empty_states` (from `ExcitedStateMethodology`) | |
| `method.gw.screening.broadening` | Mapped | `Screening.broadening` (from `ExcitedStateMethodology`) | |
| `method.gw.screening.type` | Unmapped | — | Legacy `ExcitedStateMethodology.type` (generic string) has no slot on modern `Screening`/`ExcitedStateMethodology`. |
| `method.gw.screening.k_mesh` | Partial | `BaseModelMethod.numerical_settings[KSpace.k_mesh[KMesh]]` | See `method.gw.k_mesh`. |
| `method.gw.screening.q_mesh` | Unmapped | — | See `method.gw.q_mesh`. |
| `method.gw.screening.frequency_mesh` | Unmapped | — | `FrequencyMesh` not implemented. |
| `method.frequency_mesh` | Unmapped | — | `FrequencyMesh` not implemented in nomad-simulations. |
| `method.time_mesh` | Unmapped | — | `TimeMesh` not implemented in nomad-simulations. |
| `method.force_field.force_calculations.neighbor_searching.neighbor_update_frequency` | Unmapped | — | MD force-calculation neighbor settings have no equivalent in `nomad_simulations` `force_field.py`/`model_method.py`. |
| `method.force_field.force_calculations.neighbor_searching.neighbor_update_cutoff` | Unmapped | — | See above. |
| `method.force_field.force_calculations.vdw_cutoff` | Unmapped | — | No `ForceCalculations` equivalent in nomad-simulations. |
| `method.force_field.force_calculations.coulomb_type` | Unmapped | — | See above. |
| `method.force_field.force_calculations.coulomb_cutoff` | Unmapped | — | See above. |
| `AtomParameters.core_hole.n_quantum_number` | Mapped | `atoms_state.CoreHole.spin_orbit_state[SphericalSymmetryState].n_quantum_number` | `CoreHole(SingleElectronState)` state definition. |
| `AtomParameters.core_hole.l_quantum_number` | Mapped | `atoms_state.CoreHole` → `SphericalSymmetryState.l_quantum_number` | |
| `AtomParameters.core_hole.ml_quantum_number` | Mapped | `atoms_state.CoreHole` → `SphericalSymmetryState.ml_quantum_number` | |
| `AtomParameters.core_hole.j_quantum_number` | Mapped | `atoms_state.CoreHole` → `SphericalSymmetryState.j_quantum_number` | |
| `AtomParameters.core_hole.mj_quantum_number` | Mapped | `atoms_state.CoreHole` → `SphericalSymmetryState.mj_quantum_number` | |
| `AtomParameters.core_hole.ms_quantum_bool` | Partial | `atoms_state.CoreHole` → `SphericalSymmetryState.ms_quantum_number` | Legacy is a bool (up/down); modern is a float ms value (semantic change). |
| `AtomParameters.core_hole.degeneracy` | Mapped | `atoms_state.CoreHole.degeneracy` (from `ElectronicState`) | |
| `AtomParameters.core_hole.n_electrons_excited` | Mapped | `atoms_state.CoreHole.n_excited_electrons` | Renamed. |
| `AtomParameters.core_hole.occupation` | Mapped | `atoms_state.CoreHole.occupation` (from `ElectronicState`) | |
| `AtomParameters.core_hole.dscf_state` | Mapped | `atoms_state.CoreHole.dscf_state` | Same MEnum ('initial','final'). |

## Calculation (energetics) → Outputs

**Summary:** 12 mapped, 59 partial, 31 unmapped (of 102; 11.76% mapped).

| runschema path | Status | nomad-simulations target | Notes |
| --- | --- | --- | --- |
| `calculation.energy.total.value` | Mapped | `Outputs.total_energies[].value` | `TotalEnergy.value` (joule), from `BaseEnergy` |
| `calculation.energy.total.reference` | Unmapped | — | `EnergyEntry.reference` (code-dependent offset); no equivalent on `BaseEnergy` |
| `calculation.energy.total.value_per_atom` | Unmapped | — | per-atom normalization not modeled on `TotalEnergy` |
| `calculation.energy.total.values_per_atom` | Unmapped | — | atom-resolved energies not modeled |
| `calculation.energy.total.potential` | Partial | `Outputs.potential_energies[].value` | scalar sub-field of `EnergyEntry`; maps to separate `PotentialEnergy` property, not a named field on `TotalEnergy` |
| `calculation.energy.total.kinetic` | Partial | `Outputs.kinetic_energies[].value` | scalar sub-field of `EnergyEntry`; maps to separate `KineticEnergy` property |
| `calculation.energy.total.correction` | Partial | `Outputs.total_energies[].contributions[].value` | generic `BaseEnergy` contribution, no named field |
| `calculation.energy.total.short_range` | Partial | `Outputs.total_energies[].contributions[].value` | generic contribution, no named field |
| `calculation.energy.total.long_range` | Partial | `Outputs.total_energies[].contributions[].value` | generic contribution, no named field |
| `calculation.energy.total.kind` | Unmapped | — | `Atomic.kind` label; no equivalent |
| `calculation.energy.total.n_orbitals` | Unmapped | — | `Atomic.n_orbitals`; no equivalent |
| `calculation.energy.total.n_atoms` | Unmapped | — | `Atomic.n_atoms`; no equivalent |
| `calculation.energy.total.n_spin_channels` | Unmapped | — | `Atomic.n_spin_channels`; energy props carry no `spin_channel` |
| `calculation.energy.current.value` | Partial | `Outputs.total_energies[].value` | perturbative "current" energy; representable as a labeled `TotalEnergy`, no dedicated field |
| `calculation.energy.zero_point.value` | Partial | `Outputs.total_energies[].contributions[].value` | zero-point vibration energy; generic contribution, no named field |
| `calculation.energy.kinetic_electronic.value` | Partial | `Outputs.kinetic_energies[].value` | electronic kinetic energy; maps to `KineticEnergy` property |
| `calculation.energy.electronic.value` | Partial | `Outputs.total_energies[].contributions[].value` | generic contribution, no named electronic field |
| `calculation.energy.correlation.value` | Partial | `Outputs.total_energies[].contributions[].value` | generic contribution, no named correlation field |
| `calculation.energy.exchange.value` | Partial | `Outputs.total_energies[].contributions[].value` | generic contribution, no named exchange field |
| `calculation.energy.xc.value` | Partial | `Outputs.total_energies[].contributions[].value` | generic contribution, no named xc field |
| `calculation.energy.xc_potential.value` | Partial | `Outputs.total_energies[].contributions[].value` | generic contribution, no named field |
| `calculation.energy.electrostatic.value` | Partial | `Outputs.total_energies[].contributions[].value` | generic contribution, no named field |
| `calculation.energy.nuclear_repulsion.value` | Partial | `Outputs.total_energies[].contributions[].value` | generic contribution, no named field |
| `calculation.energy.coulomb.value` | Partial | `Outputs.total_energies[].contributions[].value` | generic contribution, no named field |
| `calculation.energy.madelung.value` | Partial | `Outputs.total_energies[].contributions[].value` | generic contribution, no named field |
| `calculation.energy.ewald.value` | Partial | `Outputs.total_energies[].contributions[].value` | generic contribution, no named field |
| `calculation.energy.free.value` | Partial | `Outputs.total_energies[].contributions[].value` | free energy; representable as labeled `TotalEnergy` or contribution, no dedicated field (`HelmholtzFreeEnergy` exists in thermodynamics but is not wired into `Outputs`) |
| `calculation.energy.sum_eigenvalues.value` | Partial | `Outputs.total_energies[].contributions[].value` | generic contribution, no named field |
| `calculation.energy.total_t0.value` | Partial | `Outputs.total_energies[].value` | total energy extrapolated to T=0; representable as a labeled `TotalEnergy`, no dedicated field |
| `calculation.energy.van_der_waals.value` | Partial | `Outputs.total_energies[].contributions[].value` | vdW energy; generic contribution, no named field |
| `calculation.energy.hartree_fock_x_scaled.value` | Partial | `Outputs.total_energies[].contributions[].value` | generic contribution, no named field |
| `calculation.energy.contributions[].value` | Mapped | `Outputs.total_energies[].contributions[].value` | repeating generic `EnergyEntry` → repeating `BaseEnergy` contributions |
| `calculation.energy.types[].value` | Partial | `Outputs.total_energies[].value` | repeating generic energy "types"; each representable as a labeled `TotalEnergy`, no dedicated field |
| `calculation.energy.enthalpy` | Partial | `Enthalpy.value` | `Enthalpy(BaseEnergy)` defined in `properties/thermodynamics.py` but NOT exposed as an `Outputs` subsection |
| `calculation.energy.entropy` | Partial | `Entropy.value` | `Entropy` defined in `properties/thermodynamics.py` but NOT wired into `Outputs` |
| `calculation.energy.chemical_potential` | Partial | `Outputs.chemical_potentials[].value` | `ChemicalPotential(BaseEnergy)` is an `Outputs` subsection; legacy field is a bare scalar |
| `calculation.energy.internal` | Partial | `InternalEnergy.value` | `InternalEnergy(BaseEnergy)` in `properties/thermodynamics.py` but NOT wired into `Outputs` |
| `calculation.energy.double_counting.value` | Partial | `Outputs.total_energies[].contributions[].value` | Hubbard double-counting; generic contribution, no named field |
| `calculation.energy.correction_entropy.value` | Partial | `Outputs.total_energies[].contributions[].value` | generic contribution, no named field |
| `calculation.energy.correction_hartree.value` | Partial | `Outputs.total_energies[].contributions[].value` | generic contribution, no named field |
| `calculation.energy.correction_xc.value` | Partial | `Outputs.total_energies[].contributions[].value` | generic contribution, no named field |
| `calculation.energy.change` | Partial | `Outputs.scf_steps.delta_energies_total` | change of total energy vs previous step; closest is the per-SCF delta series (not a per-calculation scalar) |
| `calculation.energy.fermi` | Unmapped | — | electronic-structure reference (Fermi level); out of scope, handled with eigenvalues/band structure |
| `calculation.energy.highest_occupied` | Unmapped | — | electronic-structure (HOMO); out of scope, see `ElectronicEigenvalues.highest_occupied` |
| `calculation.energy.lowest_unoccupied` | Unmapped | — | electronic-structure (LUMO); out of scope, see `ElectronicEigenvalues.lowest_unoccupied` |
| `calculation.energy.kinetic.value` | Mapped | `Outputs.kinetic_energies[].value` | `KineticEnergy.value` |
| `calculation.energy.potential.value` | Mapped | `Outputs.potential_energies[].value` | `PotentialEnergy.value` |
| `calculation.energy.pressure_volume_work.value` | Partial | `Work.value` | `Work(BaseEnergy)` in `properties/thermodynamics.py` but NOT wired into `Outputs` |
| `calculation.forces.total.value` | Mapped | `Outputs.total_forces[].value` | `TotalForce.value` (newton, shape `[*,*]`) |
| `calculation.forces.total.value_raw` | Unmapped | — | unfiltered forces; no `value_raw` on `BaseForce` |
| `calculation.forces.total.kind` | Unmapped | — | `Atomic.kind` label; no equivalent |
| `calculation.forces.total.n_atoms` | Unmapped | — | `Atomic.n_atoms`; no equivalent |
| `calculation.forces.total.n_orbitals` | Unmapped | — | `Atomic.n_orbitals`; no equivalent |
| `calculation.forces.total.n_spin_channels` | Unmapped | — | `Atomic.n_spin_channels`; no equivalent |
| `calculation.forces.free.value` | Partial | `Outputs.total_forces[].contributions[].value` | free-energy forces; generic `BaseForce` contribution, no named field |
| `calculation.forces.t0.value` | Partial | `Outputs.total_forces[].contributions[].value` | T=0 forces; generic contribution, no named field |
| `calculation.forces.contributions[].value` | Mapped | `Outputs.total_forces[].contributions[].value` | repeating generic `ForcesEntry` → repeating `BaseForce` contributions |
| `calculation.forces.types[].value` | Partial | `Outputs.total_forces[].contributions[].value` | repeating force "types"; representable as generic contributions, no dedicated field |
| `calculation.stress.total.value` | Unmapped | — | stress not modeled in nomad-simulations (no `Stress`/`StressEntry` class) |
| `calculation.stress.total.values_per_atom` | Unmapped | — | atom-resolved stress not modeled |
| `calculation.stress.contributions[].value` | Unmapped | — | stress not modeled |
| `calculation.stress.types[].value` | Unmapped | — | stress not modeled |
| `calculation.thermodynamics[].enthalpy` | Partial | `Enthalpy.value` | `Enthalpy(BaseEnergy)` exists in `properties/thermodynamics.py` but is NOT an `Outputs` subsection |
| `calculation.thermodynamics[].entropy` | Partial | `Entropy.value` | `Entropy` exists but NOT wired into `Outputs` |
| `calculation.thermodynamics[].chemical_potential` | Partial | `Outputs.chemical_potentials[].value` | `ChemicalPotential` is an `Outputs` subsection |
| `calculation.thermodynamics[].kinetic_energy` | Mapped | `Outputs.kinetic_energies[].value` | `KineticEnergy.value` |
| `calculation.thermodynamics[].potential_energy` | Mapped | `Outputs.potential_energies[].value` | `PotentialEnergy.value` |
| `calculation.thermodynamics[].internal_energy` | Partial | `InternalEnergy.value` | `InternalEnergy(BaseEnergy)` exists but NOT wired into `Outputs` |
| `calculation.thermodynamics[].vibrational_free_energy_at_constant_volume` | Partial | `HelmholtzFreeEnergy.value` | `HelmholtzFreeEnergy(BaseEnergy)` exists but NOT wired into `Outputs` |
| `calculation.thermodynamics[].pressure` | Partial | `Pressure.value` | `Pressure` defined in `properties/thermodynamics.py` but NOT an `Outputs` subsection |
| `calculation.thermodynamics[].temperature` | Mapped | `Outputs.temperatures[].value` | `Temperature.value` (kelvin) |
| `calculation.thermodynamics[].volume` | Partial | `Volume.value` | `Volume` defined in `properties/thermodynamics.py` but NOT an `Outputs` subsection |
| `calculation.thermodynamics[].heat_capacity_c_v` | Partial | `HeatCapacity.value` | `HeatCapacity` exists but NOT wired into `Outputs`; c_v/c_p distinction not modeled |
| `calculation.thermodynamics[].heat_capacity_c_p` | Partial | `HeatCapacity.value` | `HeatCapacity` exists but NOT wired into `Outputs`; c_v/c_p distinction not modeled |
| `calculation.thermodynamics[].time_step` | Unmapped | — | no per-thermodynamics time-step field; cf. `WorkflowOutputs.step` at calc level |
| `calculation.vibrational_frequencies[].n_frequencies` | Unmapped | — | vibrational frequencies not modeled in nomad-simulations |
| `calculation.vibrational_frequencies[].value` | Unmapped | — | vibrational frequencies not modeled |
| `calculation.vibrational_frequencies[].raman` | Unmapped | — | vibrational frequencies not modeled |
| `calculation.vibrational_frequencies[].infrared` | Unmapped | — | vibrational frequencies not modeled |
| `calculation.scf_iteration[].energy.total.value` | Partial | `Outputs.scf_steps.energies_total` | legacy stores per-iteration sub-sections; modern collapses to flat `energies_total[]` series on `SCFSteps` |
| `calculation.scf_iteration[].energy.change` | Partial | `Outputs.scf_steps.delta_energies_total` | per-SCF total-energy delta series |
| `calculation.scf_iteration[].energy.fermi` | Unmapped | — | electronic-structure reference; out of scope |
| `calculation.scf_iteration[].forces.total.value` | Partial | `Outputs.scf_steps.delta_force_abs` | modern only records absolute force change per SCF step, not full force vectors |
| `calculation.scf_iteration[].time_calculation` | Partial | `Outputs.scf_steps.durations` | per-SCF-step time in modern flat series |
| `calculation.scf_iteration[].time_physical` | Unmapped | — | no per-SCF wall-clock counterpart on `SCFSteps` |
| `calculation.scf_iteration[].stress.total.value` | Unmapped | — | stress not modeled |
| `calculation.scf_iteration[].thermodynamics[]` | Unmapped | — | per-SCF thermodynamics not modeled |
| `calculation.scf_iteration[].pressure` | Unmapped | — | per-SCF pressure not modeled on `SCFSteps` |
| `calculation.scf_iteration[].volume` | Unmapped | — | per-SCF volume not modeled on `SCFSteps` |
| `calculation.time_calculation` | Partial | `Outputs.wall_end` | wall-clock elapsed time; closest is `SimulationTime.wall_end`/`wall_start` (no dedicated duration field) |
| `calculation.time_physical` | Partial | `Outputs.wall_end` | elapsed real time; approximated by `SimulationTime.wall_end` |
| `calculation.step` | Mapped | `WorkflowOutputs.step` | step number w.r.t. workflow (`Outputs` subclass) |
| `calculation.time` | Mapped | `TrajectoryOutputs.time` | elapsed simulated physical time (`Outputs` subclass; unit ps) |
| `calculation.volume` | Partial | `Volume.value` | `Volume` property exists but NOT wired into `Outputs` |
| `calculation.density` | Partial | `MassDensity.value` | `MassDensity(PhysicalProperty)` exists but NOT wired into `Outputs` |
| `calculation.pressure` | Partial | `Pressure.value` | `Pressure` exists but NOT wired into `Outputs` |
| `calculation.pressure_tensor` | Unmapped | — | rank-2 pressure tensor not modeled |
| `calculation.virial_tensor` | Partial | `VirialTensor.value` | `VirialTensor(BaseEnergy)` (rank `[3,3]`) exists but NOT wired into `Outputs` |
| `calculation.enthalpy` | Partial | `Enthalpy.value` | `Enthalpy` exists but NOT wired into `Outputs` |
| `calculation.temperature` | Mapped | `Outputs.temperatures[].value` | `Temperature.value` |
| `calculation.hessian_matrix` | Partial | `Hessian.value` | `Hessian(PhysicalProperty)` exists (`joule/m**2`) but NOT wired into `Outputs` |
| `calculation.n_scf_iterations` | Partial | `Outputs.scf_steps.energies_total` | count is implicit in the length of the `SCFSteps` flat series; no explicit count field |

## Calculation (electronic structure) → Outputs

**Summary:** 24 mapped, 40 partial, 43 unmapped (of 107; 22.43% mapped).

| runschema path | Status | nomad-simulations target | Notes |
| --- | --- | --- | --- |
| `calculation.eigenvalues.n_spin_channels` | Partial | `Outputs.electronic_eigenvalues[].spin_channel` | Legacy stores a count; modern splits into one `ElectronicEigenvalues` per `spin_channel` (0/1), so a count becomes an index/multiplicity restructure |
| `calculation.eigenvalues.n_bands` | Partial | `Outputs.electronic_eigenvalues[].n_levels` | Renamed; modern uses `n_levels` (per-sampling-point levels). No `n_bands` on `ElectronicEigenvalues` (only on `FermiSurface`) |
| `calculation.eigenvalues.n_kpoints` | Partial | `Outputs.electronic_eigenvalues[].value` (shape) | No dedicated quantity; k-point count is encoded in the `value`/`occupation` array shape and the `KMesh.n_points` variable |
| `calculation.eigenvalues.kpoints` | Mapped | `Outputs.electronic_eigenvalues[]` variable `KMesh.points` | k-points modeled as a `Variables` subsection (`KMesh`) referencing `KMesh.points`, not a direct quantity |
| `calculation.eigenvalues.kpoints_weights` | Unmapped | — | No k-point weight quantity on `ElectronicEigenvalues`/`KMesh` |
| `calculation.eigenvalues.kpoints_multiplicities` | Unmapped | — | No k-point multiplicity quantity |
| `calculation.eigenvalues.endpoints_labels` | Unmapped | — | Segment endpoint labels; only meaningful for band-structure path (see `high_symmetry_path_names` under band structure), no eigenvalue equivalent |
| `calculation.eigenvalues.orbital_labels` | Unmapped | — | No per-band orbital-label quantity on `ElectronicEigenvalues` |
| `calculation.eigenvalues.occupations` | Mapped | `Outputs.electronic_eigenvalues[].occupation` | Renamed plural→singular; shape reorganized per spin channel |
| `calculation.eigenvalues.energies` | Mapped | `Outputs.electronic_eigenvalues[].value` | |
| `calculation.eigenvalues.qp_linearization_prefactor` | Unmapped | — | GW quasiparticle linearization prefactor; no equivalent (related `QuasiparticleWeight` is a separate DMFT property, different semantics) |
| `calculation.eigenvalues.value_xc_potential` | Partial | `Outputs.electronic_eigenvalues[].contributions[].value` (label `'KSxc'`) | GW diagonal matrix elements modeled as labeled `contributions` on `ElectronicEigenvalues`, not a distinct quantity |
| `calculation.eigenvalues.value_correlation` | Partial | `Outputs.electronic_self_energies[].value` / `electronic_eigenvalues[].contributions[]` (`'SigC'`) | GW correlation self-energy diagonal; modern stores as `ElectronicSelfEnergy` or eigenvalue contribution |
| `calculation.eigenvalues.value_exchange` | Partial | `Outputs.electronic_self_energies[].value` / `electronic_eigenvalues[].contributions[]` (`'SigX'`) | GW exchange self-energy diagonal |
| `calculation.eigenvalues.value_xc` | Partial | `Outputs.electronic_eigenvalues[].contributions[].value` | GW xc energy diagonal as labeled contribution |
| `calculation.eigenvalues.value_qp` | Partial | `Outputs.electronic_eigenvalues[].value` | GW quasi-particle energy is the primary `value` in a GW entry (KS stored as contribution `'KS'`) |
| `calculation.eigenvalues.value_ks` | Partial | `Outputs.electronic_eigenvalues[].contributions[].value` (label `'KS'`) | Kohn-Sham diagonal stored as contribution in GW entry |
| `calculation.eigenvalues.value_ks_xc` | Partial | `Outputs.electronic_eigenvalues[].contributions[].value` (`'KSxc'`) | Kohn-Sham xc diagonal as labeled contribution |
| `calculation.eigenvalues.band_gap` | Partial | `Outputs.electronic_band_gaps[]` | Legacy nests `BandGapDeprecated` under eigenvalues; modern band gaps are a top-level `Outputs` subsection derived via normalization from eigenvalue HOMO/LUMO |
| `calculation.band_structure_electronic.path_standard` | Unmapped | — | No standard-path identifier on `ElectronicBandStructure`/`KLinePath` |
| `calculation.band_structure_electronic.reciprocal_cell` | Mapped | `Outputs.electronic_band_structures[].reciprocal_cell` | Modern resolves from `KSpace.reciprocal_lattice_vectors` during normalization |
| `calculation.band_structure_electronic.band_gap` | Partial | `Outputs.electronic_band_gaps[]` | Nested `BandGapDeprecated` → top-level `ElectronicBandGap` |
| `calculation.band_structure_electronic.energy_fermi` | Partial | `Outputs.electronic_band_structures[].highest_occupied` | No `energy_fermi`; the inherited `highest_occupied` serves as the Fermi/energy reference |
| `calculation.band_structure_electronic.segment` | Partial | `Outputs.electronic_band_structures[].k_path` (`KLinePath`) | Multiple `BandEnergies` segments restructured into a single `value` + one `KLinePath` (`k_path`) |
| `calculation.band_structure_electronic.segment.energies` | Mapped | `Outputs.electronic_band_structures[].value` | Per-segment energies concatenated into the single eigenvalue `value` |
| `calculation.band_structure_electronic.segment.occupations` | Mapped | `Outputs.electronic_band_structures[].occupation` | Inherited from `ElectronicEigenvalues` |
| `calculation.band_structure_electronic.segment.kpoints` | Mapped | `Outputs.electronic_band_structures[].k_path` variable `KLinePath.points` | Segment k-points become the k-line path points |
| `calculation.band_structure_electronic.segment.endpoints_labels` | Partial | `KLinePath` numerical settings `high_symmetry_path_names` | High-symmetry labels live in `KLinePath` (KSpace numerical settings), not on the property; not a direct field on the band structure |
| `calculation.band_structure_electronic.segment.n_bands` | Partial | `Outputs.electronic_band_structures[].n_levels` | Renamed via inheritance |
| `calculation.band_structure_electronic.segment.n_kpoints` | Partial | `KLinePath` points (shape) | Encoded in `k_path`/`value` shape |
| `calculation.band_structure_electronic.segment.n_spin_channels` | Partial | `Outputs.electronic_band_structures[].spin_channel` | Count→per-channel index restructure |
| `calculation.band_structure_electronic.segment.band_gap` | Partial | `Outputs.electronic_band_gaps[]` | Nested deprecated band gap → top-level property |
| `calculation.band_gap.index` | Partial | `Outputs.electronic_band_gaps[].spin_channel` | Legacy spin-channel index → `spin_channel` |
| `calculation.band_gap.value` | Mapped | `Outputs.electronic_band_gaps[].value` | |
| `calculation.band_gap.type` | Mapped | `Outputs.electronic_band_gaps[].type` | Same `MEnum('direct','indirect')` |
| `calculation.band_gap.energy_highest_occupied` | Unmapped | — | `ElectronicBandGap` has no HOMO field; HOMO lives on `ElectronicEigenvalues.highest_occupied` and is used to derive the gap |
| `calculation.band_gap.energy_lowest_unoccupied` | Unmapped | — | `ElectronicBandGap` has no LUMO field; LUMO lives on `ElectronicEigenvalues.lowest_unoccupied` |
| `calculation.band_gap.provenance` | Partial | `Outputs.electronic_band_gaps[].physical_property_ref` / `is_derived` | `ElectronicStructureProvenance` collapses to the generic `PhysicalProperty` derivation reference |
| `calculation.dos_electronic.kind` | Unmapped | — | `Atomic.kind`; no equivalent on `ElectronicDensityOfStates` |
| `calculation.dos_electronic.n_orbitals` | Unmapped | — | `Atomic.n_orbitals`; no equivalent |
| `calculation.dos_electronic.n_atoms` | Unmapped | — | `Atomic.n_atoms`; no equivalent |
| `calculation.dos_electronic.n_spin_channels` | Partial | `Outputs.electronic_dos[].spin_channel` | Count→per-channel index; one DOS per spin channel |
| `calculation.dos_electronic.n_energies` | Partial | `Outputs.electronic_dos[].energies.n_points` | Encoded in the `Energy` variable `n_points` |
| `calculation.dos_electronic.energies` | Mapped | `Outputs.electronic_dos[].energies.points` | Modeled as `Energy` variable grid points |
| `calculation.dos_electronic.energy_fermi` | Partial | `Outputs.electronic_dos[].energies_origin` | No `energy_fermi`; `energies_origin` (derived from eigenvalue HOMO) is the closest reference |
| `calculation.dos_electronic.energy_ref` | Mapped | `Outputs.electronic_dos[].energies_origin` | Both denote the energy-axis origin (HOMO); derived during normalization in modern |
| `calculation.dos_electronic.spin_channel` | Mapped | `Outputs.electronic_dos[].spin_channel` | |
| `calculation.dos_electronic.total` | Mapped | `Outputs.electronic_dos[].value` | Total DOS is the top-level `DOSProfile.value` (modern `total` is not a repeated subsection) |
| `calculation.dos_electronic.species_projected` | Partial | `Outputs.electronic_dos[].projected_dos[].value` | Modern `projected_dos` is atom-/orbital-projected only; species-level projection has no distinct slot |
| `calculation.dos_electronic.atom_projected` | Mapped | `Outputs.electronic_dos[].projected_dos[].value` | entity_ref = ElectronicState (no spin_orbit_state), name `'atom X'` |
| `calculation.dos_electronic.orbital_projected` | Mapped | `Outputs.electronic_dos[].projected_dos[].value` | entity_ref = ElectronicState (with spin_orbit_state), name `'orbital Y X'` |
| `calculation.dos_electronic.fingerprint` | Unmapped | — | `DosFingerprint` (bins/indices/stepsize/filling_factor/grid_id) not modeled |
| `calculation.dos_electronic.band_gap` | Partial | `Outputs.electronic_band_gaps[]` | Nested deprecated band gap → top-level property (derived from DOS in normalization) |
| `calculation.charges.kind` | Unmapped | — | `Charges` (atomic charges) not modeled in nomad-simulations |
| `calculation.charges.n_orbitals` | Unmapped | — | atomic charges not modeled |
| `calculation.charges.n_atoms` | Unmapped | — | atomic charges not modeled |
| `calculation.charges.n_spin_channels` | Unmapped | — | atomic charges not modeled |
| `calculation.charges.analysis_method` | Unmapped | — | Mulliken/Hirshfeld/Bader analysis method not modeled |
| `calculation.charges.value` | Unmapped | — | atomic charges not modeled |
| `calculation.charges.n_electrons` | Unmapped | — | atomic charges not modeled |
| `calculation.charges.spins` | Unmapped | — | atomic charges not modeled |
| `calculation.charges.total` | Unmapped | — | atomic charges not modeled |
| `calculation.charges.spin_projected.value` | Unmapped | — | `ChargesValue` not modeled |
| `calculation.charges.orbital_projected.value` | Unmapped | — | `ChargesValue` not modeled |
| `calculation.multipoles.kind` | Unmapped | — | `Multipoles` not modeled in nomad-simulations |
| `calculation.multipoles.dipole.value` | Unmapped | — | multipoles not modeled |
| `calculation.multipoles.dipole.total` | Unmapped | — | multipoles not modeled |
| `calculation.multipoles.dipole.origin` | Unmapped | — | multipoles not modeled |
| `calculation.multipoles.dipole.orbital_projected.value` | Unmapped | — | multipoles not modeled |
| `calculation.multipoles.quadrupole.value` | Unmapped | — | multipoles not modeled |
| `calculation.multipoles.quadrupole.total` | Unmapped | — | multipoles not modeled |
| `calculation.multipoles.octupole.value` | Unmapped | — | multipoles not modeled |
| `calculation.multipoles.higher_order.value` | Unmapped | — | multipoles not modeled |
| `calculation.greens_functions.type` | Partial | `Outputs.electronic_greens_functions[].local_model_type` | `MEnum('impurity','lattice')` → `local_model_type` |
| `calculation.greens_functions.matsubara_freq` | Mapped | `Outputs.electronic_greens_functions[]` variable `MatsubaraFrequency.points` | Modeled as `MatsubaraFrequency` variable |
| `calculation.greens_functions.tau` | Mapped | `Outputs.electronic_greens_functions[]` variable `ImaginaryTime.points` | Modeled as `ImaginaryTime` variable |
| `calculation.greens_functions.frequencies` | Mapped | `Outputs.electronic_greens_functions[]` variable `Frequency.points` | Modeled as real `Frequency` variable |
| `calculation.greens_functions.chemical_potential` | Mapped | `Outputs.chemical_potentials[].value` | Separate `ChemicalPotential` property in `Outputs` |
| `calculation.greens_functions.self_energy_iw` | Partial | `Outputs.electronic_self_energies[].value` | Matsubara self-energy → `ElectronicSelfEnergy.value` (HDF5), `space_id='iw'`; tensor axes restructured |
| `calculation.greens_functions.greens_function_iw` | Partial | `Outputs.electronic_greens_functions[].value` | `space_id='iw'`; value is HDF5 dataset, axis layout differs |
| `calculation.greens_functions.hybridization_function_iw` | Partial | `Outputs.hybridization_functions[].value` | `space_id='iw'`; HDF5 dataset |
| `calculation.greens_functions.greens_function_tau` | Partial | `Outputs.electronic_greens_functions[].value` | `space_id='it'` (imaginary time) |
| `calculation.greens_functions.self_energy_freq` | Partial | `Outputs.electronic_self_energies[].value` | `space_id='w'` (real frequency) |
| `calculation.greens_functions.greens_function_freq` | Partial | `Outputs.electronic_greens_functions[].value` | `space_id='w'` |
| `calculation.greens_functions.hybridization_function_freq` | Partial | `Outputs.hybridization_functions[].value` | `space_id='w'` |
| `calculation.greens_functions.orbital_occupations` | Unmapped | — | Per-orbital occupation of the Green's function not modeled on `BaseGreensFunction` (related `Occupancy` is a separate atom-orbital property, different structure) |
| `calculation.greens_functions.quasiparticle_weights` | Mapped | `Outputs.quasiparticle_weights[].value` | Separate `QuasiparticleWeight` property |
| `calculation.spectra.type` | Partial | `Outputs.absorption_spectra[]` / `xas_spectra[]` | Legacy free-form type string (XAS/RIXS/XES/ARPES) → distinct modern classes; only XAS/absorption modeled |
| `calculation.spectra.n_energies` | Partial | `Outputs.absorption_spectra[].energies.n_points` | Encoded in `Energy` variable |
| `calculation.spectra.excitation_energies` | Mapped | `Outputs.absorption_spectra[].energies.points` | Modeled as `Energy` variable grid points |
| `calculation.spectra.energy_zero_ref` | Unmapped | — | No energy-zero-reference quantity on `SpectralProfile`/`AbsorptionSpectrum` |
| `calculation.spectra.intensities` | Mapped | `Outputs.absorption_spectra[].value` | Spectral profile intensities (positive) |
| `calculation.spectra.intensities_units` | Unmapped | — | No units-string quantity (units carried by the metainfo `unit`) |
| `calculation.spectra.oscillator_strengths` | Unmapped | — | No oscillator-strength quantity on `SpectralProfile` |
| `calculation.spectra.transition_dipole_moments` | Unmapped | — | No transition-dipole-moment quantity |
| `calculation.spectra.provenance` | Partial | `Outputs.absorption_spectra[].physical_property_ref` | `ElectronicStructureProvenance` collapses to generic derivation ref |
| `calculation.hopping_matrix.n_orbitals` | Mapped | `Outputs.hopping_matrices[].n_orbitals` | |
| `calculation.hopping_matrix.n_wigner_seitz_points` | Partial | `Outputs.hopping_matrices[]` variable `WignerSeitz.n_points` | Encoded in `WignerSeitz` variable |
| `calculation.hopping_matrix.degeneracy_factors` | Mapped | `Outputs.hopping_matrices[].degeneracy_factors` | |
| `calculation.hopping_matrix.value` | Partial | `Outputs.hopping_matrices[].value` | Modern `value` is an HDF5 dataset `[n_wigner_seitz_points, n_orbitals, n_orbitals]`; legacy `[n_ws, n_orb*n_orb, 7]` layout differs; onsite (0,0,0) term also feeds `CrystalFieldSplitting` |
| `calculation.potential.kind` | Unmapped | — | Volumetric `Potential` not modeled |
| `calculation.potential.effective.value` | Unmapped | — | Volumetric potential grid not modeled |
| `calculation.potential.hartree.value` | Unmapped | — | Volumetric potential grid not modeled |
| `calculation.density_charge.value` | Unmapped | — | Volumetric charge `Density` (real-space grid) not modeled |
| `calculation.density_charge.value_hdf5` | Unmapped | — | Volumetric charge density (HDF5) not modeled |
| `calculation.radius_of_gyration.kind` | Unmapped | — | `RadiusOfGyration` exists in `Outputs.radii_of_gyration` but as a `PhysicalProperty`; the `AtomicGroup.kind` field has no equivalent |
| `calculation.radius_of_gyration.radius_of_gyration_values.value` | Partial | `Outputs.radii_of_gyration[].value` | MD structural property present in modern `Outputs`, but per-atomsgroup value structure (`RadiusOfGyrationValues`, `atomsgroup_ref`, `label`) is restructured |
