# NOMAD's Simulations Schema Plugin

This is a plugin for [NOMAD](https://nomad-lab.eu/nomad-lab/) which contains the base sections schema definitions for materials science simulations.

<!--Add Getting started section to install the plugin independently of NOMAD for local and Oasis usage (see e.g. https://github.com/FAIRmat-NFDI/nomad-measurements?tab=readme-ov-file)-->

## Importing this plug-in into NOMAD

### For Developers

In here we will use the term _local path_ to the schema to refer to the root of this project.
Under the default naming convention, this would be `"dependencies/schema/simulation/data"` with respect to the NOMAD root.

1. Update dependency files:
   1. `pyproject.toml`: add the local path under `[tool.setuptools.packages.find]`.
   2. `MANIFEST.in`: link the local path to this `README.md` file, as well as the `*.py` and `nomad_plugin.yaml` files in the source folder `simulationdataschema`.
   3. `.gitmodules`: add the local path and this project's [URL](https://github.com/nomad-coe/nomad-schema-plugin-simulation-data) to check out the relevant commit.
   4. `.vscode/settings.json`: add the local path under `"python.analysis.extraPaths"` to apply the suggestion tools for the VSCode IDE.
2. Run `./scripts/setup_dev_env.sh` to apply the settings in `pyproject.toml` and `MANIFEST.in`, bind the Python environment and download the necessary modules.
