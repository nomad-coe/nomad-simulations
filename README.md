![](https://github.com/nomad-coe/nomad-schema-plugin-simulation-data/actions/workflows/actions.yml/badge.svg)
![](https://coveralls.io/repos/github/nomad-coe/nomad-schema-plugin-simulation-data/badge.svg?branch=develop)
[![NOMAD](https://img.shields.io/badge/Open%20NOMAD-lightgray?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDI3LjUuMCwgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPgo8c3ZnIHZlcnNpb249IjEuMSIgaWQ9IkxheWVyXzEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4IgoJIHZpZXdCb3g9IjAgMCAxNTAwIDE1MDAiIHN0eWxlPSJlbmFibGUtYmFja2dyb3VuZDpuZXcgMCAwIDE1MDAgMTUwMDsiIHhtbDpzcGFjZT0icHJlc2VydmUiPgo8c3R5bGUgdHlwZT0idGV4dC9jc3MiPgoJLnN0MHtmaWxsOiMxOTJFODY7c3Ryb2tlOiMxOTJFODY7c3Ryb2tlLXdpZHRoOjE0MS4zMjI3O3N0cm9rZS1taXRlcmxpbWl0OjEwO30KCS5zdDF7ZmlsbDojMkE0Q0RGO3N0cm9rZTojMkE0Q0RGO3N0cm9rZS13aWR0aDoxNDEuMzIyNztzdHJva2UtbWl0ZXJsaW1pdDoxMDt9Cjwvc3R5bGU+CjxwYXRoIGNsYXNzPSJzdDAiIGQ9Ik0xMTM2LjQsNjM2LjVjMTUwLjgsMCwyNzMuMS0xMjEuOSwyNzMuMS0yNzIuMlMxMjg3LjIsOTIuMSwxMTM2LjQsOTIuMWMtMTUwLjgsMC0yNzMuMSwxMjEuOS0yNzMuMSwyNzIuMgoJUzk4NS42LDYzNi41LDExMzYuNCw2MzYuNXoiLz4KPHBhdGggY2xhc3M9InN0MSIgZD0iTTEzMjksOTQ2Yy0xMDYuNC0xMDYtMjc4LjgtMTA2LTM4Ni4xLDBjLTk5LjYsOTkuMy0yNTguNywxMDYtMzY1LjEsMTguMWMtNi43LTcuNi0xMy40LTE2LjItMjEuMS0yMy45CgljLTEwNi40LTEwNi0xMDYuNC0yNzgsMC0zODQuOWMxMDYuNC0xMDYsMTA2LjQtMjc4LDAtMzg0LjlzLTI3OC44LTEwNi0zODYuMSwwYy0xMDcuMywxMDYtMTA2LjQsMjc4LDAsMzg0LjkKCWMxMDYuNCwxMDYsMTA2LjQsMjc4LDAsMzg0LjljLTYzLjIsNjMtODkuMSwxNTAtNzYuNywyMzIuMWM3LjcsNTcuMywzMy41LDExMy43LDc3LjYsMTU3LjZjMTA2LjQsMTA2LDI3OC44LDEwNiwzODYuMSwwCgljMTA2LjQtMTA2LDI3OC44LTEwNiwzODYuMSwwYzEwNi40LDEwNiwyNzguOCwxMDYsMzg2LjEsMEMxNDM1LjQsMTIyNCwxNDM1LjQsMTA1MiwxMzI5LDk0NnoiLz4KPC9zdmc+Cg==)](https://nomad-lab.eu/prod/v1/staging/gui/)

# NOMAD's Simulations Schema Plugin
This is a plugin for [NOMAD](https://nomad-lab.eu) which contains the base sections schema definitions for materials science simulations..


<!-- MOVE THIS TO THE DOCUMENTATION PAGE OF THIS PLUGIN --->

## Getting started


### Install the dependencies

Clone the project and in the workspace folder, create a virtual environment (note this project uses Python 3.9):
```sh
git clone https://github.com/nomad-coe/nomad-schema-plugin-simulation-data.git
cd nomad-schema-plugin-simulation-data
python3.9 -m venv .pyenv
. .pyenv/bin/activate
```

Install the `nomad-lab` package:
```sh
pip install --upgrade pip
pip install '.[dev]' --index-url https://gitlab.mpcdf.mpg.de/api/v4/projects/2187/packages/pypi/simple
```

**Note!**
Until we have an official pypi NOMAD release with the plugins functionality. Make
sure to include NOMAD's internal package registry (via `--index-url` in the above command).

### Run the tests

You can run local tests using the `pytest` package:

```sh
python -m pytest -sv
```

where the `-s` and `-v` options toggle the output verbosity.
For interactive debugging, use the `--pdb` flag.
When debgugging in the VSCode IDE, add the following to the `launch.json` jobs:

```json
"env": {
    "_PYTEST_RAISE": "1"
}
```

Our CI/CD pipeline produces a more comprehensive test report using `coverage` and `coveralls` packages.
To emulate this locally, perform:

```sh
pip install coverage coveralls
python -m coverage run -m pytest -sv
```

## Development

The plugin is still under development. If you would like to contribute, install the package in editable mode (with the added `-e` flag) with the development dependencies:

```sh
pip install -e .[dev] --index-url https://gitlab.mpcdf.mpg.de/api/v4/projects/2187/packages/pypi/simple
```

### Setting up plugin on your local installation
Read the [NOMAD plugin documentation](https://nomad-lab.eu/prod/v1/staging/docs/howto/oasis/plugins_install.html) for all details on how to deploy the plugin on your NOMAD instance.

You need to modify the ```src/nomad_simulations/nomad_plugin.yaml``` to define the plugin adding the following content:

```yaml
plugin_type: schema
name: schemas/nomad_simulations
description: |
  This is a collection of NOMAD schemas for simulation data.
```

and define the ```nomad.yaml``` configuration file of your NOMAD instance in the root folder with the following content:

```yaml
plugins:
  include:
    - schemas/nomad_simulations
  options:
    schemas/nomad_simulations:
      python_package: nomad_simulations
```

You also need to add the package folder to the `PYTHONPATH` of the Python environment of your local NOMAD installation. This can be done by specifying the relative path to this repository. Either run the following command every time you start a new terminal for running the appworker, or add it to your virtual environment in `<path-to-local-nomad-installation>/.pyenv/bin/activate` file:

```sh
export PYTHONPATH="$PYTHONPATH:<path-to-nomad-simulations-cloned-repo>/src"
```

If you are working in this repository, you just need to activate the environment to start working using the ```nomad_simulations``` package.

### Run linting and auto-formatting

```sh
ruff check .
ruff format . --check
```

Ruff auto-formatting is also a part of the GitHub workflow actions. Make sure that before you make a Pull Request, `ruff format . --check` runs in your local without any errors otherwise the workflow action will fail.

Alternatively, if you are using VSCode as your IDE, we added the settings configuration file, `.vscode/settings.json`, such that it performs `ruff format` whenever you save progress in a file.
