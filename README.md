# PyXDM

PyXDM is a Python package for calculating XDM (Exchange-hole Dipole Moment) multipole moments using multiple atoms-in-molecules (AIM) partitioning schemes.

## Features

- **XDM multipole moments**: dipole, quadrupole, octupole.
- **AIM schemes**: Becke, Hirshfeld, Hirshfeld-I, Iterative Stockholder, MBIS.


## Installation

Create a conda environment with all the required dependencies:

```
onda env create -f environment.yaml
conda activate pyxdm
```

Install `pyxdm` in interactive mode within the activated environment:

```bash
pip install -e .
```

For developers:

```bash
pip install -e .[dev]
```

## Usage

### Command Line Interface

After installation, use the `pyxdm` command:

```bash
pyxdm <wfn_file> [--mesh <mesh_file>] [--scheme <scheme>] [--order <orders>] [--proatomdb <path>] [-v]
```

#### Arguments
- `<wfn_file>`: Path to the wavefunction file (Molden, WFN, etc.)
- `--mesh`: Optional custom mesh file for integration grid
- `--scheme`: Partitioning scheme to use (default: all available)
- `--order`: Multipole orders to calculate (e.g., 1 2 3)
- `--proatomdb`: Path to proatom database (required for Hirshfeld schemes)
- `-v`, `--verbose`: Enable verbose output

#### Example

```bash
pyxdm orca.molden.input --scheme mbis --order 1 2 3 
```

### Python API

You can also use PyXDM as a library in your own Python scripts:

```python
from pyxdm.core import XDMSession

session = XDMSession('examples/water/orca.molden.input', verbose=True)
session.load_molecule()
session.setup_grid()
session.setup_calculator()
session.setup_partition_schemes(['mbis'])
results = session.calculator.calculate_moments(
    partition_obj=session.partitions['mbis'],
    grid=session.grid,
    multipole_orders=[1, 2, 3],
)
```

## Acknowledgments

Some implementations in this package, such as the Newton-Raphson algorithm for the Becke-Roussel (BR) exchange model, are based on [postg](https://github.com/aoterodelaroza/postg). As a state-of-the-art package for XDM calculations, postg serves as an essential reference for accuracy and methodology. The aim of this package is mainly to provide easy-to-use access to a broader range of partitioning schemes beyond Hirshfeld.
