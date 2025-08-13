"""Formatting utilities for XDM output."""

import numpy as np
from typing import List, Optional, Dict, Set, Union

from . import logger

# Constants for formatting
ATOMIC_SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl"}
MULTIPOLE_ORDER_NAMES = {1: "M1^2", 2: "M2^2", 3: "M3^2"}
MULTIPOLE_FULL_NAMES = {1: "Dipole", 2: "Quadrupole", 3: "Octupole"}


def get_atomic_symbol(atomic_number: int) -> str:
    """
    Get atomic symbol from atomic number.

    Parameters
    ----------
    atomic_number : int
        Atomic number

    Returns
    -------
    str
        Atomic symbol or fallback format
    """
    return ATOMIC_SYMBOLS.get(atomic_number, f"Z{atomic_number}")


def format_scientific(value: float, precision: int = 6) -> str:
    """
    Format a number in scientific notation like POSTG.

    Parameters
    ----------
    value : float
        Value to format
    precision : int, optional
        Number of decimal places, default 6

    Returns
    -------
    str
        Formatted scientific notation string
    """
    if abs(value) < 1e-10:
        return f"{'0.0':>13}"
    else:
        return f"{value:.{precision}E}"



def print_table(
    data: List[List[str]], headers: List[str], title: Optional[str] = None
) -> None:
    """
    Print a formatted table using logger.

    Parameters
    ----------
    data : list of list of str
        Table data as list of rows
    headers : list of str
        Column headers
    title : str, optional
        Table title
    """
    if title:
        logger.header(title)

    # Calculate column widths
    col_widths = []
    for i, header in enumerate(headers):
        max_width = len(header)
        for row in data:
            if i < len(row):
                max_width = max(max_width, len(str(row[i])))
        col_widths.append(max_width + 2)

    # Print header
    header_line = ""
    for i, header in enumerate(headers):
        if i == 0:
            header_line += f"  {header:<{col_widths[i] - 2}}"
        else:
            header_line += f"{header:>{col_widths[i]}}"
    logger.info(header_line)

    # Print data rows
    for row in data:
        data_line = ""
        for i, cell in enumerate(row):
            if i < len(col_widths):
                if i == 0:
                    data_line += f"  {str(cell):<{col_widths[i] - 2}}"
                else:
                    data_line += f"{str(cell):>{col_widths[i]}}"
        logger.info(data_line)
    logger.info("")


def print_results(
    results: Dict[int, Dict[str, Dict[str, Union[np.ndarray, float]]]],
    orders: List[int],
    mol: object,
    wfn_file: str,
    mesh_file: Optional[str] = None,
) -> None:
    """
    Print calculation results in formatted tables.

    Parameters
    ----------
    results : dict
        Results dictionary with structure {order: {scheme: {atomic/total: values}}}
    orders : list of int
        List of multipole orders calculated
    mol : object
        Molecule object
    wfn_file : str
        Path to wavefunction file
    mesh_file : str, optional
        Path to mesh file if used
    """
    if not results:
        logger.warning("No successful calculations to display.")
        return

    all_schemes_set: Set[str] = set()
    for order in orders:
        if order in results:
            all_schemes_set.update(results[order].keys())
    all_schemes = sorted(list(all_schemes_set))

    if not all_schemes:
        logger.warning("No successful calculations to display.")
        return

    logger.header("PYXDM OUTPUT")
    logger.info("XDM multipole moments calculation")
    logger.info(f"Wavefunction file: {wfn_file}")
    if mesh_file:
        logger.info(f"Mesh file: {mesh_file}")
    logger.info(f"Number of atoms: {getattr(mol, 'natom', 'unknown')}")
    logger.info("")

    # Get number of atoms from first successful result
    first_order = next(iter(results.keys()))
    first_scheme = next(iter(results[first_order].keys()))
    atomic_data_sample = results[first_order][first_scheme]["atomic"]
    n_atoms = len(atomic_data_sample) if hasattr(atomic_data_sample, '__len__') else 1

    for scheme in all_schemes:
        scheme_name = scheme.upper().replace("-", " ")
        logger.header(f"{scheme_name}")
        logger.subheader("Moments")

        # Atomic moments table
        headers = ["i", "At"]
        for order in orders:
            if order in results and scheme in results[order]:
                headers.append(f"<{MULTIPOLE_ORDER_NAMES[order]}>")

        atomic_data = []
        for i in range(n_atoms):
            atomic_numbers = getattr(mol, 'numbers', [None]*n_atoms)
            atomic_number = atomic_numbers[i] if isinstance(atomic_numbers, (list, tuple)) else None
            symbol = get_atomic_symbol(int(atomic_number) if atomic_number is not None else 0)

            row = [f"{i + 1}", symbol]
            for order in orders:
                if order in results and scheme in results[order]:
                    atomic_moments = results[order][scheme]["atomic"]
                    moment = atomic_moments[i] if hasattr(atomic_moments, '__getitem__') else atomic_moments
                    row.append(format_scientific(moment))
                else:
                    row.append("    -    ")
            atomic_data.append(row)

        print_table(atomic_data, headers)

        # Total moments table
        logger.subheader("Total moments")
        total_headers = ["Order", "Moment", "Value"]
        total_data = []

        for order in orders:
            if order in results and scheme in results[order]:
                total = results[order][scheme]["total"]
                order_name = MULTIPOLE_FULL_NAMES.get(order, f"L={order}")
                total_data.append([f"L={order}", order_name, format_scientific(total)])

        print_table(total_data, total_headers)
