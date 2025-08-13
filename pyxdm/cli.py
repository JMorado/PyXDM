"""Command line interface for XDM calculations."""

import argparse
import sys

from .core import XDMSession
from .partitioning import PartitioningSchemeFactory
from .utils.formatting import log_mol_info, log_table, log_boxed_title, log_charges_populations, get_atomic_symbol
import logging
import numpy as np

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser for the CLI.
    """
    parser = argparse.ArgumentParser(
        description="Calculate XDM multipole moments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("wfn_file", help="Wavefunction file (molden, wfn, etc.)")

    parser.add_argument("--mesh", help="Optional postg mesh file")

    parser.add_argument(
        "--scheme",
        default=None,
        choices=PartitioningSchemeFactory.available_schemes(),
        help="Partitioning scheme to use (default: calculate for all schemes)",
    )

    parser.add_argument(
        "--proatomdb",
        help="Path to proatom database file (required for Hirshfeld schemes)",
    )

    return parser


def main() -> None:
    """
    Main CLI entry point.

    Parses command-line arguments, initializes the XDM session, and performs multipole moment calculations.

    Returns
    -------
    None
    """
    parser: argparse.ArgumentParser = create_argument_parser()
    args = parser.parse_args()

    try:
        # Initialize session and load molecule
        session: XDMSession = XDMSession(args.wfn_file)
        session.load_molecule()
        session.setup_grid(args.mesh)
        session.setup_calculator()
        session.setup_partition_schemes(
            [args.scheme] if args.scheme else PartitioningSchemeFactory.available_schemes(),
            proatomdb=args.proatomdb,
        )

        at_symbols = [get_atomic_symbol(num) for num in session.mol.numbers]

        # Log loaded molecule information
        log_mol_info(session.mol)

        if session.partitions is not None:
            for scheme, partition_obj in session.partitions.items():
                log_boxed_title(f"AIM Scheme: {scheme.upper()}", logger=logger)
                log_charges_populations(session, partition_obj, logger)

                # Compute partitions if needed
                if session.calculator is not None:
                    atomic_results = session.calculator.calculate_moments(
                        partition_obj=partition_obj,
                        grid=session.grid,
                        multipole_orders=[1, 2, 3],
                    )
                else:
                    logger.error("No calculator available for session.")

                # Log atomic data
                atomic_data = {}
                for key in atomic_results:
                    atomic_data[key] = list(atomic_results[key]) + [np.sum(atomic_results[key])]

                log_table(
                    logger,
                    columns=["<M1^2>", "<M2^2>", "<M3^2>"],
                    rows=at_symbols + ["Σ_atoms"],
                    data=np.column_stack([atomic_data[key] for key in atomic_data]),
                )

        log_boxed_title("PyXDM terminated successfully! :)", logger=logger)

    except Exception as e:
        import traceback

        logger.error(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
