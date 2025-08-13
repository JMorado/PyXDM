"""Command line interface for XDM calculations."""

import argparse
import sys

from .core import XDMSession
from .partitioning import PartitioningSchemeFactory
from .utils import logger

DEFAULT_MULTIPOLE_ORDERS = [1, 2, 3]


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
        "--order",
        type=int,
        nargs="+",
        default=None,
        help="Multipole orders (1=dipole, 2=quadrupole, etc.) - if not specified, calculates l=1,2,3",
    )

    parser.add_argument(
        "--proatomdb",
        help="Path to proatom database file (required for Hirshfeld schemes)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
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
        session: XDMSession = XDMSession(args.wfn_file, verbose=args.verbose)
        session.load_molecule()
        session.setup_grid(args.mesh)
        session.setup_calculator()
        session.setup_partition_schemes(
            [args.scheme]
            if args.scheme
            else PartitioningSchemeFactory.available_schemes(),
            proatomdb=args.proatomdb,
        )

        orders: list[int] = (
            args.order if args.order is not None else DEFAULT_MULTIPOLE_ORDERS
        )

        if session.partitions is not None:
            for scheme, partition_obj in session.partitions.items():
                logger.info(f"Using partitioning scheme: {scheme}")

                # Compute partitions if needed
                if session.calculator is not None:
                    atomic_results, total_results = session.calculator.calculate_moments(
                        partition_obj=partition_obj,
                        grid=session.grid,
                        multipole_orders=orders,
                    )
                    print(atomic_results)
                else:
                    logger.error("No calculator available for session.")
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
