"""Command line interface for XDM calculations."""

import argparse
import sys

from .core import XDMSession
from .partitioning import PartitioningSchemeFactory
from .utils.formatting import log_mol_info, log_table, log_boxed_title, log_charges_populations, get_atomic_symbol
import logging
import numpy as np
import time

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
        "--xdm-moments",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Order(s) of multipole moments to calculate (default: 1, 2, 3)",
    )

    parser.add_argument(
        "--radial-moments",
        type=int,
        nargs="+",
        default=None,
        help="Order(s) of radial moments to calculate (default: none)",
    )

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

    parser.add_argument(
        "--aniso",
        action="store_true",
        help="Calculate anisotropic multipole moments (default: False)",
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
    initial_time = time.time()
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
                assert partition_obj is not None, f"Partition object for {scheme} is None"
                assert session.calculator is not None, "Calculator must be set up before calculating moments"
                log_boxed_title(f"AIM Scheme: {scheme.upper()}", logger=logger)
                log_charges_populations(session, partition_obj, logger)

                # Compute partitions if needed
                atomic_results, tensor_results = session.calculator.calculate_xdm_moments(
                    partition_obj=partition_obj,
                    grid=session.grid,
                    order=args.xdm_moments,
                    anisotropic=args.aniso,
                )

                # Log atomic data
                atomic_data = {}
                for key in atomic_results:
                    atomic_data[key] = list(atomic_results[key]) + [np.sum(atomic_results[key])]

                log_table(
                    logger,
                    columns=[key for key in atomic_data],
                    rows=at_symbols + ["Σ_atoms"],
                    data=np.column_stack([atomic_data[key] for key in atomic_data]),
                )

                if args.aniso:
                    moment_keys = [key.split("_")[0] for key in tensor_results]
                    geom_factors_data = {key: np.zeros(len(at_symbols)) for key in moment_keys}

                    for key in tensor_results:
                        moment_key = key.split("_")[0]
                        for i, tensor in enumerate(tensor_results[key]):
                            logger.info(f"Atom {i} ({at_symbols[i]}) {moment_key} tensor:")
                            log_table(
                                logger,
                                columns=["x", "y", "z"],
                                rows=["x", "y", "z"],
                                data=tensor,
                            )
                            f_geom = session.calculator.geom_factor(tensor)
                            geom_factors_data[moment_key][i] = f_geom

                    logger.info("Summary of Geometric Anisotropy Factors (f_geom):")
                    log_table(
                        logger,
                        columns=[f"f({key})" for key in moment_keys],
                        rows=at_symbols,
                        data=np.column_stack(list(geom_factors_data.values())),
                    )

                if args.radial_moments:
                    # Compute radial moments
                    radial_moments = session.calculator.calculate_radial_moments(
                        partition_obj=partition_obj,
                        order=args.radial_moments,
                    )

                    log_table(
                        logger,
                        columns=[key for key in radial_moments],
                        rows=at_symbols,
                        data=np.column_stack([radial_moments[key] for key in radial_moments]),
                    )

        wall_time = time.time() - initial_time
        logger.info(f"Total wall time: {wall_time:.2f} seconds")
        log_boxed_title("PyXDM terminated successfully! :)", logger=logger)

    except Exception as e:
        import traceback

        logger.error(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
