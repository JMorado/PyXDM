"""XDM calculation session management."""

import sys
import logging
from pathlib import Path
from typing import Optional

from ..grids import load_mesh, CustomGrid
from ..partitioning import PartitioningSchemeFactory
from ..core import XDMCalculator

logger = logging.getLogger(__name__)

try:
    import horton as ht
except ImportError:
    logger.error("horton package is required for XDM calculations")
    sys.exit(1)


class XDMSession:
    """
    Session manager for XDM calculations.

    This class manages the overall workflow of XDM calculations, including
    molecule loading, grid setup, and coordination between different components.

    Attributes
    ----------
    wfn_file : Path
        Path to the wavefunction file
    mol : object or None
        Loaded molecule object
    grid : object or None
        Computational grid object
    calculator : XDMCalculator or None
        XDM calculator instance
    partitions : dict or None
        Dictionary of partition objects
    """

    def __init__(self, wfn_file: str) -> None:
        """
        Initialize XDM calculation session.

        Parameters
        ----------
        wfn_file : str
            Path to the wavefunction file

        Returns
        -------
        None
        """
        self.wfn_file = Path(wfn_file)
        self.mol = None
        self.grid = None
        self.calculator = None
        self.partitions: dict[str, object] = {}

    def load_molecule(self) -> None:
        """
        Load molecule from wavefunction file.

        Returns
        -------
        None
        """
        try:
            self.mol = ht.IOData.from_file(str(self.wfn_file))
        except Exception as e:
            logger.error(f"Error loading wavefunction file: {e}")
            sys.exit(1)

    def setup_grid(self, mesh_file: Optional[str] = None) -> None:
        """
        Setup computational grid.

        Parameters
        ----------
        mesh_file : str, optional
            Path to custom mesh file, if None uses default Becke grid

        Returns
        -------
        None
        """
        if not self.mol:
            raise RuntimeError("Molecule must be loaded before setting up grid")

        if mesh_file:
            try:
                mesh_points, mesh_weights = load_mesh(mesh_file)
                self.grid = CustomGrid(mesh_points, mesh_weights)
                logger.info(f"Loaded custom mesh with {len(mesh_points)} points")
            except Exception as e:
                logger.error(f"Error loading mesh file: {e}")
                sys.exit(1)
        else:
            agspec = ht.AtomicGridSpec("ultrafine")
            self.grid = ht.BeckeMolGrid(
                self.mol.coordinates,
                self.mol.numbers,
                self.mol.pseudo_numbers,
                agspec=agspec,
                mode="keep",
                random_rotate=False,
            )
            logger.info(f"Using Becke grid with {len(self.grid.points)} points")

    def setup_calculator(self) -> None:
        """
        Setup XDM calculator.

        Returns
        -------
        None
        """
        if not self.mol or not self.grid:
            raise RuntimeError("Molecule and grid must be set up before creating calculator")

        self.calculator = XDMCalculator(self.mol)
        logger.debug("XDM calculator initialized")

    def setup_partition_schemes(self, schemes: list[str], proatomdb: Optional[str] = None) -> dict:
        """
        Setup partitioning schemes for the session.

        Parameters
        ----------
        schemes : list of str
            List of partitioning scheme names to use
        proatomdb : str, optional
            Path to proatom database for Hirshfeld schemes

        Returns
        -------
        dict
            Dictionary of partition objects.
        """
        self.partitions = {}

        for scheme in schemes:
            try:
                # Prepare kwargs for scheme creation
                scheme_kwargs = {}
                if proatomdb:
                    scheme_kwargs["proatom_db"] = proatomdb

                partitioning = PartitioningSchemeFactory.create_scheme(scheme, **scheme_kwargs)

                partitioning.compute_weights(self.mol, self.grid)

                partition_obj = partitioning.get_partition_object()
                if partition_obj is not None:
                    self.partitions[scheme] = partition_obj
                else:
                    logger.warning(f"No partition object available for {scheme}")

            except Exception as e:
                logger.warning(f"Failed to compute {scheme} partition: {e}")
                continue

        return self.partitions
