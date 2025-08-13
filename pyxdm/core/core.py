"""Core XDM calculations with comprehensive type hints and documentation."""

import numpy as np
from typing import Any
import logging

from .exchange_hole import compute_b_sigma

logger = logging.getLogger(__name__)


class XDMCalculator:
    """
    Core calculator for XDM multipole moments.

    This class handles the computation of XDM (Exchange-hole Dipole Moment)
    multipole moments for atoms in molecules using partitioned electron densities.

    Attributes
    ----------
    mol : object
        Molecule object containing basis set and geometry information
    n_atoms : int
        Number of atoms in the molecule
    dm_alpha : np.ndarray
        Alpha spin density matrix
    dm_beta : np.ndarray
        Beta spin density matrix
    dm_full : np.ndarray
        Full density matrix
    dm_spin : np.ndarray or None
        Spin density matrix (if open-shell)
    """

    def __init__(self, mol: Any) -> None:
        """
        Initialize XDM calculator.

        Parameters
        ----------
        mol : Any
            Molecule object with basis set and coordinate information

        Returns
        -------
        None
        """
        self.mol = mol
        self.n_atoms: int = mol.natom

        # Get density matrices once for efficiency
        dm_full = mol.get_dm_full()
        dm_spin = mol.get_dm_spin()

        if dm_spin is None:
            # Closed-shell case: split total density equally
            self.dm_alpha = dm_full * 0.5
            self.dm_beta = dm_full * 0.5
        else:
            # Open-shell case: separate alpha and beta densities
            self.dm_alpha = 0.5 * (dm_full + dm_spin)
            self.dm_beta = 0.5 * (dm_full - dm_spin)

        self.dm_full = dm_full
        self.dm_spin = dm_spin

    def _compute_density_properties(self, grid_points: np.ndarray) -> dict:
        """
        Compute density and derived properties on grid points.

        Parameters
        ----------
        grid_points : np.ndarray
            Grid points coordinates (n_points x 3)

        Returns
        -------
        dict
            Dictionary containing computed density properties:
            - rho_alpha: Alpha spin density values
            - rho_beta: Beta spin density values
            - b_alpha: Alpha exchange hole radii
            - b_beta: Beta exchange hole radii
        """
        """Compute density and derived properties on grid points.

        Parameters
        ----------
        grid_points : NDArray[np.float64]
            Grid points coordinates (n_points x 3)

        Returns
        -------
        dict
            Dictionary containing computed density properties:
            - rho_alpha: Alpha spin density values
            - rho_beta: Beta spin density values
            - b_alpha: Alpha exchange hole radii
            - b_beta: Beta exchange hole radii
        """
        # Compute densities
        rho_alpha = self.mol.obasis.compute_grid_density_dm(self.dm_alpha, grid_points)
        rho_beta = self.mol.obasis.compute_grid_density_dm(self.dm_beta, grid_points)

        # Kinetic energy densities
        tau_alpha = self.mol.obasis.compute_grid_kinetic_dm(self.dm_alpha, grid_points) * 2.0
        tau_beta = self.mol.obasis.compute_grid_kinetic_dm(self.dm_beta, grid_points) * 2.0

        # Hessians and gradients
        hessian_alpha = self.mol.obasis.compute_grid_hessian_dm(self.dm_alpha, grid_points)
        hessian_beta = self.mol.obasis.compute_grid_hessian_dm(self.dm_beta, grid_points)
        nabla_alpha = self.mol.obasis.compute_grid_gradient_dm(self.dm_alpha, grid_points)
        nabla_beta = self.mol.obasis.compute_grid_gradient_dm(self.dm_beta, grid_points)

        # Laplacians
        laplacian_alpha = hessian_alpha[:, 0] + hessian_alpha[:, 3] + hessian_alpha[:, 5]
        laplacian_beta = hessian_beta[:, 0] + hessian_beta[:, 3] + hessian_beta[:, 5]

        # Gradient magnitudes squared
        nabla_alpha_mag2 = np.sum(nabla_alpha**2, axis=1)
        nabla_beta_mag2 = np.sum(nabla_beta**2, axis=1)

        # D quantities
        d_alpha = tau_alpha - 0.25 * nabla_alpha_mag2 / np.maximum(rho_alpha, 1e-30)
        d_beta = tau_beta - 0.25 * nabla_beta_mag2 / np.maximum(rho_beta, 1e-30)

        # Q quantities
        Q_alpha = (laplacian_alpha - 2 * d_alpha) / 6.0
        Q_beta = (laplacian_beta - 2 * d_beta) / 6.0

        # Exchange hole radii
        b_alpha = compute_b_sigma(rho_alpha, Q_alpha)
        b_beta = compute_b_sigma(rho_beta, Q_beta)

        return {
            "rho_alpha": rho_alpha,
            "rho_beta": rho_beta,
            "b_alpha": b_alpha,
            "b_beta": b_beta,
        }

    def _compute_moment_for_atom(
        self,
        atom_idx: int,
        grid: Any,
        weights_i: np.ndarray,
        multipole_order: int,
        density_props: dict,
    ) -> float:
        """
        Compute multipole moment for a single atom.

        Parameters
        ----------
        atom_idx : int
            Index of the atom to compute moments for
        grid : Any
            Grid object containing grid points and weights
        weights_i : np.ndarray
            Partition weights for this atom on the grid
        multipole_order : int
            Order of multipole moment to calculate
        density_props : dict
            Dictionary containing pre-computed density properties

        Returns
        -------
        float
            Computed multipole moment for the atom
        """
        # Distance from atom to grid points
        r_i = np.linalg.norm(grid.points - self.mol.coordinates[atom_idx], axis=1)

        # Cap b_sigma values
        b_alpha_capped = np.minimum(np.maximum(density_props["b_alpha"], 1e-10), r_i * 0.99)
        b_beta_capped = np.minimum(np.maximum(density_props["b_beta"], 1e-10), r_i * 0.99)

        # Exchange hole terms
        r_minus_b_alpha = np.maximum(0.0, r_i - b_alpha_capped)
        r_minus_b_beta = np.maximum(0.0, r_i - b_beta_capped)

        term_alpha = r_i**multipole_order - r_minus_b_alpha**multipole_order
        term_beta = r_i**multipole_order - r_minus_b_beta**multipole_order

        # Integrands
        integrand_alpha = density_props["rho_alpha"] * term_alpha**2
        integrand_beta = density_props["rho_beta"] * term_beta**2

        # Use global grid weights for integration
        lambda2_alpha = np.sum(grid.weights * weights_i * integrand_alpha)
        lambda2_beta = np.sum(grid.weights * weights_i * integrand_beta)

        return float(lambda2_alpha + lambda2_beta)

    def calculate_moments(self, partition_obj: Any, grid: Any, multipole_orders: list[int]) -> tuple[dict, dict]:
        """
        Calculate XDM multipole moments for all atoms.

        Parameters
        ----------
        partition_obj : Any
            Partitioning object containing atomic weights
        grid : Any
            Grid object containing grid points and coordinate information
        multipole_orders : list of int
            List of multipole orders to calculate

        Returns
        -------
        dict
            Dictionary with atomic multipoles.
        """
        logger.debug("Calculating XDM multiples moments for all atoms...")
        atomic_results = {}

        if hasattr(partition_obj, "get_grid"):
            for atom_idx in range(self.n_atoms):
                grid = partition_obj.get_grid(atom_idx)
                logger.debug(f"Atom {atom_idx} grid size = {len(grid.points)}")
                weights_i = partition_obj.cache.load("at_weights", atom_idx)
                density_props = self._compute_density_properties(grid.points)
                for order in multipole_orders:
                    atomic_moment = self._compute_moment_for_atom(atom_idx, grid, weights_i, order, density_props)
                    if order not in atomic_results:
                        atomic_results[order] = np.zeros(self.n_atoms)
                    atomic_results[order][atom_idx] = atomic_moment

        else:
            raise ValueError("")

        return atomic_results
