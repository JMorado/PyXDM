import numpy as np
from typing import Any, List, Union
import logging

from .exchange_hole import compute_b_sigma

logger = logging.getLogger(__name__)


class XDMCalculator:
    """Core calculator for XDM multipole moments."""

    def __init__(self, mol: Any) -> None:
        self.mol = mol
        self.n_atoms: int = mol.natom

        dm_full = mol.get_dm_full()
        dm_spin = mol.get_dm_spin()

        if dm_spin is None:
            self.dm_alpha = dm_full * 0.5
            self.dm_beta = dm_full * 0.5
        else:
            self.dm_alpha = 0.5 * (dm_full + dm_spin)
            self.dm_beta = 0.5 * (dm_full - dm_spin)

        self.dm_full = dm_full
        self.dm_spin = dm_spin

    def _compute_density_properties(self, grid_points: np.ndarray, anisotropic: bool = True) -> dict:
        """Compute density and derived properties on grid points."""
        rho_alpha = self.mol.obasis.compute_grid_density_dm(self.dm_alpha, grid_points)
        rho_beta = self.mol.obasis.compute_grid_density_dm(self.dm_beta, grid_points)

        tau_alpha = 2.0 * self.mol.obasis.compute_grid_kinetic_dm(self.dm_alpha, grid_points)
        tau_beta = 2.0 * self.mol.obasis.compute_grid_kinetic_dm(self.dm_beta, grid_points)

        nabla_alpha = self.mol.obasis.compute_grid_gradient_dm(self.dm_alpha, grid_points)
        nabla_beta = self.mol.obasis.compute_grid_gradient_dm(self.dm_beta, grid_points)

        hessian_alpha = self.mol.obasis.compute_grid_hessian_dm(self.dm_alpha, grid_points)
        hessian_beta = self.mol.obasis.compute_grid_hessian_dm(self.dm_beta, grid_points)

        laplacian_alpha = hessian_alpha[:, 0] + hessian_alpha[:, 3] + hessian_alpha[:, 5]
        laplacian_beta = hessian_beta[:, 0] + hessian_beta[:, 3] + hessian_beta[:, 5]

        nabla_alpha_mag2 = np.sum(nabla_alpha**2, axis=1)
        nabla_beta_mag2 = np.sum(nabla_beta**2, axis=1)

        d_alpha = tau_alpha - 0.25 * nabla_alpha_mag2 / rho_alpha
        d_beta = tau_beta - 0.25 * nabla_beta_mag2 / rho_beta

        Q_alpha = (laplacian_alpha - 2.0 * d_alpha) / 6.0
        Q_beta = (laplacian_beta - 2.0 * d_beta) / 6.0

        b_alpha = compute_b_sigma(rho_alpha, Q_alpha)
        b_beta = compute_b_sigma(rho_beta, Q_beta)

        props = {"rho_alpha": rho_alpha, "rho_beta": rho_beta, "b_alpha": b_alpha, "b_beta": b_beta}

        if anisotropic:
            # Calculate the normalized gradient vectors and add them to the properties dict
            epsilon = 1e-12
            nabla_alpha_norm = np.linalg.norm(nabla_alpha, axis=1)[:, None]
            nabla_beta_norm = np.linalg.norm(nabla_beta, axis=1)[:, None]

            props["u_alpha"] = nabla_alpha / (nabla_alpha_norm + epsilon)
            props["u_beta"] = nabla_beta / (nabla_beta_norm + epsilon)

        return props

    def _compute_moment_for_atom(
        self, atom_idx: int, grid: Any, weights_i: np.ndarray, order: int, density_props: dict, anisotropic: bool = False
    ) -> np.ndarray:
        """Compute multipole moment for a single atom."""
        r_vec = grid.points - self.mol.coordinates[atom_idx]
        r_i = np.linalg.norm(r_vec, axis=1)

        if not anisotropic:
            logger.debug(f"Calculating isotropic moment for atom {atom_idx} with order {order}")
            # Capped b_sigma values
            b_alpha_capped = np.minimum(density_props["b_alpha"], r_i)
            b_beta_capped = np.minimum(density_props["b_beta"], r_i)

            # The term that gets squared
            term_alpha_iso = r_i**order - (r_i - b_alpha_capped) ** order
            term_beta_iso = r_i**order - (r_i - b_beta_capped) ** order

            # The integrand includes the square of the term
            integrand_alpha = density_props["rho_alpha"] * term_alpha_iso**2
            integrand_beta = density_props["rho_beta"] * term_beta_iso**2

            # Perform the integration
            lambda_alpha = grid.integrate(integrand_alpha * weights_i)
            lambda_beta = grid.integrate(integrand_beta * weights_i)

            lambda_iso = lambda_alpha + lambda_beta
            return lambda_iso, None

        if anisotropic:
            logger.debug(f"Calculating anisotropic moment for atom {atom_idx} with order {order}")
            b_alpha_capped = np.minimum(density_props["b_alpha"], r_i)
            b_beta_capped = np.minimum(density_props["b_beta"], r_i)

            # Get the magnitudes of the induced moments
            moment_mag_alpha = r_i**order - (r_i - b_alpha_capped) ** order
            moment_mag_beta = r_i**order - (r_i - b_beta_capped) ** order

            # CORRECTED: Construct the vector using the density gradient direction
            term_alpha_vec = moment_mag_alpha[:, None] * density_props["u_alpha"]
            term_beta_vec = moment_mag_beta[:, None] * density_props["u_beta"]

            # The rest of your calculation is correct
            outer_alpha = np.einsum("ni,nj->nij", term_alpha_vec, term_alpha_vec)
            outer_beta = np.einsum("ni,nj->nij", term_beta_vec, term_beta_vec)

            integrand_alpha = density_props["rho_alpha"][:, None, None] * outer_alpha
            integrand_beta = density_props["rho_beta"][:, None, None] * outer_beta

            lambda_alpha_tensor = np.sum(grid.weights[:, None, None] * weights_i[:, None, None] * integrand_alpha, axis=0)
            lambda_beta_tensor = np.sum(grid.weights[:, None, None] * weights_i[:, None, None] * integrand_beta, axis=0)

            lambda_tensor = lambda_alpha_tensor + lambda_beta_tensor
            lambda_iso = np.trace(lambda_tensor)

            return lambda_iso, lambda_tensor

    def calculate_xdm_moments(self, partition_obj: Any, grid: Any, order: Union[List[int], int], anisotropic: bool = True) -> dict:
        """Calculate XDM multipole moments for all atoms."""
        logger.debug("Calculating XDM multipole moments for all atoms...")
        order = order if isinstance(order, list) else [order]
        xdm_results = {f"<M{n}^2>": np.zeros(self.n_atoms) for n in order}
        xdm_results_tensor = {f"<M{n}^2>_tensor": [None] * self.n_atoms for n in order}

        for atom_idx in range(self.n_atoms):
            grid = partition_obj.get_grid(atom_idx)
            weights_i = partition_obj.cache.load("at_weights", atom_idx)
            density_props = self._compute_density_properties(grid.points, anisotropic=anisotropic)

            for o in order:
                iso, tensor = self._compute_moment_for_atom(atom_idx, grid, weights_i, o, density_props, anisotropic=anisotropic)
                xdm_results[f"<M{o}^2>"][atom_idx] = iso
                xdm_results_tensor[f"<M{o}^2>_tensor"][atom_idx] = tensor

        return xdm_results, xdm_results_tensor

    def calculate_radial_moments(self, partition_obj: Any, order: Union[List[int], int] = 3) -> dict:
        """Calculate radial moments <r^order> for all atoms."""
        logger.debug("Calculating radial moments for all atoms...")
        order = order if isinstance(order, list) else [order]
        moments_dict = {f"<r^{o}>": np.zeros(self.mol.natom) for o in order}

        for i in range(self.mol.natom):
            subgrid = partition_obj.get_grid(i)
            weights_i = partition_obj.cache.load("at_weights", i)
            rho_subgrid = self.mol.obasis.compute_grid_density_dm(self.dm_full, subgrid.points)
            r = np.linalg.norm(subgrid.points - self.mol.coordinates[i], axis=1)

            for o in order:
                moments_dict[f"<r^{o}>"][i] = subgrid.integrate(r**o * weights_i * rho_subgrid)

        return moments_dict

    @staticmethod
    def geom_factor(tensor: np.ndarray) -> float:
        """
        Compute the geometric factor f_geom for a tensor based on its eigenvalues.

        Parameters
        ----------
        eigenvalues : np.ndarray
            Eigenvalues of the tensor (length 3).

        Returns
        -------
        float
            Geometric scaling factor f_geom.
        """
        eigenvalues = np.linalg.eigvalsh(tensor)
        assert all(eigenvalues >= 0), "Eigenvalues must be non-negative for geometric factor calculation"
        mean_val = np.mean(eigenvalues)
        geom_mean = np.prod(eigenvalues) ** (1 / 3)
        f_geom = (geom_mean**2) / (mean_val**2)
        return f_geom
