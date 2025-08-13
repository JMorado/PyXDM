"""Grid utilities for XDM calculations."""

import numpy as np
from typing import Tuple


def load_mesh(mesh_filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load mesh points and weights from mesh file.

    Parameters
    ----------
    mesh_filename : str
        Path to the mesh file

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        points: Array of grid points with shape (n_points, 3)
        weights: Array of grid weights with shape (n_points,)
    """
    points, weights = [], []

    with open(mesh_filename) as f:
        for line in f:
            if line.startswith("#") or len(line.strip()) == 0:
                continue
            parts = line.split()
            if len(parts) >= 4:
                x, y, z, w = map(float, parts[:4])
                points.append([x, y, z])
                weights.append(w)

    return np.array(points), np.array(weights)


class CustomGrid:
    """
    Custom grid class that mimics Horton's BeckeMolGrid interface.

    Attributes
    ----------
    points : np.ndarray
        Grid points array of shape (n_points, 3)
    weights : np.ndarray
        Grid weights array of shape (n_points,)
    n : int
        Number of grid points
    """

    def __init__(self, points: np.ndarray, weights: np.ndarray) -> None:
        """
        Initialize custom grid.

        Parameters
        ----------
        points : np.ndarray
            Grid points array of shape (n_points, 3)
        weights : np.ndarray
            Grid weights array of shape (n_points,)

        Returns
        -------
        None
        """
        self.points = points
        self.weights = weights
        self.n = len(points)

    def integrate(self, function_values: np.ndarray) -> float:
        """
        Integrate function values over the grid.

        Parameters
        ----------
        function_values : np.ndarray
            Values to integrate

        Returns
        -------
        float
            Integrated value
        """
        return float(np.sum(self.weights * function_values))
