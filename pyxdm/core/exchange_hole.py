"""Becke-Roussel exchange hole calculations."""

import numpy as np


def compute_b_sigma(rho_sigma: np.ndarray, Q_sigma: np.ndarray) -> np.ndarray:
    """
    Compute Becke-Roussel displacement b_sigma.

    Parameters
    ----------
    rho_sigma : np.ndarray
        Spin density values
    Q_sigma : np.ndarray
        Q values for each grid point

    Returns
    -------
    np.ndarray
        b_sigma values for each grid point
    """

    def _xfuncs(x: float, rhs: float) -> tuple[float, float]:
        """
        Exact implementation of postg's xfuncs subroutine.

        Parameters
        ----------
        x : float
            Variable x
        rhs : float
            Right-hand side value

        Returns
        -------
        tuple[float, float]
            Function value and its derivative
        """
        expo23 = np.exp(-2.0 / 3.0 * x)
        f = x * expo23 / (x - 2.0) - rhs
        df = (2.0 / 3.0) * (2.0 * x - x**2 - 3.0) / (x - 2.0) ** 2 * expo23
        return f, df

    def _bhole_exact(rho: float, quad: float) -> float:
        """
        Exact implementation of postg's bhole subroutine.

        Parameters
        ----------
        rho : float
            Density value
        quad : float
            Q value

        Returns
        -------
        float
            Computed b_sigma value
        """
        pi = np.pi
        third = 1.0 / 3.0
        third2 = 2.0 / 3.0

        rhs = third2 * (pi * rho) ** third2 * rho / quad
        x0, shift = 2.0, 1.0

        # Find initial guess
        if rhs < 0.0:
            for _ in range(16):
                x = x0 - shift
                f, _ = _xfuncs(x, rhs)
                if f < 0.0:
                    break
                shift = 0.1 * shift
            else:
                return 1e-10
        elif rhs > 0.0:
            for _ in range(16):
                x = x0 + shift
                f, _ = _xfuncs(x, rhs)
                if f > 0.0:
                    break
                shift = 0.1 * shift
            else:
                return 1e-10
        else:
            x = x0

        # Newton-Raphson iteration
        for _ in range(100):
            f, df = _xfuncs(x, rhs)
            if abs(df) < 1e-15:
                return 1e-10
            x1 = x - f / df
            if abs(x1 - x) < 1e-10:
                break
            x = x1
        else:
            return 1e-10

        # Final calculation
        expo = np.exp(-x1)
        prefac = rho / expo
        alf = (8.0 * pi * prefac) ** third
        return float(x1 / alf)

    # Vectorized computation
    b_sigma = np.zeros_like(rho_sigma)

    for i in range(len(rho_sigma)):
        rho, Q = rho_sigma[i], Q_sigma[i]

        if rho <= 1e-15 or abs(Q) <= 1e-15:
            b_sigma[i] = 1e-10
        else:
            try:
                b_sigma[i] = _bhole_exact(rho, Q)
            except Exception:
                b_sigma[i] = 1e-10

    return b_sigma
