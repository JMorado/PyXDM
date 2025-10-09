#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  
#include <cmath>
#include <vector>
#include <omp.h>

namespace py = pybind11;

std::vector<double> compute_b_sigma_cpp(const py::array_t<double>& rho_sigma, const py::array_t<double>& Q_sigma) {
    auto rho = rho_sigma.unchecked<1>();
    auto Q = Q_sigma.unchecked<1>();
    size_t n = rho.shape(0);
    std::vector<double> b_sigma(n, 1e-10);

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        double rho_i = rho(i);
        double Q_i = Q(i);

        if (rho_i <= 1e-15 || std::abs(Q_i) <= 1e-15) {
            b_sigma[i] = 1e-10;
            continue;
        }

        // Constants
        const double pi = M_PI;
        const double third = 1.0 / 3.0;
        const double third2 = 2.0 / 3.0;

        // Compute rhs
        double rhs = third2 * std::pow(pi * rho_i, third2) * rho_i / Q_i;
        double x0 = 2.0, shift = 1.0;
        double x = x0;

        // Initial guess
        if (rhs < 0.0) {
            for (int j = 0; j < 16; ++j) {
                x = x0 - shift;
                double expo23 = std::exp(-2.0 / 3.0 * x);
                double f = x * expo23 / (x - 2.0) - rhs;
                if (f < 0.0) break;
                shift *= 0.1;
            }
        } else if (rhs > 0.0) {
            for (int j = 0; j < 16; ++j) {
                x = x0 + shift;
                double expo23 = std::exp(-2.0 / 3.0 * x);
                double f = x * expo23 / (x - 2.0) - rhs;
                if (f > 0.0) break;
                shift *= 0.1;
            }
        }

        // Newton-Raphson
        double x1 = x;
        for (int j = 0; j < 100; ++j) {
            double expo23 = std::exp(-2.0 / 3.0 * x1);
            double f = x1 * expo23 / (x1 - 2.0) - rhs;
            double df = (2.0 / 3.0) * (2.0 * x1 - x1 * x1 - 3.0) / std::pow(x1 - 2.0, 2) * expo23;
            if (std::abs(df) < 1e-15) {
                x1 = 1e-10;
                break;
            }
            double x_next = x1 - f / df;
            if (std::abs(x_next - x1) < 1e-10) {
                x1 = x_next;
                break;
            }
            x1 = x_next;
        }

        // Final calculation
        double expo = std::exp(-x1);
        double prefac = rho_i / expo;
        double alf = std::pow(8.0 * pi * prefac, third);
        b_sigma[i] = x1 / alf;
        if (!std::isfinite(b_sigma[i])) b_sigma[i] = 1e-10;
    }

    return b_sigma;
}

PYBIND11_MODULE(exchange_hole_cpp, m) {
    m.def("compute_b_sigma", &compute_b_sigma_cpp, "Compute Becke-Roussel displacement b_sigma (C++/OpenMP)");
}