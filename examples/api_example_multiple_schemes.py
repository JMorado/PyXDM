from pyxdm.core import XDMSession
from pyxdm.utils.io import write_h5_output

SCHEMES = ["mbis", "iterative-stockholder", "hirshfeld", "hirshfeld-i", "becke"]

session = XDMSession("water/orca.molden.input")
session.load_molecule()
session.setup_grid()
session.setup_calculator()
session.setup_partition_schemes(SCHEMES, proatomdb="proatomdb_wb97x-D3_def2-TZVPP.h5")

for scheme in SCHEMES:
    print(f"Calculating XDM moments for partitioning scheme: {scheme}")
    xdm_results = session.calculator.calculate_xdm_moments(
        partition_obj=session.partitions[scheme],
        grid=session.grid,
        order=[1, 2, 3],
        anisotropic=False,
    )
    write_h5_output(f"pyxdm_scheme_{scheme}.h5", session, xdm_results)
