from pyxdm.core import XDMSession
from pyxdm.utils.io import write_h5_output

session = XDMSession('water/orca.molden.input')
session.load_molecule()
session.setup_grid()
session.setup_calculator()
session.setup_partition_schemes(['becke'])
xdm_results = session.calculator.calculate_xdm_moments(
    partition_obj=session.partitions['becke'],
    grid=session.grid,
    order=[1, 2, 3],
    anisotropic=False,
)
write_h5_output('pyxdm.h5', session, xdm_results)