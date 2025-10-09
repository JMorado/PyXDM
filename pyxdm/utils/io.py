"""Input/output utilities for PyXDM."""

import logging
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Any

from .formatting import get_atomic_symbol

logger = logging.getLogger(__name__)


def write_h5_output(filename: str, session, all_results: Dict[str, Any], wall_time: float) -> None:
    """
    Write calculated data to HDF5 file.

    Parameters
    ----------
    filename : str
        Output HDF5 filename
    session : XDMSession
        XDM session containing molecule and calculation data
    all_results : dict
        Dictionary containing all calculated results by scheme
    wall_time : float
        Total calculation wall time in seconds
    """
    logger.info(f"Writing results to {filename}")
    
    with h5py.File(filename, 'w') as f:
        # Write metadata
        metadata = f.create_group('metadata')
        metadata.attrs['pyxdm_version'] = '0.1.0'
        metadata.attrs['calculation_time'] = wall_time
        metadata.attrs['wavefunction_file'] = str(session.wfn_file)
        
        # Write molecular information
        molecule = f.create_group('molecule')
        molecule.create_dataset('atomic_numbers', data=session.mol.numbers)
        molecule.create_dataset('coordinates', data=session.mol.coordinates)
        
        # Calculate nelec from populations (sum of all atomic populations)
        nelec = None
        if all_results:
            first_scheme_results = next(iter(all_results.values()))
            if 'populations' in first_scheme_results:
                nelec = int(round(np.sum(first_scheme_results['populations'])))
        
        # Fallback to atomic numbers if populations not available
        if nelec is None:
            nelec = int(np.sum(session.mol.numbers))
            
        molecule.create_dataset('nelec', data=nelec)
        molecule.attrs['natom'] = session.mol.natom
        
        # Write atomic symbols as string dataset
        symbols_str = [get_atomic_symbol(num).encode('utf-8') for num in session.mol.numbers]
        molecule.create_dataset('atomic_symbols', data=symbols_str)
        
        # Write results for each partitioning scheme
        for scheme_name, results_data in all_results.items():
            scheme_group = f.create_group(f'results/{scheme_name}')
            
            # Write multipole moments
            if 'atomic_results' in results_data:
                multipole_group = scheme_group.create_group('multipole_moments')
                for key, values in results_data['atomic_results'].items():
                    multipole_group.create_dataset(key, data=values)
            
            # Write tensor multipole moments (anisotropic moments)
            if 'tensor_results' in results_data:
                tensor_group = scheme_group.create_group('multipole_moments_tensor')
                for key, tensors in results_data['tensor_results'].items():
                    tensor_group.create_dataset(key, data=np.array(tensors))
            
            # Write geometric factors
            if 'geom_factors' in results_data:
                geom_group = scheme_group.create_group('geometric_factors')
                for key, values in results_data['geom_factors'].items():
                    geom_group.create_dataset(f'f_{key}', data=values)
            
            # Write radial moments
            if 'radial_moments' in results_data:
                radial_group = scheme_group.create_group('radial_moments')
                for key, values in results_data['radial_moments'].items():
                    radial_group.create_dataset(key, data=values)
            
            # Write charges and populations if available
            if 'charges' in results_data:
                scheme_group.create_dataset('charges', data=results_data['charges'])
            if 'populations' in results_data:
                scheme_group.create_dataset('populations', data=results_data['populations'])
