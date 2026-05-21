import scipy.io as sio
import numpy as np
import torch
import datetime
from typing import Dict, Union

def export_fields_to_mat(
    data_dict: Dict[str, Union[torch.Tensor, np.ndarray]], 
    filename: str = "ThermalPINN_Output.mat",
    metadata: str = "Bi-Porous Vapor Chamber Optimization Results"
) -> None:
    """
    Exports high-fidelity predicted tensor fields to MATLAB for robust control 
    systems analysis and generating Q1 journal-ready contour plots.
    
    Args:
        data_dict: Dictionary containing flow and thermal fields (e.g., 'T_f', 'Velocity').
        filename: Target .mat file path.
        metadata: Description of the simulation run.
    """
    # Initialize dictionary with header and metadata
    export_dict = {
        '__header__': f'ThermalPINN Automated Export - {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}'.encode('utf-8'),
        'metadata': metadata
    }
    
    # Process tensors securely for MATLAB consumption
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            # Detach from computational graph, move to CPU, and convert to NumPy
            export_dict[key] = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            export_dict[key] = value
        else:
            raise TypeError(f"Unsupported data type for key '{key}'. Must be torch.Tensor or numpy.ndarray.")
            
    try:
        sio.savemat(filename, export_dict)
        print(f"[SUCCESS] Field data rigorously compiled and exported to: {filename}")
    except Exception as e:
        print(f"[ERROR] MATLAB export failed during scientific compilation: {str(e)}")
