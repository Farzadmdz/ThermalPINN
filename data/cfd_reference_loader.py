import os
import numpy as np
import scipy.io as sio
import torch
from typing import Dict, Tuple, Optional

class CFDTensorLoader:
    """
    High-fidelity Data Loader for Computational Fluid Dynamics (CFD) reference fields.
    
    Designed to ingest discrete Finite Volume/Finite Element Method (FVM/FEM) datasets,
    normalize thermophysical fields, and generate autograd-ready PyTorch tensors 
    for supervised pre-training of Physics-Informed Neural Networks (PINNs).
    """
    
    def __init__(self, data_dir: str, device: Optional[torch.device] = None):
        """
        Initializes the data loader with the target directory and hardware acceleration setup.
        """
        self.data_dir = data_dir
        # Hardware acceleration allocation (CUDA for NVIDIA GPUs, fallback to CPU)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaling_factors: Dict[str, Dict[str, float]] = {}

    def _normalize_field(self, field: np.ndarray, field_name: str) -> torch.Tensor:
        """
        Applies Min-Max scaling to map physical dimensions (e.g., temperature in Kelvin, 
        pressure in Pa) to a [-1, 1] range. This is mathematically critical to prevent 
        vanishing or exploding gradients during deep neural network backpropagation.
        """
        f_min, f_max = np.min(field), np.max(field)
        self.scaling_factors[field_name] = {'min': float(f_min), 'max': float(f_max)}
        
        # Avoid division by zero for uniform/constant fields
        if f_max - f_min < 1e-8:
            normalized = np.zeros_like(field)
        else:
            normalized = 2.0 * (field - f_min) / (f_max - f_min) - 1.0
            
        # Convert to PyTorch tensor with explicit memory tracking for Autograd
        return torch.tensor(normalized, dtype=torch.float32, device=self.device, requires_grad=True)

    def load_matlab_dataset(self, filename: str) -> Dict[str, torch.Tensor]:
        """
        Parses MATLAB (.mat) CFD exports containing spatial coordinates and continuous fields.
        Returns a dictionary of normalized tensors ready for the PINN loss function.
        """
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"[Error] CFD reference geometry not found at: {filepath}")
            
        try:
            # Ingest structural MATLAB arrays
            mat_data = sio.loadmat(filepath)
            
            # Extract and normalize continuous thermofluid fields
            tensors = {
                'X': self._normalize_field(mat_data.get('x_coords', np.zeros(1)), 'X'),
                'Y': self._normalize_field(mat_data.get('y_coords', np.zeros(1)), 'Y'),
                'U': self._normalize_field(mat_data.get('velocity_u', np.zeros(1)), 'U'),
                'V': self._normalize_field(mat_data.get('velocity_v', np.zeros(1)), 'V'),
                'P': self._normalize_field(mat_data.get('pressure', np.zeros(1)), 'P'),
                'T_fluid': self._normalize_field(mat_data.get('temperature_f', np.zeros(1)), 'T_fluid'),
                'T_solid': self._normalize_field(mat_data.get('temperature_s', np.zeros(1)), 'T_solid')
            }
            
            print(f"[System] Successfully compiled {tensors['X'].shape[0]} high-fidelity data points.")
            print(f"[System] Tensors allocated to: {self.device}")
            
            return tensors
            
        except Exception as e:
            raise RuntimeError(f"Failed to compile CFD dataset. Ensure valid matrix dimensions. Details: {str(e)}")
