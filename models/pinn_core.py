import torch
import torch.nn as nn
from torch.autograd import grad
from typing import List, Tuple

class ThermalPINNSolver(nn.Module):
    """
    Core Physics-Informed Neural Network (PINN) architecture for solving 
    Conjugate Heat Transfer (CHT) and fluid flow in bi-porous vapor chambers.
    
    This module implicitly embeds the Navier-Stokes and Energy equations 
    into the loss landscape, bypassing traditional finite-volume meshing.
    """
    def __init__(self, layers: List[int], use_gpu: bool = True):
        super(ThermalPINNSolver, self).__init__()
        
        # Hardware acceleration allocation
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.activation = nn.Tanh()
        
        # Deep Neural Network Architecture Construction
        self.linears = nn.ModuleList([
            nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)
        ])
        
        # Physical Parameters (e.g., Copper wick, working fluid properties)
        self.k_solid = torch.tensor(400.0, device=self.device)  # Thermal conductivity (W/m.K)
        self.rho_fluid = torch.tensor(998.2, device=self.device) # Density (kg/m^3)
        self.mu_fluid = torch.tensor(0.001, device=self.device)  # Dynamic viscosity (Pa.s)
        
        self.to(self.device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass mapping spatial coordinates to thermofluid state fields.
        
        Returns:
            Tuple containing continuous fields: (u, v, p, T_fluid, T_solid)
        """
        inputs = torch.cat([x, y], dim=1)
        for i in range(len(self.linears) - 1):
            inputs = self.activation(self.linears[i](inputs))
        
        outputs = self.linears[-1](inputs)
        
        # Decoupling the output tensor into specific physical fields
        u = outputs[:, 0:1]     # x-velocity
        v = outputs[:, 1:2]     # y-velocity
        p = outputs[:, 2:3]     # Pressure field
        T_f = outputs[:, 3:4]   # Fluid temperature
        T_s = outputs[:, 4:5]   # Solid (wick) temperature
        
        return u, v, p, T_f, T_s

    def compute_gradients(self, field: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
        """
        Computes exact spatial derivatives using PyTorch's automatic differentiation engine.
        Essential for evaluating the PDE residuals.
        """
        return grad(field, coordinate, grad_outputs=torch.ones_like(field), 
                    create_graph=True, retain_graph=True)[0]
