import torch
from torch.autograd import grad
from typing import Tuple, Dict

class ThermofluidPhysicsEvaluator:
    """
    Advanced Physics-Informed residual evaluator for Conjugate Heat Transfer (CHT) 
    and incompressible fluid dynamics within free-flow and bi-porous media domains.
    
    This module computes the exact partial differential equation (PDE) residuals 
    using PyTorch's automatic differentiation (Autograd) engine, eliminating the 
    need for spatial discretization or traditional finite-volume meshing.
    """
    def __init__(self, fluid_props: Dict[str, float], porous_props: Dict[str, float] = None):
        """
        Initialize the physical evaluator with thermodynamic and material properties.
        """
        self.rho = fluid_props.get('density', 998.2)          # kg/m^3
        self.mu = fluid_props.get('dynamic_viscosity', 1e-3)  # Pa.s
        self.cp = fluid_props.get('specific_heat', 4182.0)    # J/(kg.K)
        self.k_f = fluid_props.get('thermal_conductivity', 0.6) # W/(m.K)
        self.nu = self.mu / self.rho
        self.alpha = self.k_f / (self.rho * self.cp)
        
        # Bi-porous wick properties (Darcy-Forchheimer terms)
        if porous_props:
            self.permeability = porous_props.get('permeability', 1e-10) # m^2
            self.forchheimer_coeff = porous_props.get('forchheimer_coeff', 0.1)
            self.porosity = porous_props.get('porosity', 0.6)
        else:
            self.permeability = None

    def _compute_derivatives(self, output: torch.Tensor, input_coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes first and second-order spatial derivatives via autograd.
        """
        # First-order derivatives: [d(out)/dx, d(out)/dy]
        grad_1st = grad(output, input_coords, grad_outputs=torch.ones_like(output), 
                        create_graph=True, retain_graph=True)[0]
        
        # Second-order derivatives for diffusion/viscous terms
        grad_2nd_x = grad(grad_1st[:, 0:1], input_coords, grad_outputs=torch.ones_like(grad_1st[:, 0:1]), 
                          create_graph=True, retain_graph=True)[0][:, 0:1]
        grad_2nd_y = grad(grad_1st[:, 1:2], input_coords, grad_outputs=torch.ones_like(grad_1st[:, 1:2]), 
                          create_graph=True, retain_graph=True)[0][:, 1:2]
        
        return grad_1st, (grad_2nd_x, grad_2nd_y)

    def compute_residuals(self, coords: torch.Tensor, preds: Tuple[torch.Tensor, ...], is_porous: bool = False) -> Dict[str, torch.Tensor]:
        """
        Evaluates the governing conservation laws mapping to the loss landscape.
        
        Args:
            coords: Spatiotemporal tensor [x, y]
            preds: Tuple containing (u, v, p, T_f) fields
            is_porous: Boolean flag triggering Darcy-Forchheimer momentum sinks
            
        Returns:
            Dictionary of coupled physics residuals (continuity, momentum, energy)
        """
        u, v, p, T_f = preds
        
        # Calculate gradients using rigorous tensor calculus
        u_grad, (u_xx, u_yy) = self._compute_derivatives(u, coords)
        v_grad, (v_xx, v_yy) = self._compute_derivatives(v, coords)
        p_grad, _ = self._compute_derivatives(p, coords)
        T_grad, (T_xx, T_yy) = self._compute_derivatives(T_f, coords)
        
        u_x, u_y = u_grad[:, 0:1], u_grad[:, 1:2]
        v_x, v_y = v_grad[:, 0:1], v_grad[:, 1:2]
        p_x, p_y = p_grad[:, 0:1], p_grad[:, 1:2]
        T_x, T_y = T_grad[:, 0:1], T_grad[:, 1:2]

        # 1. Mass Conservation (Continuity: div(U) = 0)
        res_continuity = u_x + v_y

        # Velocity magnitude for non-linear drag
        speed = torch.sqrt(u**2 + v**2 + 1e-8)

        # Porous media sink terms (Darcy-Forchheimer)
        sink_x, sink_y = 0.0, 0.0
        if is_porous and self.permeability:
            darcy_term = self.nu / self.permeability
            forchheimer_term = (self.forchheimer_coeff / torch.sqrt(torch.tensor(self.permeability))) * speed
            sink_x = -(darcy_term + forchheimer_term) * u
            sink_y = -(darcy_term + forchheimer_term) * v

        # 2. Momentum Conservation (Navier-Stokes X & Y)
        res_momentum_x = (u * u_x + v * u_y) + (1/self.rho) * p_x - self.nu * (u_xx + u_yy) - sink_x
        res_momentum_y = (u * v_x + v * v_y) + (1/self.rho) * p_y - self.nu * (v_xx + v_yy) - sink_y

        # 3. Energy Conservation (Convection-Diffusion)
        res_energy = (u * T_x + v * T_y) - self.alpha * (T_xx + T_yy)

        return {
            'continuity': res_continuity,
            'momentum_x': res_momentum_x,
            'momentum_y': res_momentum_y,
            'energy': res_energy
                          }
