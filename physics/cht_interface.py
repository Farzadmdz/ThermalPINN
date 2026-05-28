import torch
from torch.autograd import grad
from typing import Dict, Tuple

class CHTInterfaceEvaluator:
    """
    Differentiable Physics-Informed evaluator for Conjugate Heat Transfer (CHT) 
    and phase-change kinetics at the solid-fluid interface.
    
    Enforces Neumann and Dirichlet boundary constraints, specifically formulated 
    for advanced thermal spreaders and bi-porous vapor chambers where 
    latent heat of vaporization dictates the interfacial heat flux.
    """
    
    def __init__(self, k_solid: float = 400.0, k_fluid: float = 0.6, h_fg: float = 2260000.0):
        """
        Initializes interfacial thermodynamic properties.
        Default values represent a Copper-Water working pair. For advanced composites 
        (e.g., graphene-copper), k_solid should be adjusted to the effective conductivity.
        
        Args:
            k_solid: Thermal conductivity of the solid wick structure [W/m.K]
            k_fluid: Thermal conductivity of the working fluid [W/m.K]
            h_fg: Latent heat of vaporization for phase change [J/kg]
        """
        self.k_s = torch.tensor(k_solid, requires_grad=False)
        self.k_f = torch.tensor(k_fluid, requires_grad=False)
        self.h_fg = torch.tensor(h_fg, requires_grad=False)

    def _get_normal_gradients(self, field: torch.Tensor, coords: torch.Tensor, normal_vector: Tuple[float, float]) -> torch.Tensor:
        """
        Computes the directional derivative (gradient normal to the interface) 
        using continuous automatic differentiation.
        """
        gradients = grad(field, coords, grad_outputs=torch.ones_like(field), 
                         create_graph=True, retain_graph=True)[0]
        
        grad_x, grad_y = gradients[:, 0:1], gradients[:, 1:2]
        nx, ny = normal_vector
        
        # Dot product: grad(T) . n
        normal_grad = grad_x * nx + grad_y * ny
        return normal_grad

    def evaluate_interface_residuals(self, 
                                     coords_interface: torch.Tensor, 
                                     T_solid: torch.Tensor, 
                                     T_fluid: torch.Tensor, 
                                     mass_flux: torch.Tensor = None,
                                     normal_vector: Tuple[float, float] = (0.0, 1.0)) -> Dict[str, torch.Tensor]:
        """
        Computes the physics residuals at the solid-fluid boundary.
        
        Args:
            coords_interface: Spatiotemporal coordinates precisely at the interface boundary
            T_solid: Predicted solid temperature field tensor
            T_fluid: Predicted fluid/vapor temperature field tensor
            mass_flux: Predicted evaporation/condensation mass flux [kg/m^2.s]
            normal_vector: Unit normal vector pointing from solid to fluid
            
        Returns:
            Dictionary containing temperature continuity and heat flux conservation residuals.
        """
        
        # 1. Temperature Continuity Constraint (Dirichlet Coupling)
        # Mathematical representation: T_fluid = T_solid
        res_temp_continuity = T_solid - T_fluid
        
        # 2. Heat Flux Conservation with Phase Change (Neumann Coupling)
        # Calculate normal temperature gradients exactly via Autograd
        dT_dn_solid = self._get_normal_gradients(T_solid, coords_interface, normal_vector)
        dT_dn_fluid = self._get_normal_gradients(T_fluid, coords_interface, normal_vector)
        
        # Fourier's Law at the interface
        q_solid = -self.k_s * dT_dn_solid
        q_fluid = -self.k_f * dT_dn_fluid
        
        # Interfacial Energy Balance: q_solid = q_fluid + (m_dot * h_fg)
        if mass_flux is not None:
            phase_change_energy = mass_flux * self.h_fg
            res_flux_conservation = q_solid - q_fluid - phase_change_energy
        else:
            # Pure sensible conjugate heat transfer (no phase change)
            res_flux_conservation = q_solid - q_fluid
            
        return {
            'interface_temperature_residual': res_temp_continuity,
            'interface_flux_residual': res_flux_conservation
        }
