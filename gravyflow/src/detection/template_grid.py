"""
Template Grid for Matched Filtering

Defines parameter space coverage for template-based gravitational wave searches.
"""

from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp


@dataclass
class TemplateGrid:
    """
    Defines parameter space grid for matched filtering templates.
    
    Uses a simple uniform grid in component masses. For production use,
    metric-based template placement would be more efficient.
    
    Args:
        mass_1_range: (min, max) for primary mass in solar masses
        mass_2_range: (min, max) for secondary mass in solar masses
        num_mass_1_points: Grid resolution for mass_1
        num_mass_2_points: Grid resolution for mass_2 (defaults to mass_1)
    
    Usage:
        >>> grid = TemplateGrid(mass_1_range=(5.0, 75.0), mass_2_range=(5.0, 75.0))
        >>> m1, m2 = grid.get_parameters()
        >>> print(f"Number of templates: {len(m1)}")
    """
    
    mass_1_range: Tuple[float, float] = (5.0, 75.0)
    mass_2_range: Tuple[float, float] = (5.0, 75.0)
    num_mass_1_points: int = 32
    num_mass_2_points: int = None  # Defaults to num_mass_1_points
    
    def __post_init__(self):
        if self.num_mass_2_points is None:
            self.num_mass_2_points = self.num_mass_1_points
    
    def get_parameters(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate template parameters covering the mass space.
        
        Enforces m1 >= m2 convention to avoid duplicate templates.
        
        Returns:
            Tuple of (mass_1, mass_2) arrays, each of shape (num_templates,)
        """
        m1_vals = jnp.linspace(
            self.mass_1_range[0], 
            self.mass_1_range[1], 
            self.num_mass_1_points
        )
        m2_vals = jnp.linspace(
            self.mass_2_range[0], 
            self.mass_2_range[1], 
            self.num_mass_2_points
        )
        
        # Create meshgrid
        m1_grid, m2_grid = jnp.meshgrid(m1_vals, m2_vals, indexing='ij')
        
        # Flatten
        m1_flat = m1_grid.ravel()
        m2_flat = m2_grid.ravel()
        
        # Enforce m1 >= m2 to avoid duplicates
        mask = m1_flat >= m2_flat
        
        return m1_flat[mask], m2_flat[mask]
    
    @property
    def num_templates(self) -> int:
        """Total number of templates in grid."""
        m1, _ = self.get_parameters()
        return len(m1)
    
    def get_chirp_masses(self) -> jnp.ndarray:
        """
        Get chirp mass for each template.
        
        M_chirp = (m1 * m2)^(3/5) / (m1 + m2)^(1/5)
        """
        m1, m2 = self.get_parameters()
        return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
