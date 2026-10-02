"""
Data module for scHiC latent diffusion.

Provides dataset loading from .scool format (cooler library).
"""

from .schic import (
    ScHiCDataset,
    sparse_to_dense,
    create_schic_dataloader,
    create_schic_dataloader_from_config,
)

__all__ = [
    "ScHiCDataset",
    "sparse_to_dense",
    "create_schic_dataloader",
    "create_schic_dataloader_from_config",
]
