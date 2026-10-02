"""
Geometric-informed latent diffusion model for single-cell Hi-C data.

Implements the EM-based framework from diffusion_scHiC.md:
  - ZINB observation model with geometric low-rank structure
  - Score network trained via denoising score matching (DSM)
  - EM algorithm: MAP E-step + ZINB/DSM M-steps
  - Likelihood-guided DDIM inference
"""

__version__ = "0.1.0"
