import os
from pathlib import Path
from typing import Dict, Optional, List, Any

import torch
import torch.nn as nn
import numpy as np
from pykeops.torch import LazyTensor


d3_autoang = 0.52917726  # for converting distance from bohr to angstrom
d3_autoev = 27.21138505  # for converting a.u. to eV

d3_k1 = 16.000
d3_k2 = 4 / 3
d3_k3 = -4.000
d3_maxc = 5  # maximum number of coordination complexes


def poly_smoothing(r: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Computes a smooth step from 1 to 0 starting at 1 bohr before the cutoff

    Args:
        r (Tensor): (n_edges,)
        cutoff (float): ()

    Returns:
        r (Tensor): Smoothed `r`
    """
    cuton = cutoff - 1
    x = (cutoff - r) / (cutoff - cuton)
    x2 = x**2
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    return torch.where(
        r <= cuton,
        torch.ones_like(x),
        torch.where(r >= cutoff, torch.zeros_like(x), 6 * x5 - 15 * x4 + 10 * x3),
    )


def ncoord(atomic_numbers, bond_distance, rcov, cnthr, k1, cutoff_smoothing=None):
    """
    Compute DFT-D3 coordination numbers using KeOps for efficient pairwise distance calculations.

    Parameters:
    - atomic_numbers (torch.Tensor): Tensor of shape (N,) with atomic numbers for each atom.
    - bond_distance (torch.Tensor): Tensor of shape (N, N) with distance matrix.
    - rcov (torch.Tensor or dict): Covalent radii for each atomic number, either as a tensor or lookup table.
    - cnthr (float): Cutoff radius for coordination number calculation (in same units as positions).
    - k1 (float): Steepness parameter for the damping function (default: 16.0, typical for DFT-D3).
    - cutoff_smoothing (str or None): Smoothing method for cutoff ('poly' or None).

    Returns:
    - torch.Tensor: Coordination numbers for each atom, shape (N,).
    """

    # Convert rcov to tensor if it's a dictionary
    if isinstance(rcov, dict):
        rcov = torch.tensor([rcov[z.item()] for z in atomic_numbers])
    else:
        #        rcov = rcov.to(device=device, dtype=dtype)
        rcov = rcov

    # Get covalent radii for each atom pair
    rcov_i = rcov[atomic_numbers - 1][..., None]
    rcov_j = rcov[atomic_numbers - 1][..., None]

    rcov_i = LazyTensor(rcov_i[:, None])  # Shape (N, 1)
    rcov_j = LazyTensor(rcov_j[None])  # Shape (1, N)
    rco = rcov_i + rcov_j  # Sum of covalent radii, shape (N, N)

    # Compute normalized distance
    rr = rco / bond_distance  # Shape (N, N)

    # Damping function: 1 / (1 + exp(-k1 * (rr - 1)))
    damp = 1.0 / (1.0 + (-k1 * (rr - 1.0)).exp())  # Shape (N, N)

    # Apply cutoff if specified
    if cnthr is not None:
        # Mask distances beyond cnthr
        mask = (R_ij <= cnthr).float()  # Shape (N, N), 1.0 if R_ij <= cnthr, else 0.0
        damp = damp * mask

        # Apply polynomial smoothing if requested
        if cutoff_smoothing == "poly":
            # Polynomial smoothing: 6th-order polynomial for smooth cutoff
            t = R_ij / cnthr
            poly = 6 * t**5 - 15 * t**4 + 10 * t**3
            smooth = (t <= 1.0).float() * poly  # Apply only where t <= 1
            damp = damp * smooth

    # Sum over j to get coordination number for each atom i
    cn = damp.sum(dim=1)  # Shape (N,)

    return cn.view(-1)  # Ensure output is shape (N,)


def getc6(Z, nc, c6ab, k3):
    Z = torch.tensor(Z, dtype=torch.int64, device="cpu")
    # N = Z.shape[0]
    # i, j = torch.meshgrid(
    #     torch.arange(N, device="cpu"), torch.arange(N, device="cpu"), indexing="ij"
    # )

    # Zi, Zj = Z[i], Z[j]  # (N,N)

    # nci, ncj = nc[i], nc[j]  # (N,N)

    # c6ij = c6ab[Zi - 1, Zj - 1]  # (N, N, 5, 5, 3)
    # cn0 = c6ij[..., 0]  # (N, N, 5, 5)
    # cn1 = c6ij[..., 1]
    # cn2 = c6ij[..., 2]

    # Reshape cn0, cn1, cn2 to (N, N, 1, 25) for PyKeOps compatibility
    cn0 = cn0.reshape(N, N, 1, 25)  # Flatten (5, 5) to 25
    cn1 = cn1.reshape(N, N, 1, 25)
    cn2 = cn2.reshape(N, N, 1, 25)

    nci_lazy = LazyTensor(nci[:, :, None, None])  # N,N, 1, 1
    ncj_lazy = LazyTensor(ncj[:, :, None, None])
    cn1_lazy = LazyTensor(cn1)  # N, N, 1, 25
    cn2_lazy = LazyTensor(cn2)

    r2 = (cn1_lazy - nci_lazy) ** 2 + (cn2_lazy - ncj_lazy) ** 2
    weights = (-k3 * r2).exp()
    weights_sum = weights.sum(dim=3)
    weights = weights / (weights_sum + 1e-8)

    # Perform the weighted sum using KeOps
    cn0_lazy = LazyTensor(cn0)  # Shape: (-1, 1, 25)
    c6 = (weights * cn0_lazy).sum(dim=3)  # Sum over the last dimension

    return c6


class KeopsDFTD3Module(nn.Module):
    def __init__(
        self,
        params: Dict[str, float],
        trainable: bool = False,
        dtype=torch.float32,
        device="cpu",
        return_c6: bool = False,
    ):
        """
        params (Dict): parameters for damping functions.
        trainable (bool): whether the damping parameters are trainable.
        dtype : data type of tensor.
        device: device of tensor.
        """
        super().__init__()

        d3_filepath = str(Path(os.path.abspath(__file__)).parent / "params" / "dftd3_params.npz")
        d3_params = np.load(d3_filepath)
        c6ab = torch.tensor(d3_params["c6ab"], dtype=dtype, device=device)
        r0ab = torch.tensor(d3_params["r0ab"], dtype=dtype, device=device)
        rcov = torch.tensor(d3_params["rcov"], dtype=dtype, device=device)
        r2r4 = torch.tensor(d3_params["r2r4"], dtype=dtype, device=device)

        a1 = params["rs6"]
        a2 = params["rs18"]

        s6 = params["s6"]
        s8 = params["s18"]

        # (95, 95, 5, 5, 3) c0, c1, c2 for coordination number dependent c6ab term.
        self.register_buffer("c6ab", c6ab)
        self.register_buffer("r0ab", r0ab)  # atom pair distance (95, 95)
        self.register_buffer("rcov", rcov)  # atom covalent distance (95)
        self.register_buffer("r2r4", r2r4)  # (95,)

        if trainable:
            # Define learnable parameters
            self.a1 = nn.Parameter(torch.tensor(a1, dtype=dtype, device=device))
            self.a2 = nn.Parameter(torch.tensor(a2, dtype=dtype, device=device))

            self.s6 = nn.Parameter(torch.tensor(s6, dtype=dtype, device=device))
            self.s8 = nn.Parameter(torch.tensor(s8, dtype=dtype, device=device))

        else:
            self.register_buffer("a1", torch.tensor(a1))
            self.register_buffer("a2", torch.tensor(a2))
            self.register_buffer("s6", torch.tensor(s6))
            self.register_buffer("s8", torch.tensor(s8))
        self.return_c6 = return_c6

    def calc_energy(self, atomic_numbers, positions, eps=1e-12):
        """
        atomic_numbers: (M,) tensor of atomic numbers
        positions: (M,3) tensor of particle positions
        returns: (M,) tensor of per-atom dispersion energies (self-terms removed)
        """
        a1 = LazyTensor(self.a1[None, None, None])
        a2 = LazyTensor(self.a2[None, None, None])
        s6 = LazyTensor(self.s6[None, None, None])
        s8 = LazyTensor(self.s8[None, None, None])

        # Compute pairwise C6 and C8 coefficients
        x_i = LazyTensor(positions[:, None])
        y_j = LazyTensor(positions[None])

        r2r4_i = self.r2r4[atomic_numbers][..., None]
        r2r4_j = self.r2r4[atomic_numbers][..., None]

        r2r4_i = LazyTensor(r2r4_i[:, None])
        r2r4_j = LazyTensor(
            r2r4_j[None],
        )

        # Compute pairwise distances
        d = (x_i - y_j).norm(dim=-1)
        d2 = d**2
        d6 = d2**3
        d8 = d6 * d2
        # d10 = d8 * d2

#        nc = ncoord(atomic_numbers, d, self.rcov, None, d3_k1, "none")
#        print("nc", nc.shape)

#        c6 = getc6(atomic_numbers, nc, self.c6ab, d3_k3)[...,0]
#        print("c6", c6.shape)

#        c8 = 3 * c6 * (r2r4_i * r2r4_j)
#        print("c8", c8.shape)
        c6 = LazyTensor(torch.ones(positions.shape[0], positions.shape[0]), axis=1)
        c8 = LazyTensor(torch.ones(positions.shape[0], positions.shape[0]), axis=1)
        # BJ damping function: R_ij^n / (R_ij^n + (a1 * R0 + a2)^n)
        tmp = a1 * (c8 / c6).sqrt() + a2
        tmp2 = tmp**2
        tmp6 = tmp2**3
        tmp8 = tmp6 * tmp2
        d6_damped = 1 / (d6 + tmp6 + eps)
        d8_damped = 1 / (d8 + tmp8 + eps)
        # Apply mask to remove self-interactions
        mask = d > eps

        # Compute dispersion energy with scaling factors
        D_ij = -s6 * c6 * d6_damped - s8 * c8 * d8_damped
        D_ij = D_ij * mask
        # Sum over j to get per-atom energies
        a_i = D_ij.sum(dim=1)
        print("ai.shape", a_i.shape)
        if self.return_c6:
            return a_i.sum(), c6
        return a_i.sum()

    def calc_energy_and_forces(
        self,
        Z: torch.Tensor,
        pos: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """Forward computation of dispersion energy, force and stress

        Args:
            Z (Tensor): (n_atoms,) atomic numbers.
            pos (Tensor): atom positions in angstrom
            cell (Tensor): cell size in angstrom, None for non periodic system.
            pbc (Tensor): pbc condition, None for non periodic system.
            shift_pos (Tensor):  (n_atoms, 3) shift vector (length unit).
            damping (str): damping method. "zero", "bj", "zerom", "bjm"

        Returns:
            results (list): calculated results. Contains following:
                "energy": ()
                "forces": (n_atoms, 3)
                "stress": (6,)
        """
        pos.requires_grad_(True)

        if self.return_c6:
            E_disp, C6 = self.forward(Z, pos)
        else:
            E_disp = self.forward(Z, pos)

        E_disp.sum().backward()
        forces = -pos.grad  # [eV/angstrom]
        results_list = (
            [
                {
                    "energy": E_disp.item(),
                    "forces": forces.cpu().numpy(),
                    "atC6": C6.cpu().numpy(),
                }
            ]
            if self.return_c6
            else [{"energy": E_disp.item(), "forces": forces.cpu().numpy()}]
        )
        return results_list
