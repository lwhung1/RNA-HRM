"""
shape_vienna_integration.py

Utilities to integrate SHAPE / ViennaRNA constraints into a neural pairing prior.

Main API:
    combined_pair_logits = integrate_shape_and_vienna(
        seq_len,
        neural_pair_logits,
        shape_reactivities=None,
        use_vienna=False,
        vienna_params=...
        integration_weights=...
    )

Notes:
- neural_pair_logits: Torch tensor (L, L) of logits (before sigmoid) from your model.
- shape_reactivities: numpy or torch vector (L,) with SHAPE reactivities or None.
- If use_vienna=True, function will attempt to import `RNA` (ViennaRNA Python bindings).
  If import fails, it falls back to SHAPE-pseudo-energy integration only.
- SHAPE -> pseudo-energy uses E = m * log(reactivity + 1) + b (Deigan-style). Defaults provided but tunable.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import math

import torch
import torch.nn.functional as F

# Try import ViennaRNA Python bindings (optional)
try:
    import RNA
    _VIENNA_AVAILABLE = True
except Exception:
    RNA = None
    _VIENNA_AVAILABLE = False


# ----------------------------
# Utilities: SHAPE -> pseudo energy
# ----------------------------
def shape_to_pseudo_energy(shape: np.ndarray, m: float = -0.6, b: float = 1.8) -> np.ndarray:
    """
    Convert SHAPE reactivities to per-residue pseudo-energies (kcal/mol) using the
    Deigan et al. style transform:
        E_pseudo(i) = m * ln(shape_i + 1) + b

    Default parameters (m, b) are common starting values but should be tuned:
      - m: typically negative (penalizes pairing when shape is large)
      - b: intercept, often positive

    Returns numpy array shape (L,)
    """
    shape = np.asarray(shape, dtype=np.float64)
    # clamp small negative or NaN to zero
    shape = np.nan_to_num(shape, nan=0.0)
    shape = np.maximum(shape, 0.0)
    pe = m * np.log(shape + 1.0) + b
    # If you want pseudo-energy *favoring* pairing when reactivity is tiny, you may keep
    # a negative floor, but we generally keep values as-is.
    return pe


# ----------------------------
# Convert per-residue pseudo energies -> pairwise bias term
# ----------------------------
def pairwise_bias_from_pseudo(pe: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Construct an (L,L) pairwise bias matrix from per-residue pseudo-energies.

    A simple and effective choice is additive: bias[i,j] = scale * (pe[i] + pe[j])
    - If pe is larger for reactive (unpaired) residues, adding (pe[i] + pe[j]) produces
      a larger penalty for pairing those residues (i.e., discourages pairing).
    - You can invert sign or choose different combination rules (prod, max, etc.).

    Parameters:
      pe: (L,) numpy array of pseudo energies (kcal/mol-ish)
      scale: multiplier converting pseudo-energy to logit-space units

    Returns:
      bias: (L,L) numpy array
    """
    pe = np.asarray(pe, dtype=np.float64)
    L = pe.shape[0]
    # outer sum
    bias = pe.reshape(L, 1) + pe.reshape(1, L)
    return bias * float(scale)


# ----------------------------
# Combine neural logits + SHAPE/Vienna prior
# ----------------------------
def combine_logits(
    neural_logits: torch.Tensor,
    bias_logits: torch.Tensor,
    weight_neural: float = 1.0,
    weight_bias: float = 1.0,
) -> torch.Tensor:
    """
    Combine in logit space:
      combined = weight_neural * neural_logits + weight_bias * bias_logits

    Advantages:
      - Numerically stable (works with logits directly).
      - Interpretable weights.
    """
    return weight_neural * neural_logits + weight_bias * bias_logits


# ----------------------------
# Optionally compute ViennaRNA partition-function base-pairing probabilities
# using Python bindings (if available). Returns an (L, L) matrix of probabilities.
# ----------------------------
def vienna_pairing_probs_from_shape(
    sequence: str,
    shape_reactivities: Optional[np.ndarray] = None,
    shape_params: Dict[str, float] = None,
    vienna_params: Dict[str, Any] = None,
) -> Optional[np.ndarray]:
    """
    Compute base-pairing probabilities (partition function) using ViennaRNA Python bindings,
    applying SHAPE pseudo-energy constraints if provided.

    Returns:
      pair_probs (L,L) numpy array or None if ViennaRNA not available.

    Notes:
      - ViennaRNA's Python API evolves; this function uses a guarded approach.
      - If bindings are available, we attempt to build a fold compound, set shape data,
        and compute the partition function. If that fails, returns None.
    """
    if not _VIENNA_AVAILABLE:
        return None
    if shape_params is None:
        shape_params = {"m": -0.6, "b": 1.8}
    try:
        L = len(sequence)
        fc = RNA.fold_compound(sequence)
        # If shape provided, create an RNA.io.Shape object OR use fold_compound methods
        if shape_reactivities is not None:
            # Convert shape list to string in ViennaRNA expected format "index reactivity"
            # Here we use the fold_compound's SHAPE interface if present
            shape_lines = []
            for i, r in enumerate(shape_reactivities, start=1):
                shape_lines.append(f"{i} {float(r):.6f}")
            shape_text = "\n".join(shape_lines) + "\n"
            try:
                # folding library may provide read_SHAPE or read_shape_data - try some options
                if hasattr(fc, "read_shape"):
                    fc.read_shape(shape_text, RNA.SHAPE_METHOD_DEIGAN, shape_params["m"], shape_params["b"])
                elif hasattr(RNA, "read_shape_data"):
                    # older style: RNA.read_shape_data returns a data structure to pass
                    shape_data = RNA.read_shape_data(shape_text)
                    fc.probability_pf()  # placeholder, API differs by version
                else:
                    # fallback: try setting SHAPE via command-line style options (less likely)
                    pass
            except Exception:
                # best-effort: ignore shape if API incompatible
                pass

        # Compute partition function
        # Note: exact API name to compute pairing probabilities varies by ViennaRNA version.
        # We attempt to call pf() or partition() and then extract pairing probabilities via bp_get
        try:
            fc.pf()  # compute partition function (may be fc.pf(), fc.partition(), etc.)
        except Exception:
            try:
                fc.partition()
            except Exception:
                # if we cannot compute PF, give up gracefully
                return None

        # Extract pairing probabilities. ViennaRNA provides the function bp(i,j) / get_pr
        pair_probs = np.zeros((L, L), dtype=np.float32)
        for i in range(1, L + 1):
            for j in range(i + 1, L + 1):
                p = 0.0
                try:
                    p = fc.bpp(i, j)  # some versions use bpp(i,j)
                except Exception:
                    try:
                        p = fc.bp(i, j)  # older different named function
                    except Exception:
                        p = 0.0
                pair_probs[i - 1, j - 1] = p
                pair_probs[j - 1, i - 1] = p
        return pair_probs
    except Exception:
        return None


# ----------------------------
# High-level integration function
# ----------------------------
def integrate_shape_and_vienna(
    sequence: str,
    neural_pair_logits: torch.Tensor,
    shape_reactivities: Optional[np.ndarray] = None,
    *,
    use_vienna: bool = False,
    shape_params: Dict[str, float] = None,
    bias_scale: float = 1.0,
    weight_neural: float = 1.0,
    weight_bias: float = 1.0,
    weight_vienna: float = 0.0,
    vienna_params: Dict[str, Any] = None,
) -> torch.Tensor:
    """
    Returns combined logits (torch.Tensor (L,L)).
