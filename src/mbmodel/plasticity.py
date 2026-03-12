__author__ = "Kavya Velliangiri"
__credits__ = ["Kavya Velliangiri"]
__license__ = "MIT"
__version__ = "v3"
__maintainer__ = "Kavya Velliangiri"

"""Dopaminergic plasticity rules for KC->MBON synapses.

Two implementations:
  - GkaniasDPR : Dopaminergic Plasticity Rule from Gkanias et al. (2022) with
                 fast (D_△) and slow (D_▽) DA components.
  - DopaminePlasticity : simplified single-DA scalar rule (original v1).

Reference equations (thesis eqs 5-7, based on Gkanias et al., 2022):

  Fast DA component (eq. D_△, DopR1/cAMP pathway):
    τ_short · dD_△/dt = d(t) - D_△        [fast DA tracker, τ = 200 ms]

  Slow DA component (eq. D_▽, DopR2/ER-Ca²⁺ pathway):
    τ_long  · dD_▽/dt = d(t) - D_▽        [slow DA tracker, τ = 2000 ms]

  Dopaminergic factor (thesis eq 7):
    δ_j(t) = D_▽_j(t) - D_△_j(t)          [slow minus fast]

  Weight update rule (DPR, thesis eq 8):
    dW_ij/dt = η · δ_j(t) · [k_i(t) + W_ij(t) - w_rest]

The sign convention δ = D_▽ − D_△ (slow minus fast) ensures that
coincidence of KC activity with a fast DA transient drives depression
(δ < 0), matching Handler et al. (2019) forward-conditioning results.

where d_j(t) is the dopamine signal at MBON j's compartment, k_i(t) is KC i
activity, W_ij(t) is the KC_i -> MBON_j synaptic weight, and w_rest is the
resting/baseline weight.
"""

import numpy as np
from mbmodel.utils import euler_step


class GkaniasDPR:
    """Dopaminergic Plasticity Rule (Gkanias et al., 2022).

    Models synaptic plasticity at KC->MBON synapses via two dopamine
    components that track the DA signal at different timescales:

      Fast DA component D_△ (DopR1/cAMP, τ_short):
        τ_short · dD_△_j/dt = d_j(t) - D_△_j

      Slow DA component D_▽ (DopR2/ER-Ca²⁺, τ_long):
        τ_long · dD_▽_j/dt = d_j(t) - D_▽_j

      Dopaminergic factor (thesis eq 7):
        δ_j(t) = D_▽_j(t) - D_△_j(t)       [slow minus fast]

      Weight update (DPR):
        dW_ij/dt = η · δ_j(t) · [k_i(t) + W_ij(t) - w_rest]

    The sign of δ determines depression (δ < 0) or potentiation (δ > 0).
    During DA onset, D_△ (fast) rises before D_▽ (slow) → δ < 0 → depression.
    After DA offset, D_△ decays first, D_▽ lingers → δ > 0 → potentiation.

    Parameters
    ----------
    learning_rate : float
        Global learning rate η (default: 0.01)
    w_rest : float
        Resting synaptic weight w_rest (default: 1.0)
    tau_short : float
        Fast DA filter time constant τ_short in ms (default: 200.0)
    tau_long : float
        Slow DA filter time constant τ_long in ms (default: 2000.0)
    w_min, w_max : float
        Weight bounds
    """

    def __init__(
        self,
        learning_rate=0.01,
        w_rest=1.0,
        tau_short=200.0,
        tau_long=2000.0,
        w_min=0.0,
        w_max=10.0,
        kc_gating_only=False,
    ):
        self.learning_rate = learning_rate
        self.w_rest = w_rest
        self.tau_short = tau_short
        self.tau_long = tau_long
        self.w_min = w_min
        self.w_max = w_max
        self.kc_gating_only = kc_gating_only

        # State variables, initialized lazily on first update
        self.D_up = None    # D_△ (fast, τ_short, DopR1/cAMP), shape (n_mbon,)
        self.D_down = None  # D_▽ (slow, τ_long, DopR2/ER-Ca²⁺), shape (n_mbon,)
        self._buf = None    # reusable (n_mbon, n_kc) work buffer; avoids ~5MB alloc/step

    # ------------------------------------------------------------------
    # DA signal update
    # ------------------------------------------------------------------

    def update(self, da_per_mbon, kc_rates, W, dt):
        """Update D_△, D_▽ and synaptic weights for one timestep.

        Parameters
        ----------
        da_per_mbon : ndarray, shape (n_mbon,)
            Dopamine input to each MBON: W_DAN_MBON @ DANs.r
        kc_rates : ndarray, shape (n_kc,)
            KC firing rates k_i(t)
        W : ndarray, shape (n_mbon, n_kc)
            Current KC->MBON weight matrix
        dt : float
            Timestep in ms

        Returns
        -------
        W : ndarray, shape (n_mbon, n_kc)
            Updated weight matrix
        """
        # Lazy initialization
        if self.D_up is None:
            self.D_up   = np.zeros_like(da_per_mbon)
            self.D_down = np.zeros_like(da_per_mbon)
        if self._buf is None:
            self._buf = np.empty_like(W)

        # τ_short · dD_△/dt = d - D_△
        self.D_up = euler_step(self.D_up, da_per_mbon, self.tau_short, dt)

        # τ_long · dD_▽/dt = d - D_▽
        self.D_down = euler_step(self.D_down, da_per_mbon, self.tau_long, dt)

        # Dopaminergic factor: δ_j = D_▽_j - D_△_j  (thesis eq 7)
        # D_▽ (slow, τ_long) minus D_△ (fast, τ_short).
        # During DA onset, D_△ rises faster → δ < 0 → depression.
        # After DA offset, D_△ decays faster, D_▽ lingers → δ > 0 → potentiation.
        # Matches Handler et al. (2019): DopR1 (fast) drives acquisition/depression.
        delta = self.D_down - self.D_up  # (n_mbon,)

        # DPR weight update (in-place to avoid ~5 MB of temporaries per step):
        if self.kc_gating_only:
            # MV-compatible form (Bennett et al. 2021): dW_ij = η · dt · δ_j · k_i
            # Uses only KC activity for gating — stable for bidirectional
            # plasticity (no runaway from W-dependent positive feedback).
            np.copyto(self._buf, kc_rates[np.newaxis, :])        # buf = k
        else:
            # Original Gkanias DPR: dW_ij = η · dt · δ_j · [k_i + W_ij − w_rest]
            np.add(kc_rates[np.newaxis, :], W, out=self._buf)    # buf = k + W
            self._buf -= self.w_rest                               # buf = k + W - w_rest
        self._buf *= (self.learning_rate * dt) * delta[:, np.newaxis]  # buf = dW
        W += self._buf                                        # W updated in-place
        np.clip(W, self.w_min, self.w_max, out=W)
        return W

    @property
    def delta(self):
        """Current dopaminergic factor δ = D_▽ - D_△ (thesis eq 7)."""
        if self.D_up is None:
            return None
        return self.D_down - self.D_up

    def reset(self):
        """Reset DA state to zero (keep buffers allocated for reuse)."""
        if self.D_up is not None:
            self.D_up[:]   = 0.0
            self.D_down[:] = 0.0
        # _buf intentionally kept; W shape is constant across trials


# ---------------------------------------------------------------------------
# Simple baseline rule (v1 -- kept for reference)
# ---------------------------------------------------------------------------

class DopaminePlasticity:
    """Simple dopamine-modulated anti-Hebbian plasticity.

    Single scalar DA signal; weight update:
        dW/dt = η · (DA - DA_baseline) · k_i · r_j

    Parameters
    ----------
    learning_rate : float
    DA_baseline : float
        Tonic DA level (Hz)
    tau_DA : float
        DA clearance time constant (ms)
    w_min, w_max : float
    """

    def __init__(
        self,
        learning_rate=0.001,
        DA_baseline=20.0,
        tau_DA=500.0,
        w_min=0.0,
        w_max=10.0,
    ):
        self.eta = learning_rate
        self.DA_baseline = DA_baseline
        self.tau_DA = tau_DA
        self.w_min = w_min
        self.w_max = w_max
        self.DA = DA_baseline

    def update_DA(self, r_DAN, dt):
        r_mean = np.mean(r_DAN) if isinstance(r_DAN, np.ndarray) else r_DAN
        self.DA = euler_step(self.DA, r_mean, self.tau_DA, dt)

    def update_weights(self, W, r_pre, r_post, dt):
        DA_signal = self.DA - self.DA_baseline
        dW = self.eta * np.outer(r_post, r_pre) * DA_signal * dt
        W = np.clip(W + dW, self.w_min, self.w_max)
        return W

    def reset(self):
        self.DA = self.DA_baseline
