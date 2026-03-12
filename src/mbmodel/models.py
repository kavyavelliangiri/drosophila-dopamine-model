__author__ = "Kavya Velliangiri"
__credits__ = ["Kavya Velliangiri"]
__license__ = "MIT"
__version__ = "v3"
__maintainer__ = "Kavya Velliangiri"

"""Rate-based neuron population and mushroom body network.

Implements the hybridized model from the review (Huang et al., 2024 network
dynamics + Gkanias et al., 2022 plasticity rule).

Key equations (approximated steady-state forms, Huang et al., 2024):

  KC (eq 5.1 → 6.1):
    τ_KC · d(Δx_KC)/dt = w_odor(t) · x_odor(t) − Δx_KC

  DAN (eq 5.2 → 6.2):
    τ_DAN · d(Δx_DAN)/dt = w_punish · x_punish(t)
                           + Σ_i w_KD,ij · Δx_KC,i
                           + Σ_l w_MD,lj · Δx_MBON,l
                           − Δx_DAN

  MBON (eq 5.3 → 6.3):
    τ_MBON · d(Δx_MBON)/dt = f_a,MBON( Σ_i w_KM,ij(t) · Δx_KC,i
                                        + Σ_l w_MM,lj · Δx_MBON,l
                                        + B_MBON )
                              − B_MBON − Δx_MBON

  MBON activation (piecewise-linear, eq 3.3):
    f_a,MBON(x) = clip(x, 0, M_MBON)

  KC / DAN activation (linear, eqs 3.1-3.2):
    f_a,KC = f_a,DAN = x
"""

import numpy as np
from mbmodel.utils import relu, euler_step
from mbmodel.plasticity import GkaniasDPR


class NeuronPopulation:
    """Rate-based neuron population.

    Implements:
        τ · dr/dt = f_a(I) − r

    where f_a is 'relu' (linear + clamp, for KCs/MBONs) or 'linear' (for
    KCs/DANs in the Huang et al. approximation).

    Parameters
    ----------
    n : int
        Number of neurons.
    tau : float
        Decay time constant (ms).
    r_max : float
        Maximum firing rate (Hz); used by relu activation (eq 3.3).
    activation : str
        'relu'   -- piecewise linear: clip(x, 0, r_max)  [MBONs, eq 3.3]
        'linear' -- identity (no clamp)                  [KCs, DANs, eqs 3.1-3.2]
    name : str, optional
    """

    def __init__(self, n, tau, r_max, activation, name=None):
        self.n = n
        self.tau = tau
        self.r_max = r_max
        self.activation = activation
        self.name = name or f"NeuronPopulation{n}"
        self.r = np.zeros(n)

    def update(self, input_current, dt):
        """One Euler step: τ · dr/dt = f_a(I) − r.

        Parameters
        ----------
        input_current : ndarray, shape (n,)
        dt : float
            Timestep (ms)
        """
        if self.activation == 'relu':
            target = relu(input_current, self.r_max)
        elif self.activation == 'linear':
            target = input_current
        else:
            raise ValueError(f"Unknown activation: {self.activation!r}")

        self.r = euler_step(self.r, target, self.tau, dt)
        return self.r

    def reset(self):
        self.r = np.zeros(self.n)

    def __repr__(self):
        return f"<{self.name} mean={np.mean(self.r):.2f} Hz>"


class MushroomBodyNetwork:
    """Full mushroom body circuit with Gkanias DPR plasticity.

    Network connectivity:
      - KC  -> MBON  : plastic weights W_KC_MBON  (n_mbon × n_kc)
      - KC  -> DAN   : fixed weights   W_KC_DAN   (n_dan  × n_kc)   [optional]
      - DAN -> MBON  : fixed weights   W_DAN_MBON (n_mbon × n_dan)  [optional]
      - MBON-> DAN   : fixed weights   W_MBON_DAN (n_dan  × n_mbon) [optional]
      - MBON-> MBON  : fixed weights   W_MBON_MBON(n_mbon × n_mbon) [optional]

    Activation functions:
      - KCs  : linear   (Huang et al. eq 3.1)
      - DANs : linear   (Huang et al. eq 3.2)
      - MBONs: relu     (piecewise linear, Huang et al. eq 3.3)

    Parameters
    ----------
    config : dict
        Network and plasticity parameters.  Keys:
          dt, tau_KC, tau_MBON, tau_DAN,
          r_max_KC, r_max_MBON, r_max_DAN,
          n_KCs, n_MBONs, n_DANs  (if no connectome),
          connectome_path, ids_path  (optional),
          learning_rate, w_rest, tau_short, tau_long
    """

    def __init__(self, config):
        from mbmodel.connectivity import create_random_sparse

        self.config = config
        self.dt = config.get('dt', 0.1)

        # ---- load / build connectivity -----------------------------------
        if 'connectome_path' in config:
            W_data = np.load(config['connectome_path'], allow_pickle=True)
            ids_data = np.load(config['ids_path'], allow_pickle=True)

            def _load(key):
                """Load matrix and normalise by the maximum row-sum.

                Dividing by max row-sum ensures that the maximum total
                synaptic drive from one pathway into any single postsynaptic
                neuron is ≤ 1.0, preventing runaway amplification in
                recurrent pathways (e.g. DAN→DAN spectral radius > 1).
                """
                M = W_data[key].astype(float)
                row_max = M.sum(axis=1).max()
                return M / row_max if row_max > 0 else M

            # Plastic weights
            self.W_KC_MBON   = _load('kc_to_mbon')    # (n_mbon, n_kc)
            # Fixed weights – all 6 non-plastic pathways
            self.W_DAN_MBON  = _load('dan_to_mbon')   # (n_mbon, n_dan)
            self.W_KC_DAN    = _load('kc_to_dan')     # (n_dan,  n_kc)
            self.W_MBON_DAN  = _load('mbon_to_dan')   # (n_dan,  n_mbon)
            self.W_MBON_MBON = _load('mbon_to_mbon')  # (n_mbon, n_mbon)
            # Scale W_DAN_DAN by 0.5 so max row-sum = 0.5 < 1.0.
            # Row-sum normalization alone gives max row-sum = 1.0, which pins
            # the most-connected DAN at r_max after shock offset (recurrent
            # input = 1.0 × r_max → relu target = r_max → no decay).
            # With max row-sum = 0.5, DANs decay to baseline within ~3 τ_DAN.
            self.W_DAN_DAN   = _load('dan_to_dan') * 0.5  # (n_dan,  n_dan)

            self.kc_ids   = ids_data['kc_ids']
            self.mbon_ids = ids_data['mbon_ids']
            self.dan_ids  = ids_data['dan_ids']

            n_kc   = self.W_KC_MBON.shape[1]
            n_mbon = self.W_KC_MBON.shape[0]
            n_dan  = self.W_DAN_MBON.shape[1]
        else:
            n_kc   = config['n_KCs']
            n_mbon = config['n_MBONs']
            n_dan  = config['n_DANs']
            self.W_KC_MBON   = create_random_sparse(n_kc,   n_mbon, sparsity=0.1)
            self.W_DAN_MBON  = create_random_sparse(n_dan,  n_mbon, sparsity=0.3)
            self.W_KC_DAN    = np.zeros((n_dan,  n_kc))
            self.W_MBON_DAN  = np.zeros((n_dan,  n_mbon))
            self.W_MBON_MBON = np.zeros((n_mbon, n_mbon))
            self.W_DAN_DAN   = np.zeros((n_dan,  n_dan))

        # ---- neuron populations -----------------------------------------
        # KCs and DANs use linear activation (Huang et al. eqs 3.1, 3.2)
        # MBONs use piecewise-linear / relu (Huang et al. eq 3.3)
        self.KCs = NeuronPopulation(
            n=n_kc,
            tau=config.get('tau_KC', 10.0),
            r_max=config.get('r_max_KC', 100.0),
            activation='linear',
            name='KCs',
        )
        self.MBONs = NeuronPopulation(
            n=n_mbon,
            tau=config.get('tau_MBON', 20.0),
            r_max=config.get('r_max_MBON', 100.0),
            activation='relu',
            name='MBONs',
        )
        # DANs use relu to prevent runaway amplification from the recurrent
        # W_DAN_DAN loop (spectral radius > 1 in biological connectome data).
        # The linear-activation approximation (Huang et al. eq 3.2) holds for
        # small perturbations but requires clamping for biological realism.
        self.DANs = NeuronPopulation(
            n=n_dan,
            tau=config.get('tau_DAN', 15.0),
            r_max=config.get('r_max_DAN', 100.0),
            activation='relu',
            name='DANs',
        )

        self.W_initial = self.W_KC_MBON.copy()

        # ---- plasticity (Gkanias et al., 2022 DPR) ----------------------
        self.plasticity = GkaniasDPR(
            learning_rate=config.get('learning_rate', 0.01),
            w_rest=config.get('w_rest', 1.0),
            tau_short=config.get('tau_short', 200.0),
            tau_long=config.get('tau_long', 2000.0),
            kc_gating_only=config.get('kc_gating_only', False),
        )

    # ------------------------------------------------------------------
    def set_compartmental_plasticity(self, aversive_dan_idx, appetitive_dan_idx,
                                     mbon_sign_mask):
        """Enable compartment-specific valence-opponent plasticity.

        Implements the mixed-valence (MV) model from Bennett et al. (2021):
        the effective DA signal at each MBON is sign-flipped according to
        compartment identity, so that aversive DA (PPL1) depresses approach
        MBONs but potentiates avoid MBONs, and vice versa for appetitive
        DA (PAM).

        Effective DA per MBON:
            da_eff_j = sign_j · (da_aversive_j − da_appetitive_j)

        where sign_j = +1 for approach MBONs, −1 for avoid MBONs.
        The existing GkaniasDPR then processes da_eff normally:
        positive input → D_up rises fast → δ<0 → depression;
        negative input → D_up drops fast → δ>0 → potentiation.

        Parameters
        ----------
        aversive_dan_idx : list of int
            DAN indices for aversive neurons (e.g., PPL1).
        appetitive_dan_idx : list of int
            DAN indices for appetitive neurons (e.g., PAM).
        mbon_sign_mask : ndarray, shape (n_mbon,)
            +1 for approach MBONs, −1 for avoid MBONs, 0 for unclassified.
        """
        self._avr_dan_idx = np.asarray(aversive_dan_idx)
        self._app_dan_idx = np.asarray(appetitive_dan_idx)
        self._mbon_sign   = np.asarray(mbon_sign_mask, dtype=float)
        self._compartmental = True

    # ------------------------------------------------------------------
    def step(self, I_odor, x_punish=0.0):
        """Advance the network by one timestep.

        Parameters
        ----------
        I_odor : ndarray, shape (n_KC,)
            Odor-driven input current to KCs (= w_odor · x_odor, eq 6.1).
        x_punish : float or ndarray, shape (n_DAN,), optional
            Punishment signal (1 during shock, 0 otherwise).
            Scaled internally by w_punish = 1.0; override in config if needed.
        """
        dt = self.dt

        # -- KC update (eq 5.1 / 6.1) ------------------------------------
        self.KCs.update(I_odor, dt)

        # -- DAN update (eq 5.2 / 6.2) ------------------------------------
        # I_DAN = w_punish · x_punish + Σ w_KD · Δx_KC + Σ w_MD · Δx_MBON
        # x_punish may be a scalar (all DANs) or ndarray (per-DAN, e.g. PPL-only)
        w_punish = self.config.get('w_punish', 1.0)
        I_DAN = (
            w_punish * np.asarray(x_punish)
            + self.W_KC_DAN    @ self.KCs.r
            + self.W_MBON_DAN  @ self.MBONs.r
            + self.W_DAN_DAN   @ self.DANs.r
        )
        self.DANs.update(I_DAN, dt)

        # -- MBON update (eq 5.3 / 6.3) -----------------------------------
        # I_MBON = Σ w_KM · Δx_KC + Σ w_MM · Δx_MBON
        I_MBON = self.W_KC_MBON @ self.KCs.r + self.W_MBON_MBON @ self.MBONs.r
        self.MBONs.update(I_MBON, dt)

        # -- DA signal per MBON -------------------------------------------
        if getattr(self, '_compartmental', False):
            # Compartmental MV model (Bennett et al. 2021):
            # Split DA into aversive and appetitive components, then apply
            # sign mask so aversive DA depresses approach / potentiates avoid.
            da_avr = self.W_DAN_MBON[:, self._avr_dan_idx] @ \
                     self.DANs.r[self._avr_dan_idx]
            da_app = self.W_DAN_MBON[:, self._app_dan_idx] @ \
                     self.DANs.r[self._app_dan_idx]
            da_per_mbon = self._mbon_sign * (da_avr - da_app)
        else:
            da_per_mbon = self.W_DAN_MBON @ self.DANs.r  # (n_mbon,)

        # -- Plasticity (Gkanias DPR) -------------------------------------
        self.W_KC_MBON = self.plasticity.update(
            da_per_mbon, self.KCs.r, self.W_KC_MBON, dt
        )

    # ------------------------------------------------------------------
    def reset_activity(self):
        """Reset firing rates; keep weights and plasticity state."""
        self.KCs.reset()
        self.MBONs.reset()
        self.DANs.reset()
        self.plasticity.reset()

    def reset_weights(self):
        """Restore weights to initial values."""
        self.W_KC_MBON = self.W_initial.copy()

    def get_weight_change(self):
        """Return ΔW = W_current − W_initial."""
        return self.W_KC_MBON - self.W_initial
