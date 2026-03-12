"""Odor stimulus generation and sensory adaptation.

Implements ORN/KC sensory adaptation (Huang et al., 2024, eq 2.1):

    dw_odor/dt = -x_odor(t)/τ_adapt  · w_odor
               + (1 - x_odor(t))/τ_recover · (A_odor - w_odor)

where:
  w_odor    : current odor weight (modulates KC input strength)
  x_odor(t) : 1 when odor is present, 0 otherwise
  A_odor    : odorant amplitude (analogous to concentration)
  τ_adapt   : adaptation time constant (odor ON)
  τ_recover : recovery time constant (odor OFF)
"""

import numpy as np
import os

# =====================================================================
# OR → Glomerulus mapping (Couto et al. 2005; Fishilevich & Vosshall 2005)
# =====================================================================
# Maps olfactory receptor names (as they appear in the DOOr database)
# to the antennal-lobe glomerulus they innervate.  Each OR is expressed
# in a class of ORNs that project to a single glomerulus.

OR_GLOMERULUS_MAP = {
    # Antenna — basiconic sensilla
    'Or2a':  'DA4m',
    'Or7a':  'DL5',
    'Or9a':  'VM3',
    'Or10a': 'DL1',
    'Or19a': 'DC1',
    'Or22a': 'DM2',
    'Or22b': 'DM5',
    'Or23a': 'DA3',
    'Or33b': 'DA2',   # also expressed with Or47a in ab5B
    'Or35a': 'VC3',
    'Or42a': 'VM7',
    'Or42b': 'DM1',
    'Or43a': 'DA4l',
    'Or43b': 'VM2',
    'Or47a': 'DM3',
    'Or47b': 'VA1lm',
    'Or49b': 'VA5',
    'Or56a': 'DA2',
    'Or59b': 'DM4',
    'Or59c': 'DL4',
    'Or65a': 'DL3',
    'Or67a': 'DM6',
    'Or67b': 'VA3',
    'Or67c': 'VC4',
    'Or67d': 'DA1',
    'Or69a': 'D',
    'Or82a': 'VA6',
    'Or83c': 'DC3',
    'Or85a': 'DM5',
    'Or85b': 'VM5d',
    'Or85f': 'DL4',
    'Or88a': 'VA1d',
    'Or92a': 'VA2',
    'Or98a': 'V',
    # Palp ORs
    'Or46a': 'VA7l',
    'Or71a': 'VC2',
    # Antenna — coeloconic
    'Or35a': 'VC3',
    # Special cases
    'Gr21a.Gr63a': 'V',     # CO₂ receptor → V glomerulus
    'ab4B':        'VM2',    # sensillum-defined, Or43b partner → VM2
    'Ir75a':       'DP1l',
    'Ir75b':       'VL1',
    'Ir75c':       'VL1',
    'Ir76a':       'VM4',
    'Ir84a':       'VL2a',
    'Ir92a':       'VM1',
}


def create_sparse_odor(n_kcs, sparsity=0.1, strength=50.0, noise=0.1):
    """Create sparse odor input pattern.
    
    Parameters
    ----------
    n_kcs : int
        Number of KCs
    sparsity : float
        Fraction of KCs activated (0-1)
    strength : float
        Mean input strength
    noise : float
        Noise level (std relative to strength)
    
    Returns
    -------
    I : ndarray, shape (n_kcs,)
        Input currents
    active_indices : ndarray
        Which KCs are active
    """
    n_active = int(n_kcs * sparsity)
    active_indices = np.random.choice(n_kcs, n_active, replace=False)
    
    I = np.zeros(n_kcs)
    I[active_indices] = strength * (1 + noise * np.random.randn(n_active))
    I = np.maximum(I, 0)  # No negative inputs
    
    return I, active_indices


def temporal_odor_profile(duration, dt, rise_time=50, fall_time=100):
    """Create temporal profile for odor presentation.
    
    Returns envelope that rises, sustains, then falls.
    
    Parameters
    ----------
    duration : float
        Total duration (ms)
    dt : float
        Timestep (ms)
    rise_time : float
        Time to reach max (ms)
    fall_time : float
        Time to decay (ms)
    
    Returns
    -------
    envelope : ndarray
        Temporal profile (0 to 1)
    """
    n_steps = int(duration / dt)
    time = np.arange(n_steps) * dt
    envelope = np.zeros(n_steps)
    
    for i, t in enumerate(time):
        if t < rise_time:
            envelope[i] = t / rise_time
        elif t < duration - fall_time:
            envelope[i] = 1.0
        else:
            time_in_fall = t - (duration - fall_time)
            envelope[i] = 1.0 - (time_in_fall / fall_time)
    
    return np.clip(envelope, 0, 1)


class OdorAdaptation:
    """ORN sensory adaptation for a single odor channel (Huang et al., 2024 eq 2.1).

    Tracks the adaptive weight w_odor that modulates how strongly an odor
    drives the downstream KCs.  When the odor is continuously present the
    weight decays (adaptation); when the odor is removed it recovers toward
    the amplitude A_odor.

    ODE:
        dw/dt = -x_odor/τ_adapt  · w
              + (1 - x_odor)/τ_recover · (A - w)

    Parameters
    ----------
    A_odor : float or ndarray
        Odorant amplitude (concentration proxy).  Shape must match the
        number of odor channels.
    tau_adapt : float
        Adaptation time constant (ms) -- controls decay while odor is ON.
    tau_recover : float
        Recovery time constant (ms) -- controls return to A_odor while OFF.
    """

    def __init__(self, A_odor, tau_adapt=1000.0, tau_recover=3000.0):
        self.A_odor = np.atleast_1d(np.asarray(A_odor, dtype=float))
        self.tau_adapt = tau_adapt
        self.tau_recover = tau_recover
        self.w = self.A_odor.copy()  # starts at full amplitude

    def step(self, x_odor, dt):
        """Update adaptive weight for one timestep.

        Parameters
        ----------
        x_odor : float or ndarray
            1 when the odor is present, 0 otherwise (same shape as A_odor).
        dt : float
            Timestep (ms).

        Returns
        -------
        w : ndarray
            Updated weight(s).
        """
        x = np.asarray(x_odor, dtype=float)
        dw = (
            -(x / self.tau_adapt) * self.w
            + ((1.0 - x) / self.tau_recover) * (self.A_odor - self.w)
        )
        self.w = self.w + dw * dt
        return self.w.copy()

    def reset(self):
        """Reset weight to full amplitude A_odor."""
        self.w = self.A_odor.copy()


# =====================================================================
# DOOr odor response pipeline
# =====================================================================

def load_door_odor(csv_path, threshold=0.01):
    """Load odorant response profile from a DOOr CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the DOOr CSV (columns: ORs, Odor, Response).
    threshold : float
        Minimum absolute response to include.

    Returns
    -------
    or_responses : dict
        ``{or_name: response}`` for responses above *threshold*.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    or_responses = {}
    for _, row in df.iterrows():
        or_name = str(row['ORs']).strip()
        response = float(row['Response'])
        if response > threshold:
            or_responses[or_name] = response
    return or_responses


def create_odor_from_door(csv_path, pn_kc_weights_path, pn_ann_path,
                          n_kc, strength=15.0, sparsity=0.05):
    """Create a biologically grounded KC activation pattern from DOOr data.

    Pipeline:
      1. Load DOOr OR response profile
      2. Map activated ORs → glomeruli via ``OR_GLOMERULUS_MAP``
      3. Load PN annotations → assign each PN to its glomerulus
      4. Build PN activation vector (PN drive = OR response magnitude)
      5. Multiply by cached ``W_PN_KC`` → raw KC drive
      6. Sparsify: keep only top *sparsity* fraction of KCs (APL inhibition)
      7. Normalize so mean active-KC drive = *strength* Hz

    Parameters
    ----------
    csv_path : str
        DOOr odorant CSV path.
    pn_kc_weights_path : str
        Path to ``.npz`` with ``weights`` key (n_kc × n_pn).
    pn_ann_path : str
        Path to PN annotations CSV with ``root_id``, ``glomerulus`` columns.
    n_kc : int
        Number of KCs in the model.
    strength : float
        Mean drive to active KCs (Hz).
    sparsity : float
        Fraction of KCs to keep active.

    Returns
    -------
    I_odor : ndarray, shape (n_kc,)
        KC input currents.
    active_kc_indices : ndarray
        Indices of active KCs.
    """
    import pandas as pd

    # Step 1: DOOr → OR responses
    or_responses = load_door_odor(csv_path, threshold=0.01)
    print(f"  DOOr: {len(or_responses)} active ORs")

    # Step 2: OR → glomerulus
    glom_responses = {}
    for or_name, resp in or_responses.items():
        glom = OR_GLOMERULUS_MAP.get(or_name)
        if glom:
            # Take max if multiple ORs map to same glomerulus
            glom_responses[glom] = max(glom_responses.get(glom, 0), resp)
    print(f"  Activated glomeruli: {len(glom_responses)} "
          f"({', '.join(sorted(glom_responses.keys()))})")

    # Step 3: Load PN annotations and W_PN_KC
    pn_ann = pd.read_csv(pn_ann_path)
    data = np.load(pn_kc_weights_path, allow_pickle=True)
    W_PN_KC = data['weights']  # (n_kc, n_pn)
    pn_ids = data['pn_ids'] if 'pn_ids' in data else None
    n_pn = W_PN_KC.shape[1]

    # Step 4: Build PN activation vector
    pn_activation = np.zeros(n_pn)
    n_matched = 0
    for idx, row in pn_ann.iterrows():
        glom = row.get('glomerulus')
        if pd.notna(glom) and glom in glom_responses and idx < n_pn:
            pn_activation[idx] = glom_responses[glom]
            n_matched += 1

    # If too few PNs matched by glomerulus, use PN cell_type substring matching
    if n_matched < 3:
        print(f"  WARNING: Only {n_matched} PNs matched by glomerulus annotation")
        print(f"  Trying cell_type substring matching ...")
        for idx, row in pn_ann.iterrows():
            if idx >= n_pn:
                break
            ctype = str(row.get('cell_type', ''))
            for glom, resp in glom_responses.items():
                if glom in ctype:
                    pn_activation[idx] = max(pn_activation[idx], resp)
                    n_matched += 1
        print(f"  After substring matching: {n_matched} PNs activated")

    # If still too few, distribute activation proportionally to all PNs
    # with non-zero KC connectivity (fallback)
    if pn_activation.sum() == 0:
        print("  FALLBACK: No PN-glomerulus mapping available, using "
              "proportional activation")
        pn_has_output = (W_PN_KC.sum(axis=0) > 0)
        total_or_drive = sum(or_responses.values())
        pn_activation[pn_has_output] = total_or_drive / pn_has_output.sum()

    print(f"  PN activation: {(pn_activation > 0).sum()} PNs active, "
          f"mean drive = {pn_activation[pn_activation > 0].mean():.4f}")

    # Step 5: Multiply W_PN_KC @ pn_activation → raw KC drive
    # W_PN_KC is (n_kc, n_pn), but may differ from model n_kc if cache is stale
    if W_PN_KC.shape[0] != n_kc:
        print(f"  WARNING: W_PN_KC has {W_PN_KC.shape[0]} KCs but model has "
              f"{n_kc}. Padding/truncating.")
        I_kc_raw = np.zeros(n_kc)
        n = min(W_PN_KC.shape[0], n_kc)
        I_kc_raw[:n] = (W_PN_KC[:n, :] @ pn_activation)
    else:
        I_kc_raw = W_PN_KC @ pn_activation

    # Step 6: Sparsify — keep top sparsity fraction (APL lateral inhibition)
    n_active = max(1, int(n_kc * sparsity))
    threshold_idx = np.argsort(I_kc_raw)[-n_active:]
    active_mask = np.zeros(n_kc, dtype=bool)
    active_mask[threshold_idx] = True
    # Only keep KCs with positive drive
    active_mask &= (I_kc_raw > 0)
    active_kc_indices = np.where(active_mask)[0]

    # Step 7: Normalize active KCs so mean = strength
    I_odor = np.zeros(n_kc)
    if len(active_kc_indices) > 0:
        raw_vals = I_kc_raw[active_kc_indices]
        mean_raw = raw_vals.mean()
        if mean_raw > 0:
            I_odor[active_kc_indices] = raw_vals * (strength / mean_raw)
        else:
            I_odor[active_kc_indices] = strength

    print(f"  Odor pattern: {len(active_kc_indices)} KCs active "
          f"({100 * len(active_kc_indices) / n_kc:.1f}%), "
          f"mean drive = {strength:.1f} Hz")

    return I_odor, active_kc_indices


def create_odor_from_glomeruli(active_glomeruli, pn_kc_weights_path,
                               pn_ann_path, n_kc, strength=15.0,
                               sparsity=0.05):
    """Create KC activation pattern from a list of activated glomeruli.

    Useful for odorants whose glomerular activation is known from imaging
    studies (e.g. Semmelhack & Wang 2009) rather than from DOOr receptor
    data.  All activated glomeruli are treated as equally active.

    Parameters
    ----------
    active_glomeruli : list of str
        Names of activated glomeruli (e.g. ``['DM1', 'DM2', 'DM3']``).
    pn_kc_weights_path : str
        Path to ``.npz`` with ``weights`` key (n_kc x n_pn).
    pn_ann_path : str
        Path to PN annotations CSV with ``root_id``, ``glomerulus`` columns.
    n_kc : int
        Number of KCs in the model.
    strength : float
        Mean drive to active KCs (Hz).
    sparsity : float
        Fraction of KCs to keep active.

    Returns
    -------
    I_odor : ndarray, shape (n_kc,)
        KC input currents.
    active_kc_indices : ndarray
        Indices of active KCs.
    """
    import pandas as pd

    active_set = set(active_glomeruli)

    # Load PN annotations and W_PN_KC
    pn_ann = pd.read_csv(pn_ann_path)
    data = np.load(pn_kc_weights_path, allow_pickle=True)
    W_PN_KC = data['weights']  # (n_kc, n_pn)
    n_pn = W_PN_KC.shape[1]

    # Build PN activation: 1.0 for PNs in activated glomeruli, 0 otherwise
    pn_activation = np.zeros(n_pn)
    n_activated = 0
    for idx, row in pn_ann.iterrows():
        if idx >= n_pn:
            break
        glom = row.get('glomerulus', '')
        if pd.notna(glom) and glom in active_set:
            pn_activation[idx] = 1.0
            n_activated += 1

    print(f"  Activated glomeruli: {len(active_set)} "
          f"({', '.join(sorted(active_set))})")
    print(f"  PN activation: {n_activated}/{n_pn} PNs active")

    # Multiply W_PN_KC @ pn_activation → raw KC drive
    if W_PN_KC.shape[0] != n_kc:
        I_kc_raw = np.zeros(n_kc)
        n = min(W_PN_KC.shape[0], n_kc)
        I_kc_raw[:n] = (W_PN_KC[:n, :] @ pn_activation)
    else:
        I_kc_raw = W_PN_KC @ pn_activation

    # Sparsify — keep top sparsity fraction (APL lateral inhibition)
    n_active = max(1, int(n_kc * sparsity))
    threshold_idx = np.argsort(I_kc_raw)[-n_active:]
    active_mask = np.zeros(n_kc, dtype=bool)
    active_mask[threshold_idx] = True
    active_mask &= (I_kc_raw > 0)
    active_kc_indices = np.where(active_mask)[0]

    # Normalize active KCs so mean = strength
    I_odor = np.zeros(n_kc)
    if len(active_kc_indices) > 0:
        raw_vals = I_kc_raw[active_kc_indices]
        mean_raw = raw_vals.mean()
        if mean_raw > 0:
            I_odor[active_kc_indices] = raw_vals * (strength / mean_raw)
        else:
            I_odor[active_kc_indices] = strength

    print(f"  Odor pattern: {len(active_kc_indices)} KCs active "
          f"({100 * len(active_kc_indices) / n_kc:.1f}%), "
          f"mean drive = {strength:.1f} Hz")

    return I_odor, active_kc_indices


def create_odor_from_door_fallback(csv_path, n_kc, kc_ann_path=None,
                                   strength=15.0, sparsity=0.05):
    """Fallback: create KC pattern from DOOr data without PN→KC connectivity.

    Uses KC subtype annotations (KCab, KCg, KCa'b') to probabilistically
    assign odor responses.  KCab neurons are broadly tuned (respond to many
    odors) while KCg neurons are narrowly tuned.

    The probability of a KC being activated is proportional to:
      - The total number of activated glomeruli (more glomeruli → higher
        probability, especially for broadly-sampling KCab)
      - A subtype-specific scaling: KCab > KCg > KCa'b'

    Parameters
    ----------
    csv_path : str
        DOOr odorant CSV path.
    n_kc : int
        Number of KCs.
    kc_ann_path : str, optional
        Path to KC annotations CSV.  If provided, uses subtype info.
    strength : float
        Mean drive to active KCs (Hz).
    sparsity : float
        Fraction of KCs to keep active.

    Returns
    -------
    I_odor : ndarray, shape (n_kc,)
    active_kc_indices : ndarray
    """
    import pandas as pd

    or_responses = load_door_odor(csv_path, threshold=0.01)

    # Map to glomeruli
    glom_responses = {}
    for or_name, resp in or_responses.items():
        glom = OR_GLOMERULUS_MAP.get(or_name)
        if glom:
            glom_responses[glom] = max(glom_responses.get(glom, 0), resp)

    n_activated_glom = len(glom_responses)
    total_or_drive = sum(glom_responses.values())
    n_total_glom = 51  # approximate total olfactory glomeruli in Drosophila
    fraction_active = n_activated_glom / n_total_glom

    print(f"  Fallback odor: {len(or_responses)} active ORs → "
          f"{n_activated_glom} glomeruli ({fraction_active:.1%} of total)")

    # Assign activation probability per KC based on subtype
    # Each KC samples ~7 random PNs; probability of getting at least 1
    # active PN ≈ 1 - (1 - fraction_active)^n_inputs
    n_inputs_per_kc = {'KCab': 7, 'KCg': 6, 'KCa\'b\'': 9, 'default': 7}

    prob = np.full(n_kc, 1 - (1 - fraction_active) ** 7)  # default

    if kc_ann_path and os.path.exists(kc_ann_path):
        kc_ann = pd.read_csv(kc_ann_path)
        for idx, row in kc_ann.iterrows():
            if idx >= n_kc:
                break
            sub = str(row.get('cell_sub_class', ''))
            if 'KCab' in sub:
                n_in = n_inputs_per_kc['KCab']
            elif 'KCg' in sub:
                n_in = n_inputs_per_kc['KCg']
            elif "KCa" in sub:
                n_in = n_inputs_per_kc['KCa\'b\'']
            else:
                n_in = n_inputs_per_kc['default']
            prob[idx] = 1 - (1 - fraction_active) ** n_in

    # Scale prob by total OR drive (stronger odor → more KCs above threshold)
    prob *= total_or_drive

    # Draw KC activation: deterministic threshold on prob
    n_active = max(1, int(n_kc * sparsity))
    top_indices = np.argsort(prob)[-n_active:]
    active_mask = np.zeros(n_kc, dtype=bool)
    active_mask[top_indices] = True
    active_kc_indices = np.where(active_mask)[0]

    # Normalize
    I_odor = np.zeros(n_kc)
    raw_vals = prob[active_kc_indices]
    mean_raw = raw_vals.mean()
    if mean_raw > 0:
        I_odor[active_kc_indices] = raw_vals * (strength / mean_raw)
    else:
        I_odor[active_kc_indices] = strength

    print(f"  Odor pattern: {len(active_kc_indices)} KCs active, "
          f"mean drive = {strength:.1f} Hz")

    return I_odor, active_kc_indices
