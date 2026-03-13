"""V4: Forward vs Backward Conditioning.

Emulates the paradigm from Gkanias et al. (2022) eLife 75611 §4:

  Forward  conditioning: CS (odor) on for T_cs ms; US (PPL shock) overlaps
                         the final T_us ms of CS.  US arrives WHILE KCs are
                         still active → D_△ (fast) rises before D_▽ (slow)
                         → δ = D_▽ − D_△ < 0 during KC activity
                         → net KC→MBON depression.

  Backward conditioning: US delivered first (T_us ms); CS begins T_isi ms
                         after US ends.  When KCs activate, D_△ (τ_short=200ms)
                         has largely decayed but D_▽ (τ_long=2000ms) remains
                         elevated → δ = D_▽ − D_△ > 0 → net KC→MBON potentiation.

Trial structure (following Gkanias et al. single-phase example):
  Phase 1  blank        : 0 – t_blank
  Phase 2  CS only      : t_blank – t_us_on       (KCs ramp to steady state)
  Phase 3  CS + US      : t_us_on – t_cs_off      (DA arrives while KCs active)
  Phase 4  CS only      : t_cs_off – t_cs_off      [N/A here: US ends with CS]
  Phase 5  blank        : t_cs_off – trial_end

Outputs:
  - MBON firing rates over time for both protocols
  - KC->MBON weight change heatmaps (ΔW = W_final - W_initial)
  - Mean weight change per MBON (bar chart)
  - Approach vs avoid MBON group firing rates (naive/forward/backward)
  - Preference index summary bar chart
  - Training dynamics (D_up, D_down, delta, cumulative ΔW)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from mbmodel.models import MushroomBodyNetwork
from mbmodel.stimuli import create_sparse_odor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_ppl_indices(dan_ann_path, ids_path):
    """Return row indices (into DAN weight matrices) of PPL neurons."""
    dan_ann = pd.read_csv(dan_ann_path)
    ids     = np.load(ids_path, allow_pickle=True)
    dan_ids = ids['dan_ids']

    ppl_root_ids = set(
        dan_ann.loc[dan_ann['cell_type'].str.startswith('PPL', na=False), 'root_id']
    )
    id2idx = {rid: i for i, rid in enumerate(dan_ids)}
    ppl_indices = [id2idx[rid] for rid in ppl_root_ids if rid in id2idx]
    return sorted(ppl_indices)


def load_mbon_labels(mbon_ann_path, ids_path):
    """Return list of MBON cell_type labels ordered by weight matrix row."""
    mbon_ann = pd.read_csv(mbon_ann_path)
    ids      = np.load(ids_path, allow_pickle=True)
    mbon_ids = ids['mbon_ids']
    id2type  = mbon_ann.set_index('root_id')['cell_type'].to_dict()
    return [id2type.get(rid, f'MBON{i:02d}') for i, rid in enumerate(mbon_ids)]


def classify_mbons():
    """Classify MBONs into approach, avoid, and other groups.

    Classification based on Aso et al. (2014) and Aso & Rubin (2016):
      - Approach (glutamatergic): PPL1-targeted compartments; depression of
        these drives learned aversion.
      - Avoid (GABAergic): depression of these drives learned approach.
      - Other (cholinergic): mixed valence roles, excluded from preference index.

    Returns dict with keys 'approach', 'avoid', 'other', each a list of
    row indices into the MBON weight matrices.
    """
    approach = [0, 3, 4, 18, 20, 28, 39, 41, 44, 46, 47]  # 11 MBONs
    avoid    = [1, 2, 5, 7, 10, 14, 21, 23, 24, 26, 42, 45]  # 12 MBONs
    all_idx  = set(range(48))
    other    = sorted(all_idx - set(approach) - set(avoid))  # 25 MBONs
    return {'approach': approach, 'avoid': avoid, 'other': other}


def build_mbon_sign_mask(n_mbon, mbon_groups):
    """Build sign mask from manual classification: +1 approach, -1 avoid, 0 other.

    Reference: Aso et al. (2014b) / Aso & Rubin (2016).
    """
    mask = np.zeros(n_mbon)
    mask[mbon_groups['approach']] = +1.0
    mask[mbon_groups['avoid']]   = -1.0
    return mask


def derive_mbon_sign_from_connectivity(W_DAN_MBON, ppl_indices, pam_indices,
                                        mode='hard'):
    """Derive MBON valence sign mask from DAN→MBON connectivity.

    For each MBON j, computes the fraction of dopaminergic input from PPL
    (aversive) vs PAM (appetitive) DANs.  MBONs receiving primarily PPL input
    are classified as approach (PPL1 innervates approach compartments per
    Aso et al. 2014b), and those receiving primarily PAM input as avoid.

    Parameters
    ----------
    W_DAN_MBON : ndarray, shape (n_mbon, n_dan)
        DAN→MBON connectivity matrix (row = postsynaptic MBON).
    ppl_indices : list of int
        Column indices of PPL (aversive) DANs.
    pam_indices : list of int
        Column indices of PAM (appetitive) DANs.
    mode : str
        'hard'       — binary sign: +1 if ppl_fraction > 0.5, −1 if < 0.5, 0 if equal/no input
        'continuous' — smooth sign: 2 × ppl_fraction − 1 (ranges from −1 to +1)

    Returns
    -------
    sign_mask : ndarray, shape (n_mbon,)
        +1 for approach (PPL-dominated), −1 for avoid (PAM-dominated).
    ppl_fractions : ndarray, shape (n_mbon,)
        PPL fraction for each MBON (useful for analysis).
    """
    n_mbon = W_DAN_MBON.shape[0]
    ppl_input = W_DAN_MBON[:, ppl_indices].sum(axis=1)  # (n_mbon,)
    pam_input = W_DAN_MBON[:, pam_indices].sum(axis=1)  # (n_mbon,)
    total = ppl_input + pam_input

    ppl_fractions = np.where(total > 0, ppl_input / total, 0.0)

    if mode == 'hard':
        sign_mask = np.zeros(n_mbon)
        sign_mask[ppl_fractions > 0.5] = +1.0
        sign_mask[ppl_fractions < 0.5] = -1.0
        # MBONs with exactly 0.5 or no input remain 0
    elif mode == 'continuous':
        sign_mask = np.where(total > 0, 2.0 * ppl_fractions - 1.0, 0.0)
    else:
        raise ValueError(f"Unknown mode: {mode!r}; use 'hard' or 'continuous'")

    return sign_mask, ppl_fractions


def print_mbon_sign_comparison(mbon_labels, manual_sign, derived_sign,
                                ppl_fractions, mbon_groups):
    """Print comparison table: connectivity-derived vs manual MBON classification."""
    print("\n" + "=" * 80)
    print("MBON VALENCE CLASSIFICATION: Connectivity-Derived vs Manual (Aso et al. 2014b)")
    print("=" * 80)
    print(f"{'MBON':<25s} {'Manual':>8s} {'Derived':>8s} {'PPL%':>6s} {'Match':>6s}")
    print("-" * 80)

    n_match = 0
    n_classified = 0
    for j, lbl in enumerate(mbon_labels):
        m_sign = manual_sign[j]
        d_sign = derived_sign[j]
        ppl_pct = ppl_fractions[j] * 100

        m_str = {1.0: '+1 app', -1.0: '-1 avd', 0.0: ' 0 oth'}[m_sign]
        d_str = {1.0: '+1 app', -1.0: '-1 avd', 0.0: ' 0 oth'}.get(
            d_sign, f'{d_sign:+.2f}')

        if m_sign != 0:
            n_classified += 1
            match = (m_sign > 0 and d_sign > 0) or (m_sign < 0 and d_sign < 0)
            if match:
                n_match += 1
            match_str = 'YES' if match else 'NO'
        else:
            match_str = '-'

        print(f"{lbl:<25s} {m_str:>8s} {d_str:>8s} {ppl_pct:>5.1f}% {match_str:>6s}")

    print("-" * 80)
    print(f"Agreement on classified MBONs: {n_match}/{n_classified} "
          f"({100*n_match/n_classified:.0f}%)" if n_classified > 0 else "No classified MBONs")
    print(f"Derived:  {int((derived_sign > 0).sum())} approach, "
          f"{int((derived_sign < 0).sum())} avoid, "
          f"{int((derived_sign == 0).sum())} neutral")
    print("=" * 80)


def load_pam_indices(dan_ann_path, ids_path):
    """Return row indices (into DAN weight matrices) of PAM neurons."""
    dan_ann = pd.read_csv(dan_ann_path)
    ids     = np.load(ids_path, allow_pickle=True)
    dan_ids = ids['dan_ids']

    pam_root_ids = set(
        dan_ann.loc[dan_ann['cell_type'].str.startswith('PAM', na=False), 'root_id']
    )
    id2idx = {rid: i for i, rid in enumerate(dan_ids)}
    pam_indices = [id2idx[rid] for rid in pam_root_ids if rid in id2idx]
    return sorted(pam_indices)


def build_ppl_stimulus(n_dan, ppl_indices, strength=80.0):
    """Build a per-DAN input vector that activates only PPL neurons."""
    x = np.zeros(n_dan)
    x[ppl_indices] = strength
    return x


def build_pam_stimulus(n_dan, pam_indices, strength=80.0):
    """Build a per-DAN input vector that activates only PAM neurons."""
    x = np.zeros(n_dan)
    x[pam_indices] = strength
    return x


def train_trial(net, odor_pattern, ppl_stimulus, dt,
                trial_duration, cs_on, cs_off, us_on, us_off):
    """Run one training trial; only updates weights, no trace storage."""
    n_steps      = int(trial_duration / dt)
    zeros_kc     = np.zeros(net.KCs.n)
    zeros_dan    = np.zeros(net.DANs.n)

    net.reset_activity()
    for step in range(n_steps):
        t = step * dt
        I_odor   = odor_pattern if cs_on <= t < cs_off else zeros_kc
        x_punish = ppl_stimulus if us_on <= t < us_off else zeros_dan
        net.step(I_odor, x_punish)


def record_training_trial(net, odor_pattern, ppl_stimulus, dt,
                          trial_duration, cs_on, cs_off, us_on, us_off,
                          odor_kcs, record_every=10):
    """Run one training trial and record internal dynamics for visualization.

    Records per-step: mean KC rate (odor-active KCs), mean DA-to-MBON,
    D_up mean, D_down mean, delta mean, and cumulative mean ΔW.
    """
    n_steps   = int(trial_duration / dt)
    zeros_kc  = np.zeros(net.KCs.n)
    zeros_dan = np.zeros(net.DANs.n)
    n_rec     = n_steps // record_every

    t_vec       = np.zeros(n_rec)
    mean_kc_r   = np.zeros(n_rec)
    mean_da     = np.zeros(n_rec)
    mean_d_up   = np.zeros(n_rec)
    mean_d_down = np.zeros(n_rec)
    mean_delta  = np.zeros(n_rec)
    cum_mean_dw = np.zeros(n_rec)

    W_before = net.W_KC_MBON.copy()

    net.reset_activity()
    rec = 0
    for step in range(n_steps):
        t = step * dt
        I_odor   = odor_pattern if cs_on <= t < cs_off else zeros_kc
        x_punish = ppl_stimulus if us_on <= t < us_off else zeros_dan
        net.step(I_odor, x_punish)

        if step % record_every == 0 and rec < n_rec:
            t_vec[rec]       = t
            mean_kc_r[rec]   = net.KCs.r[odor_kcs].mean()
            mean_da[rec]     = (net.W_DAN_MBON @ net.DANs.r).mean()
            mean_d_up[rec]   = net.plasticity.D_up.mean()
            mean_d_down[rec] = net.plasticity.D_down.mean()
            mean_delta[rec]  = net.plasticity.delta.mean()
            cum_mean_dw[rec] = (net.W_KC_MBON - W_before).mean()
            rec += 1

    return {
        't': t_vec, 'mean_kc_r': mean_kc_r, 'mean_da': mean_da,
        'mean_d_up': mean_d_up, 'mean_d_down': mean_d_down,
        'mean_delta': mean_delta, 'cum_mean_dw': cum_mean_dw,
    }


def test_trial(net, odor_pattern, dt, trial_duration,
               cs_on, cs_off, record_every=10):
    """Run one test trial (odor only); record MBON and DAN at coarse rate.

    record_every : int
        Record every Nth step (default 10 → 1 ms resolution at dt=0.1 ms).

    Returns dict: t (ms), mbon_r (n_rec × n_mbon), dan_r (n_rec × n_dan)
    """
    n_steps   = int(trial_duration / dt)
    zeros_kc  = np.zeros(net.KCs.n)
    zeros_dan = np.zeros(net.DANs.n)
    n_rec     = n_steps // record_every

    mbon_r = np.zeros((n_rec, net.MBONs.n))
    dan_r  = np.zeros((n_rec, net.DANs.n))
    t_vec  = np.zeros(n_rec)

    net.reset_activity()
    rec = 0
    for step in range(n_steps):
        t        = step * dt
        I_odor   = odor_pattern if cs_on <= t < cs_off else zeros_kc
        net.step(I_odor, zeros_dan)
        if step % record_every == 0 and rec < n_rec:
            mbon_r[rec] = net.MBONs.r
            dan_r[rec]  = net.DANs.r
            t_vec[rec]  = t
            rec += 1

    return {'t': t_vec, 'mbon_r': mbon_r, 'dan_r': dan_r}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_v4():
    print("=" * 65)
    print("V4: Forward vs Backward Conditioning (PPL DANs)")
    print("=" * 65)

    # -- Config --------------------------------------------------------------
    with open('experiments/configs/config.yaml') as f:
        config = yaml.safe_load(f)

    # -- Parameter choices (documented) --------------------------------------
    # eta = 5e-6: produces partial (~5-50%) weight changes over 5 trials,
    #   avoiding saturation while still showing clear directional effects.
    # odor_strength = 15.0: KC rates ~15 Hz, within the biological range
    #   (~5-30 Hz for sparse odor-responsive KCs in Drosophila).
    # W_KC_DAN = 0: gated DA release — DA at KC-MBON synapses reflects
    #   reinforcement timing, not odor-driven DAN activity (Handler et al. 2019).
    # W_KC_MBON rescaling: connectome normalization yields max weight ~0.0004;
    #   the DPR operates around w_rest=1.0, so we rescale max weight to w_rest
    #   to place synapses in the regime where the DPR produces meaningful updates.

    # Plasticity timescales (tau_short=200ms, tau_long=2000ms) are well-resolved
    # at 1ms. Override to 1ms for speed; revert to config dt for fine-grain work.
    config['dt'] = 1.0

    config['learning_rate'] = 5e-6

    # -- Network -------------------------------------------------------------
    net_fwd  = MushroomBodyNetwork(config)
    net_bwd  = MushroomBodyNetwork(config)
    net_naive = MushroomBodyNetwork(config)

    # Suppress KC→DAN feedback so the DA signal to the plasticity rule
    # reflects shock timing, not odor-driven DAN activity.  In biology,
    # DA release at KC-MBON synapses is gated by reinforcement (Handler
    # et al., 2019); KC activity alone does not trigger it.
    net_fwd.W_KC_DAN[:] = 0.0
    net_bwd.W_KC_DAN[:] = 0.0
    net_naive.W_KC_DAN[:] = 0.0

    # Rescale KC→MBON weights: after max-row-sum normalization the weights
    # are ~0.0004, far too small for MBONs to respond and for the DPR
    # (w_rest=1.0) to operate meaningfully.  Scale so max weight ≈ w_rest.
    w_rest = config.get('w_rest', 1.0)
    for net in (net_fwd, net_bwd, net_naive):
        wmax = net.W_KC_MBON.max()
        if wmax > 0:
            net.W_KC_MBON *= (w_rest / wmax)
        net.W_initial = net.W_KC_MBON.copy()

    n_kc   = net_fwd.KCs.n
    n_mbon = net_fwd.MBONs.n
    n_dan  = net_fwd.DANs.n
    dt     = config['dt']

    print(f"\nNetwork: {n_kc} KCs, {n_mbon} MBONs, {n_dan} DANs")

    # -- PPL identification --------------------------------------------------
    ppl_indices = load_ppl_indices(
        'data/connectomes/processed/mb_circuit_right_dan_annotations.csv',
        'data/connectomes/processed/mb_circuit_right_ids.npz',
    )
    print(f"PPL neurons: {len(ppl_indices)} (indices {ppl_indices})")

    mbon_labels = load_mbon_labels(
        'data/connectomes/processed/mb_circuit_right_mbon_annotations.csv',
        'data/connectomes/processed/mb_circuit_right_ids.npz',
    )

    # -- MBON classification -------------------------------------------------
    mbon_groups = classify_mbons()
    print(f"MBON groups: {len(mbon_groups['approach'])} approach, "
          f"{len(mbon_groups['avoid'])} avoid, "
          f"{len(mbon_groups['other'])} other")

    # -- Odor pattern (fixed random sparse KCs) ------------------------------
    np.random.seed(7)
    odor_pattern, odor_kcs = create_sparse_odor(n_kc, sparsity=0.05, strength=15.0)
    print(f"Odor: {len(odor_kcs)} KCs active ({100*len(odor_kcs)/n_kc:.1f}%)")

    ppl_stim = build_ppl_stimulus(n_dan, ppl_indices, strength=80.0)

    # -- Trial parameters (Gkanias et al. 2022 paradigm) ---------------------
    # Forward:  CS (2000 ms odor) with US overlapping the final 500 ms.
    #           When DA arrives at t=us_on, KCs are already active → D_△ rises
    #           fast, D_▽ rises slow → δ = D_▽ − D_△ < 0, k>0
    #           → depression of CS-responsive KC→MBON synapses.
    # Backward: US (500 ms shock) first; CS begins 500 ms after US ends.
    #           When KCs activate, D_△ has decayed (τ=200ms, ~8% remaining at
    #           500ms post-US) but D_▽ still elevated (τ=2000ms, ~78% remaining)
    #           → δ = D_▽ − D_△ > 0 → potentiation of CS-responsive KC→MBON synapses.
    n_trials       = 5
    trial_duration = 4000   # ms  (total per trial)

    # Forward conditioning
    fwd_cs_on  = 500         # odor onset
    fwd_cs_off = 2500        # odor offset  (2000 ms CS)
    fwd_us_on  = 2000        # shock onset  -- overlaps LAST 500 ms of CS
    fwd_us_off = 2500        # shock offset (= CS offset)

    # Backward conditioning
    bwd_us_on  = 500         # shock onset
    bwd_us_off = 1000        # shock offset (500 ms shock)
    bwd_cs_on  = 1500        # odor onset   -- 500 ms after shock ends
    bwd_cs_off = 3500        # odor offset  (2000 ms CS)

    print(f"\nForward:  CS {fwd_cs_on}–{fwd_cs_off} ms | "
          f"US {fwd_us_on}–{fwd_us_off} ms (US overlaps end of CS)")
    print(f"Backward: US {bwd_us_on}–{bwd_us_off} ms | "
          f"CS {bwd_cs_on}–{bwd_cs_off} ms (CS starts 500 ms after US ends)")
    print(f"Trials: {n_trials}\n")

    # -- Record first training trial for dynamics visualization ---------------
    print("  Recording trial 1 dynamics ...", end=' ', flush=True)
    rec_fwd = record_training_trial(
        net_fwd, odor_pattern, ppl_stim, dt, trial_duration,
        fwd_cs_on, fwd_cs_off, fwd_us_on, fwd_us_off, odor_kcs)
    rec_bwd = record_training_trial(
        net_bwd, odor_pattern, ppl_stim, dt, trial_duration,
        bwd_cs_on, bwd_cs_off, bwd_us_on, bwd_us_off, odor_kcs)
    print("done")

    # -- Remaining training trials -------------------------------------------
    for trial in range(1, n_trials):
        print(f"  Trial {trial + 1}/{n_trials} ...", end=' ', flush=True)

        train_trial(net_fwd, odor_pattern, ppl_stim, dt, trial_duration,
                    fwd_cs_on, fwd_cs_off, fwd_us_on, fwd_us_off)

        train_trial(net_bwd, odor_pattern, ppl_stim, dt, trial_duration,
                    bwd_cs_on, bwd_cs_off, bwd_us_on, bwd_us_off)

        print("done")

    # -- Final test trial (odor only, no shock) -------------------------------
    # Present CS alone using forward timing; no PPL shock.
    # Test trial timing: CS 500-2500 ms (same for all 3 conditions).
    test_cs_on  = 500
    test_cs_off = 2500
    print("\nRunning test trials (odor only) ...")
    test_fwd   = test_trial(net_fwd, odor_pattern, dt, trial_duration,
                            test_cs_on, test_cs_off, record_every=1)
    test_bwd   = test_trial(net_bwd, odor_pattern, dt, trial_duration,
                            test_cs_on, test_cs_off, record_every=1)
    test_naive = test_trial(net_naive, odor_pattern, dt, trial_duration,
                            test_cs_on, test_cs_off, record_every=1)

    dW_fwd = net_fwd.get_weight_change()   # (n_mbon, n_kc)
    dW_bwd = net_bwd.get_weight_change()

    # -- Weight change analysis ----------------------------------------------
    print(f"\nForward  ΔW: mean={dW_fwd.mean():.4f}  "
          f"min={dW_fwd.min():.4f}  max={dW_fwd.max():.4f}")
    print(f"Backward ΔW: mean={dW_bwd.mean():.4f}  "
          f"min={dW_bwd.min():.4f}  max={dW_bwd.max():.4f}")

    # ========================================================================
    # FIGURE 1 — MBON responses during test trial
    # ========================================================================
    t_vec = test_fwd['t']

    fig1, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, sharey=True)

    # Compute shared y-limit across both conditions
    y_max = max(test_fwd['mbon_r'].max(), test_bwd['mbon_r'].max()) * 1.1

    for ax, traces, label, color in zip(
        axes,
        [test_fwd['mbon_r'], test_bwd['mbon_r']],
        ['Forward conditioning (CS then CS+US)', 'Backward conditioning (US then CS)'],
        ['steelblue', 'firebrick'],
    ):
        # Plot each MBON as a faint line, mean as bold
        mean_r = traces.mean(axis=1)
        for j in range(n_mbon):
            ax.plot(t_vec, traces[:, j], lw=0.4, alpha=0.25, color=color)
        ax.plot(t_vec, mean_r, lw=2, color=color, label='Mean MBON')
        ax.axvspan(test_cs_on, test_cs_off, alpha=0.12, color='dodgerblue', label='Odor CS (test)')
        ax.set_ylabel('Firing rate (Hz)', fontsize=10)
        ax.set_ylim(-1, max(y_max, 5))
        ax.set_title(label, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel('Time (ms)', fontsize=10)
    fig1.suptitle('MBON Responses After Conditioning — Test Trial (Odor Only, No Shock)\n'
                  'Forward: CS+US overlap (Gkanias et al. 2022 paradigm) | '
                  'Backward: US → CS',
                  fontweight='bold', fontsize=11)
    fig1.tight_layout()
    fig1.savefig('results/v4c_mbon_responses.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: results/v4c_mbon_responses.png")

    # ========================================================================
    # FIGURE 2 — KC→MBON weight change heatmaps
    # ========================================================================
    # Only show the odor-active KCs to keep the plot legible
    odor_kcs_sorted = np.sort(odor_kcs)

    vmax = max(np.abs(dW_fwd[:, odor_kcs_sorted]).max(),
               np.abs(dW_bwd[:, odor_kcs_sorted]).max()) + 1e-9

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 9))

    for ax, dW, title in zip(
        axes2,
        [dW_fwd, dW_bwd],
        ['Forward Conditioning', 'Backward Conditioning'],
    ):
        im = ax.imshow(
            dW[:, odor_kcs_sorted],
            aspect='auto', cmap='RdBu_r',
            vmin=-vmax, vmax=vmax,
            interpolation='nearest',
        )
        plt.colorbar(im, ax=ax, label='ΔW (weight change)')
        ax.set_xlabel(f'Odor-active KCs ({len(odor_kcs_sorted)} of {n_kc})', fontsize=9)
        ax.set_ylabel('MBON', fontsize=9)
        ax.set_title(f'KC→MBON ΔW — {title}', fontweight='bold')
        ax.set_yticks(range(n_mbon))
        ax.set_yticklabels(mbon_labels, fontsize=6)

    fig2.suptitle(f'KC→MBON Synaptic Weight Changes After {n_trials} Trials',
                  fontweight='bold', fontsize=12)
    fig2.tight_layout()
    fig2.savefig('results/v4c_weight_heatmaps.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/v4c_weight_heatmaps.png")

    # ========================================================================
    # FIGURE 3 — Mean ΔW per MBON (bar chart)
    # ========================================================================
    mean_dW_fwd = dW_fwd[:, odor_kcs_sorted].mean(axis=1)
    mean_dW_bwd = dW_bwd[:, odor_kcs_sorted].mean(axis=1)

    x = np.arange(n_mbon)
    width = 0.4

    fig3, ax3 = plt.subplots(figsize=(16, 5))
    ax3.bar(x - width / 2, mean_dW_fwd, width, label='Forward',
            color='steelblue', alpha=0.8)
    ax3.bar(x + width / 2, mean_dW_bwd, width, label='Backward',
            color='firebrick', alpha=0.8)
    ax3.axhline(0, color='k', lw=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(mbon_labels, rotation=45, ha='right', fontsize=7)
    ax3.set_ylabel('Mean ΔW (over odor-active KCs)', fontsize=10)
    ax3.set_title('Per-MBON Mean Weight Change: Forward vs Backward Conditioning',
                  fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    fig3.tight_layout()
    fig3.savefig('results/v4c_mean_dw_per_mbon.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/v4c_mean_dw_per_mbon.png")

    # ========================================================================
    # FIGURE 4 — Approach vs Avoid MBON firing rates (test trial)
    # ========================================================================
    approach_idx = mbon_groups['approach']
    avoid_idx    = mbon_groups['avoid']

    fig4, (ax_app, ax_avd) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for ax, idx, group_name in [(ax_app, approach_idx, 'Approach MBONs (glutamatergic)'),
                                 (ax_avd, avoid_idx, 'Avoid MBONs (GABAergic)')]:
        mean_naive = test_naive['mbon_r'][:, idx].mean(axis=1)
        mean_fwd_r = test_fwd['mbon_r'][:, idx].mean(axis=1)
        mean_bwd_r = test_bwd['mbon_r'][:, idx].mean(axis=1)

        ax.axvspan(test_cs_on, test_cs_off, alpha=0.12, color='dodgerblue', label='Odor CS')
        ax.plot(t_vec, mean_naive, lw=2, color='gray', label='Naive')
        ax.plot(t_vec, mean_fwd_r, lw=2, color='steelblue', label='Forward')
        ax.plot(t_vec, mean_bwd_r, lw=2, color='firebrick', label='Backward')
        ax.set_ylabel('Mean firing rate (Hz)', fontsize=10)
        ax.set_title(group_name, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.25)

    ax_avd.set_xlabel('Time (ms)', fontsize=10)
    fig4.suptitle('Approach vs Avoid MBON Group Responses — Test Trial\n'
                  '(Naive = no training, Forward = CS+US, Backward = US→CS)',
                  fontweight='bold', fontsize=11)
    fig4.tight_layout()
    fig4.savefig('results/v4c_approach_avoid_rates.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/v4c_approach_avoid_rates.png")

    # ========================================================================
    # FIGURE 5 — Preference summary bar chart
    # ========================================================================
    # Mean firing rate during CS-on window for each group × condition
    cs_mask = (t_vec >= test_cs_on) & (t_vec < test_cs_off)

    conditions = {
        'Naive':    test_naive['mbon_r'],
        'Forward':  test_fwd['mbon_r'],
        'Backward': test_bwd['mbon_r'],
    }
    cond_colors = {'Naive': 'gray', 'Forward': 'steelblue', 'Backward': 'firebrick'}

    bar_data = {}
    for cond_name, mbon_r in conditions.items():
        bar_data[(cond_name, 'Approach')] = mbon_r[cs_mask][:, approach_idx].mean()
        bar_data[(cond_name, 'Avoid')]    = mbon_r[cs_mask][:, avoid_idx].mean()

    fig5, ax5 = plt.subplots(figsize=(8, 5))
    bar_x = np.arange(3)  # 3 conditions
    width5 = 0.35
    cond_names = ['Naive', 'Forward', 'Backward']

    app_vals = [bar_data[(c, 'Approach')] for c in cond_names]
    avd_vals = [bar_data[(c, 'Avoid')]    for c in cond_names]
    colors   = [cond_colors[c] for c in cond_names]

    bars_app = ax5.bar(bar_x - width5/2, app_vals, width5, label='Approach',
                       color=colors, alpha=0.85, edgecolor='k', linewidth=0.5)
    bars_avd = ax5.bar(bar_x + width5/2, avd_vals, width5, label='Avoid',
                       color=colors, alpha=0.45, edgecolor='k', linewidth=0.5,
                       hatch='///')

    ax5.set_xticks(bar_x)
    ax5.set_xticklabels(cond_names, fontsize=11)
    ax5.set_ylabel('Mean firing rate during CS (Hz)', fontsize=10)
    ax5.set_title('MBON Group Responses: Approach vs Avoid\n'
                  '(solid = approach, hatched = avoid)', fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(axis='y', alpha=0.3)
    fig5.tight_layout()
    fig5.savefig('results/v4c_preference_summary.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/v4c_preference_summary.png")

    # ========================================================================
    # FIGURE 6 — Training dynamics (trial 1)
    # ========================================================================
    fig6, axes6 = plt.subplots(5, 2, figsize=(16, 14), sharex='col')

    row_labels = ['Stimulus timing', 'Mean KC rate (odor KCs)',
                  'DA input to MBONs', 'D_up (fast) & D_down (slow)',
                  'δ (D_down − D_up) & cumulative ΔW']

    for col, (rec, cond_label, cs_on, cs_off, us_on, us_off) in enumerate([
        (rec_fwd, 'Forward', fwd_cs_on, fwd_cs_off, fwd_us_on, fwd_us_off),
        (rec_bwd, 'Backward', bwd_cs_on, bwd_cs_off, bwd_us_on, bwd_us_off),
    ]):
        t = rec['t']

        # Row 0: Stimulus timing
        ax = axes6[0, col]
        ax.axvspan(cs_on, cs_off, alpha=0.3, color='dodgerblue', label='CS (odor)')
        ax.axvspan(us_on, us_off, alpha=0.3, color='red', label='US (shock)')
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title(f'{cond_label} Conditioning', fontweight='bold', fontsize=12)
        ax.legend(fontsize=8, loc='upper right')

        # Row 1: Mean KC rate
        ax = axes6[1, col]
        ax.plot(t, rec['mean_kc_r'], color='green', lw=1.5)
        ax.set_ylabel('KC rate (Hz)', fontsize=9)
        ax.grid(alpha=0.25)

        # Row 2: DA input
        ax = axes6[2, col]
        ax.plot(t, rec['mean_da'], color='purple', lw=1.5)
        ax.set_ylabel('DA to MBONs', fontsize=9)
        ax.grid(alpha=0.25)

        # Row 3: D_up and D_down
        ax = axes6[3, col]
        ax.plot(t, rec['mean_d_up'], color='orange', lw=1.5, label='D_up (τ=200ms)')
        ax.plot(t, rec['mean_d_down'], color='darkred', lw=1.5, label='D_down (τ=2000ms)')
        ax.set_ylabel('DA filter', fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.25)

        # Row 4: delta and cumulative ΔW
        ax = axes6[4, col]
        ax.plot(t, rec['mean_delta'], color='navy', lw=1.5, label='δ = D_down − D_up')
        ax_tw = ax.twinx()
        ax_tw.plot(t, rec['cum_mean_dw'], color='crimson', lw=1.5, ls='--',
                   label='Cum. mean ΔW')
        ax.set_ylabel('δ', fontsize=9, color='navy')
        ax_tw.set_ylabel('Cum. ΔW', fontsize=9, color='crimson')
        ax.set_xlabel('Time (ms)', fontsize=10)
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_tw.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
        ax.grid(alpha=0.25)

    # Add row labels on the left side
    for i, label in enumerate(row_labels):
        axes6[i, 0].annotate(label, xy=(-0.25, 0.5), xycoords='axes fraction',
                             fontsize=9, fontweight='bold', ha='right', va='center',
                             rotation=90)

    fig6.suptitle('Training Dynamics — Trial 1\n'
                  'Key mechanism: δ sign during KC-active window determines '
                  'depression vs potentiation',
                  fontweight='bold', fontsize=12)
    fig6.tight_layout()
    fig6.savefig('results/v4c_training_dynamics.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/v4c_training_dynamics.png")

    plt.show()
    print("\n" + "=" * 65)
    print("V4 Complete.")
    print("=" * 65)


if __name__ == '__main__':
    run_v4()
