"""V4e: Compartmental plasticity — valence-opponent DA signaling.

Implements the mixed-valence (MV) model from Bennett et al. (2021):
  - PPL (aversive DA) depresses approach MBONs, potentiates avoid MBONs
  - PAM (appetitive DA) depresses avoid MBONs, potentiates approach MBONs

The effective DA signal per MBON is:
    da_eff_j = sign_j · (da_aversive_j − da_appetitive_j)
where sign_j = +1 (approach), −1 (avoid), 0 (other/unclassified).

Compared to v4c (uniform DA), this should produce:
  - Forward conditioning (aversive): approach depressed, avoid potentiated
  - Backward conditioning: weaker/opposite effects due to temporal asymmetry
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import yaml

from mbmodel.models import MushroomBodyNetwork
from mbmodel.stimuli import create_sparse_odor

from v4_conditioning import (
    load_ppl_indices, load_mbon_labels, classify_mbons,
    train_trial, record_training_trial, test_trial,
)


def load_pam_indices(dan_ann_path, ids_path):
    """Return row indices (into DAN weight matrices) of PAM neurons."""
    import pandas as pd
    dan_ann = pd.read_csv(dan_ann_path)
    ids     = np.load(ids_path, allow_pickle=True)
    dan_ids = ids['dan_ids']

    pam_root_ids = set(
        dan_ann.loc[dan_ann['cell_type'].str.startswith('PAM', na=False), 'root_id']
    )
    id2idx = {rid: i for i, rid in enumerate(dan_ids)}
    return sorted([id2idx[rid] for rid in pam_root_ids if rid in id2idx])


def build_mbon_sign_mask(n_mbon, mbon_groups):
    """Build sign mask: +1 approach, -1 avoid, 0 other."""
    mask = np.zeros(n_mbon)
    mask[mbon_groups['approach']] = +1.0
    mask[mbon_groups['avoid']]   = -1.0
    return mask


def run_v4e():
    print("=" * 65)
    print("V4e: Compartmental Plasticity (MV Model, Bennett et al. 2021)")
    print("=" * 65)

    # -- Config --------------------------------------------------------------
    with open('experiments/configs/config.yaml') as f:
        config = yaml.safe_load(f)
    config['dt'] = 1.0
    config['learning_rate'] = 5e-6
    # Use KC-gating-only weight update (Bennett et al. 2021 MV form):
    # dW = η · δ · k  instead of  dW = η · δ · (k + W - w_rest)
    # Stable for bidirectional plasticity (no W-dependent runaway).
    config['kc_gating_only'] = True

    # -- Networks ------------------------------------------------------------
    net_fwd   = MushroomBodyNetwork(config)
    net_bwd   = MushroomBodyNetwork(config)
    net_naive = MushroomBodyNetwork(config)

    for net in (net_fwd, net_bwd, net_naive):
        net.W_KC_DAN[:] = 0.0
        # Zero recurrent feedback to isolate the compartmental plasticity
        # pathway: shock → PPL DANs → DAN_MBON → sign-flipped DA → DPR.
        # Without this, DAN→DAN spreads PPL to PAM (diluting aversive DA),
        # MBON→DAN creates odor-driven PAM DA, and MBON→MBON lets
        # potentiated avoid MBONs boost approach MBON firing indirectly.
        net.W_MBON_DAN[:] = 0.0
        net.W_DAN_DAN[:] = 0.0
        net.W_MBON_MBON[:] = 0.0
        w_rest = config.get('w_rest', 1.0)
        wmax = net.W_KC_MBON.max()
        if wmax > 0:
            net.W_KC_MBON *= (w_rest / wmax)
        net.W_initial = net.W_KC_MBON.copy()

    n_kc   = net_fwd.KCs.n
    n_mbon = net_fwd.MBONs.n
    n_dan  = net_fwd.DANs.n
    dt     = config['dt']
    print(f"\nNetwork: {n_kc} KCs, {n_mbon} MBONs, {n_dan} DANs")

    # -- DAN classification --------------------------------------------------
    dan_ann_path = 'data/connectomes/processed/mb_circuit_right_dan_annotations.csv'
    ids_path     = 'data/connectomes/processed/mb_circuit_right_ids.npz'

    ppl_indices = load_ppl_indices(dan_ann_path, ids_path)
    pam_indices = load_pam_indices(dan_ann_path, ids_path)
    print(f"PPL (aversive):  {len(ppl_indices)} DANs")
    print(f"PAM (appetitive): {len(pam_indices)} DANs")

    # -- MBON classification & sign mask -------------------------------------
    mbon_groups = classify_mbons()
    mbon_sign   = build_mbon_sign_mask(n_mbon, mbon_groups)
    approach_idx = mbon_groups['approach']
    avoid_idx    = mbon_groups['avoid']
    print(f"MBON sign mask: {int((mbon_sign == +1).sum())} approach (+1), "
          f"{int((mbon_sign == -1).sum())} avoid (-1), "
          f"{int((mbon_sign == 0).sum())} other (0)")

    # -- Enable compartmental plasticity on all networks ---------------------
    for net in (net_fwd, net_bwd, net_naive):
        net.set_compartmental_plasticity(ppl_indices, pam_indices, mbon_sign)

    mbon_labels = load_mbon_labels(
        'data/connectomes/processed/mb_circuit_right_mbon_annotations.csv',
        ids_path,
    )

    # -- Odor pattern --------------------------------------------------------
    np.random.seed(7)
    odor_pattern, odor_kcs = create_sparse_odor(n_kc, sparsity=0.05, strength=15.0)
    print(f"Odor: {len(odor_kcs)} KCs active ({100*len(odor_kcs)/n_kc:.1f}%)")

    # Shock stimulus: only PPL neurons (aversive conditioning)
    ppl_stim = np.zeros(n_dan)
    ppl_stim[ppl_indices] = 80.0

    # -- Trial parameters ----------------------------------------------------
    n_trials       = 5
    trial_duration = 4000

    fwd_cs_on, fwd_cs_off = 500, 2500
    fwd_us_on, fwd_us_off = 2000, 2500

    bwd_us_on, bwd_us_off = 500, 1000
    bwd_cs_on, bwd_cs_off = 1500, 3500

    test_cs_on, test_cs_off = 500, 2500

    print(f"\nForward:  CS {fwd_cs_on}–{fwd_cs_off} ms | "
          f"US {fwd_us_on}–{fwd_us_off} ms")
    print(f"Backward: US {bwd_us_on}–{bwd_us_off} ms | "
          f"CS {bwd_cs_on}–{bwd_cs_off} ms")
    print(f"Trials: {n_trials}\n")

    # -- Record trial 1 dynamics ---------------------------------------------
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

    # -- Test trials ---------------------------------------------------------
    print("\nRunning test trials (odor only) ...")
    test_fwd   = test_trial(net_fwd, odor_pattern, dt, trial_duration,
                            test_cs_on, test_cs_off, record_every=1)
    test_bwd   = test_trial(net_bwd, odor_pattern, dt, trial_duration,
                            test_cs_on, test_cs_off, record_every=1)
    test_naive = test_trial(net_naive, odor_pattern, dt, trial_duration,
                            test_cs_on, test_cs_off, record_every=1)

    dW_fwd = net_fwd.get_weight_change()
    dW_bwd = net_bwd.get_weight_change()
    odor_kcs_sorted = np.sort(odor_kcs)

    # -- Weight change analysis ----------------------------------------------
    print(f"\nForward  ΔW: mean={dW_fwd.mean():.6f}  "
          f"min={dW_fwd.min():.6f}  max={dW_fwd.max():.6f}")
    print(f"Backward ΔW: mean={dW_bwd.mean():.6f}  "
          f"min={dW_bwd.min():.6f}  max={dW_bwd.max():.6f}")

    for label, idx in [('Approach', approach_idx), ('Avoid', avoid_idx)]:
        dw_f = dW_fwd[np.ix_(idx, odor_kcs_sorted)].mean()
        dw_b = dW_bwd[np.ix_(idx, odor_kcs_sorted)].mean()
        print(f"  {label:8s} MBONs — Fwd ΔW: {dw_f:+.6f}, Bwd ΔW: {dw_b:+.6f}")

    t_vec = test_fwd['t']
    cs_mask = (t_vec >= test_cs_on) & (t_vec < test_cs_off)

    # ========================================================================
    # FIGURE 1 — Approach vs Avoid MBON rates (compartmental model)
    # ========================================================================
    fig1, (ax_app, ax_avd) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for ax, idx, group_name in [
        (ax_app, approach_idx, 'Approach MBONs (glutamatergic) — PPL should DEPRESS'),
        (ax_avd, avoid_idx,   'Avoid MBONs (GABAergic) — PPL should POTENTIATE'),
    ]:
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
    fig1.suptitle('Compartmental MV Model — Approach vs Avoid MBON Responses\n'
                  'Aversive conditioning with valence-opponent DA signaling',
                  fontweight='bold', fontsize=11)
    fig1.tight_layout()
    fig1.savefig('results/v4e_compartmental_approach_avoid.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: results/v4e_compartmental_approach_avoid.png")

    # ========================================================================
    # FIGURE 2 — Per-MBON weight change
    # ========================================================================
    mean_dW_fwd = dW_fwd[:, odor_kcs_sorted].mean(axis=1)
    mean_dW_bwd = dW_bwd[:, odor_kcs_sorted].mean(axis=1)

    x = np.arange(n_mbon)
    width = 0.4

    fig2, ax2 = plt.subplots(figsize=(16, 5))
    ax2.bar(x - width/2, mean_dW_fwd, width, label='Forward',
            color='steelblue', alpha=0.8)
    ax2.bar(x + width/2, mean_dW_bwd, width, label='Backward',
            color='firebrick', alpha=0.8)
    ax2.axhline(0, color='k', lw=0.8)

    for i in approach_idx:
        ax2.axvspan(i - 0.5, i + 0.5, alpha=0.08, color='green')
    for i in avoid_idx:
        ax2.axvspan(i - 0.5, i + 0.5, alpha=0.08, color='red')

    ax2.set_xticks(x)
    ax2.set_xticklabels(mbon_labels, rotation=45, ha='right', fontsize=7)
    ax2.set_ylabel('Mean ΔW (over odor-active KCs)', fontsize=10)
    ax2.set_title('Compartmental MV Model — Per-MBON Weight Change\n'
                  'Green = approach (expect depression), Red = avoid (expect potentiation)',
                  fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    fig2.tight_layout()
    fig2.savefig('results/v4e_compartmental_dw_per_mbon.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/v4e_compartmental_dw_per_mbon.png")

    # ========================================================================
    # FIGURE 3 — Preference summary
    # ========================================================================
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    bar_x = np.arange(3)
    width3 = 0.35
    cond_names = ['Naive', 'Forward', 'Backward']
    colors = ['gray', 'steelblue', 'firebrick']

    app_vals, avd_vals = [], []
    for mbon_r in [test_naive['mbon_r'], test_fwd['mbon_r'], test_bwd['mbon_r']]:
        app_vals.append(mbon_r[cs_mask][:, approach_idx].mean())
        avd_vals.append(mbon_r[cs_mask][:, avoid_idx].mean())

    ax3.bar(bar_x - width3/2, app_vals, width3, label='Approach',
            color=colors, alpha=0.85, edgecolor='k', linewidth=0.5)
    ax3.bar(bar_x + width3/2, avd_vals, width3, label='Avoid',
            color=colors, alpha=0.45, edgecolor='k', linewidth=0.5, hatch='///')
    ax3.set_xticks(bar_x)
    ax3.set_xticklabels(cond_names, fontsize=11)
    ax3.set_ylabel('Mean firing rate during CS (Hz)', fontsize=10)
    ax3.set_title('Compartmental MV Model — Preference Summary\n'
                  '(solid = approach, hatched = avoid)', fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    fig3.tight_layout()
    fig3.savefig('results/v4e_compartmental_preference.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/v4e_compartmental_preference.png")

    # ========================================================================
    # FIGURE 4 — Training dynamics
    # ========================================================================
    fig4, axes4 = plt.subplots(5, 2, figsize=(16, 14), sharex='col')

    for col, (rec, cond_label, cs_on, cs_off, us_on, us_off) in enumerate([
        (rec_fwd, 'Forward', fwd_cs_on, fwd_cs_off, fwd_us_on, fwd_us_off),
        (rec_bwd, 'Backward', bwd_cs_on, bwd_cs_off, bwd_us_on, bwd_us_off),
    ]):
        t = rec['t']

        ax = axes4[0, col]
        ax.axvspan(cs_on, cs_off, alpha=0.3, color='dodgerblue', label='CS (odor)')
        ax.axvspan(us_on, us_off, alpha=0.3, color='red', label='US (PPL shock)')
        ax.set_ylim(0, 1); ax.set_yticks([])
        ax.set_title(f'{cond_label} — Compartmental MV', fontweight='bold', fontsize=12)
        ax.legend(fontsize=8, loc='upper right')

        ax = axes4[1, col]
        ax.plot(t, rec['mean_kc_r'], color='green', lw=1.5)
        ax.set_ylabel('KC rate (Hz)', fontsize=9); ax.grid(alpha=0.25)

        ax = axes4[2, col]
        ax.plot(t, rec['mean_da'], color='purple', lw=1.5)
        ax.set_ylabel('DA to MBONs', fontsize=9); ax.grid(alpha=0.25)

        ax = axes4[3, col]
        ax.plot(t, rec['mean_d_up'], color='orange', lw=1.5, label='D_up (τ=200ms)')
        ax.plot(t, rec['mean_d_down'], color='darkred', lw=1.5, label='D_down (τ=2000ms)')
        ax.set_ylabel('DA filter', fontsize=9)
        ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.25)

        ax = axes4[4, col]
        ax.plot(t, rec['mean_delta'], color='navy', lw=1.5, label='δ = D_down − D_up')
        ax_tw = ax.twinx()
        ax_tw.plot(t, rec['cum_mean_dw'], color='crimson', lw=1.5, ls='--',
                   label='Cum. mean ΔW')
        ax.set_ylabel('δ', fontsize=9, color='navy')
        ax_tw.set_ylabel('Cum. ΔW', fontsize=9, color='crimson')
        ax.set_xlabel('Time (ms)', fontsize=10)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_tw.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
        ax.grid(alpha=0.25)

    fig4.suptitle('Training Dynamics — Compartmental MV Model (Trial 1)\n'
                  'DA signal is sign-flipped for approach vs avoid compartments',
                  fontweight='bold', fontsize=12)
    fig4.tight_layout()
    fig4.savefig('results/v4e_compartmental_dynamics.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/v4e_compartmental_dynamics.png")

    # ========================================================================
    # FIGURE 5 — KC→MBON weight change heatmaps
    # ========================================================================
    vmax = max(np.abs(dW_fwd[:, odor_kcs_sorted]).max(),
               np.abs(dW_bwd[:, odor_kcs_sorted]).max()) + 1e-9

    fig5, axes5 = plt.subplots(1, 2, figsize=(16, 9))

    for ax, dW, title in zip(
        axes5,
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
        # Color-code MBON labels by group
        ylabels = []
        ycolors = []
        for i, lbl in enumerate(mbon_labels):
            ylabels.append(lbl)
            if i in approach_idx:
                ycolors.append('green')
            elif i in avoid_idx:
                ycolors.append('red')
            else:
                ycolors.append('black')
        ax.set_yticklabels(ylabels, fontsize=6)
        for ticklabel, color in zip(ax.get_yticklabels(), ycolors):
            ticklabel.set_color(color)

    fig5.suptitle(f'KC→MBON Synaptic Weight Changes After {n_trials} Trials\n'
                  'Compartmental MV Model (green = approach, red = avoid)',
                  fontweight='bold', fontsize=12)
    fig5.tight_layout()
    fig5.savefig('results/v4e_compartmental_weight_heatmaps.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/v4e_compartmental_weight_heatmaps.png")

    # ========================================================================
    # FIGURE 6 — Training vs Post-Training: MBON rates during trial 1 vs test
    # ========================================================================
    # Use the recorded training trial data (trial 1) for "during training"
    # and the test trial data for "post-training"
    t_train = rec_fwd['t']
    t_test  = t_vec

    fig6, axes6 = plt.subplots(2, 2, figsize=(16, 10))

    # Top row: During training (trial 1) — approach and avoid group rates
    # recorded from record_training_trial (we have mean KC rates but not
    # per-MBON rates — we need to re-derive from the recording)
    # Instead, show the training trial MBON response using a quick re-run

    # Run single training trials recording MBON rates for visualization
    # Reuse net_naive (untrained) for "during training, trial 1" comparison
    net_train_viz = MushroomBodyNetwork(config)
    net_train_viz.W_KC_DAN[:] = 0.0
    net_train_viz.W_MBON_DAN[:] = 0.0
    net_train_viz.W_DAN_DAN[:] = 0.0
    net_train_viz.W_MBON_MBON[:] = 0.0
    wmax = net_train_viz.W_KC_MBON.max()
    if wmax > 0:
        net_train_viz.W_KC_MBON *= (w_rest / wmax)
    net_train_viz.W_initial = net_train_viz.W_KC_MBON.copy()
    net_train_viz.set_compartmental_plasticity(ppl_indices, pam_indices, mbon_sign)

    # Record MBON rates during a forward training trial
    n_steps_train = int(trial_duration / dt)
    record_every = 10
    n_rec_train = n_steps_train // record_every
    zeros_kc  = np.zeros(n_kc)
    zeros_dan = np.zeros(n_dan)

    train_mbon_r = np.zeros((n_rec_train, n_mbon))
    t_train_vec  = np.zeros(n_rec_train)

    net_train_viz.reset_activity()
    rec_i = 0
    for step in range(n_steps_train):
        t_now = step * dt
        I_odor   = odor_pattern if fwd_cs_on <= t_now < fwd_cs_off else zeros_kc
        x_punish = ppl_stim    if fwd_us_on <= t_now < fwd_us_off else zeros_dan
        net_train_viz.step(I_odor, x_punish)
        if step % record_every == 0 and rec_i < n_rec_train:
            train_mbon_r[rec_i] = net_train_viz.MBONs.r
            t_train_vec[rec_i]  = t_now
            rec_i += 1

    # Top-left: During training — approach MBONs
    ax = axes6[0, 0]
    ax.axvspan(fwd_cs_on, fwd_cs_off, alpha=0.12, color='dodgerblue', label='CS')
    ax.axvspan(fwd_us_on, fwd_us_off, alpha=0.12, color='red', label='US')
    ax.plot(t_train_vec, train_mbon_r[:, approach_idx].mean(axis=1),
            lw=2, color='green', label='Approach')
    ax.plot(t_train_vec, train_mbon_r[:, avoid_idx].mean(axis=1),
            lw=2, color='red', label='Avoid')
    ax.set_ylabel('Mean firing rate (Hz)', fontsize=10)
    ax.set_title('During Training (Forward, Trial 1)', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.25)

    # Top-right: During training — all individual MBONs
    ax = axes6[0, 1]
    ax.axvspan(fwd_cs_on, fwd_cs_off, alpha=0.12, color='dodgerblue')
    ax.axvspan(fwd_us_on, fwd_us_off, alpha=0.12, color='red')
    for j in approach_idx:
        ax.plot(t_train_vec, train_mbon_r[:, j], lw=0.5, alpha=0.4, color='green')
    for j in avoid_idx:
        ax.plot(t_train_vec, train_mbon_r[:, j], lw=0.5, alpha=0.4, color='red')
    ax.set_ylabel('Firing rate (Hz)', fontsize=10)
    ax.set_title('During Training — Individual MBONs', fontweight='bold')
    ax.grid(alpha=0.25)

    # Bottom-left: Post-training test — approach and avoid groups
    ax = axes6[1, 0]
    ax.axvspan(test_cs_on, test_cs_off, alpha=0.12, color='dodgerblue', label='CS')
    ax.plot(t_test, test_fwd['mbon_r'][:, approach_idx].mean(axis=1),
            lw=2, color='green', label='Approach (fwd)')
    ax.plot(t_test, test_fwd['mbon_r'][:, avoid_idx].mean(axis=1),
            lw=2, color='red', label='Avoid (fwd)')
    ax.plot(t_test, test_naive['mbon_r'][:, approach_idx].mean(axis=1),
            lw=2, color='green', ls='--', alpha=0.5, label='Approach (naive)')
    ax.plot(t_test, test_naive['mbon_r'][:, avoid_idx].mean(axis=1),
            lw=2, color='red', ls='--', alpha=0.5, label='Avoid (naive)')
    ax.set_ylabel('Mean firing rate (Hz)', fontsize=10)
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_title('Post-Training Test (Forward, Odor Only)', fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.25)

    # Bottom-right: Post-training test — individual MBONs
    ax = axes6[1, 1]
    ax.axvspan(test_cs_on, test_cs_off, alpha=0.12, color='dodgerblue')
    for j in approach_idx:
        ax.plot(t_test, test_fwd['mbon_r'][:, j], lw=0.5, alpha=0.4, color='green')
    for j in avoid_idx:
        ax.plot(t_test, test_fwd['mbon_r'][:, j], lw=0.5, alpha=0.4, color='red')
    ax.set_ylabel('Firing rate (Hz)', fontsize=10)
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_title('Post-Training Test — Individual MBONs', fontweight='bold')
    ax.grid(alpha=0.25)

    fig6.suptitle('Training vs Post-Training MBON Activity (Forward Conditioning)\n'
                  'Green = approach MBONs, Red = avoid MBONs',
                  fontweight='bold', fontsize=12)
    fig6.tight_layout()
    fig6.savefig('results/v4e_compartmental_train_vs_test.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/v4e_compartmental_train_vs_test.png")

    # ========================================================================
    # Print numerical summary
    # ========================================================================
    print("\n" + "-" * 50)
    print("COMPARTMENTAL MODEL SUMMARY")
    print("-" * 50)
    print(f"  Approach MBONs during CS (Hz):")
    print(f"    Naive:    {app_vals[0]:.2f}")
    print(f"    Forward:  {app_vals[1]:.2f}  (Δ = {app_vals[1]-app_vals[0]:+.2f})")
    print(f"    Backward: {app_vals[2]:.2f}  (Δ = {app_vals[2]-app_vals[0]:+.2f})")
    print(f"  Avoid MBONs during CS (Hz):")
    print(f"    Naive:    {avd_vals[0]:.2f}")
    print(f"    Forward:  {avd_vals[1]:.2f}  (Δ = {avd_vals[1]-avd_vals[0]:+.2f})")
    print(f"    Backward: {avd_vals[2]:.2f}  (Δ = {avd_vals[2]-avd_vals[0]:+.2f})")

    expected_fwd = (app_vals[1] < app_vals[0]) and (avd_vals[1] >= avd_vals[0])
    print(f"\n  Forward conditioning check:")
    print(f"    Approach depressed from naive? {app_vals[1] < app_vals[0]}")
    print(f"    Avoid potentiated from naive?  {avd_vals[1] >= avd_vals[0]}")
    if expected_fwd:
        print("    --> CORRECT: aversive learning produces approach depression + avoid potentiation")
    else:
        print("    --> UNEXPECTED: check model parameters or connectivity")

    plt.show()
    print("\n" + "=" * 65)
    print("V4e Complete.")
    print("=" * 65)


if __name__ == '__main__':
    run_v4e()
