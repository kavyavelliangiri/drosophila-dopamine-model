"""V4g: Apple cider vinegar (ACV) odor experiment with connectivity-derived valence.

Combines:
  - Gkanias et al. (2022) DPR — two-timescale dopaminergic plasticity rule
  - Bennett et al. (2021) MV model — compartmental valence-opponent DA signaling
  - Huang et al. (2024) rate dynamics
  - Connectivity-derived MBON valence from FlyWire DAN→MBON connectivity

ACV glomerular activation from Semmelhack & Wang (2009) J Neurosci 29:15511:
  Normal concentration: DM1, DM4, DP1m, DM2, DM3, VM2, VA2  (7 glomeruli)
  High concentration:   + DM5                                 (8 glomeruli)

Runs both concentrations with connectivity-derived compartmental plasticity.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

from mbmodel.models import MushroomBodyNetwork
from mbmodel.stimuli import create_odor_from_glomeruli

from v4_conditioning import (
    load_ppl_indices, load_pam_indices, load_mbon_labels, classify_mbons,
    build_mbon_sign_mask, derive_mbon_sign_from_connectivity,
    print_mbon_sign_comparison,
    train_trial, record_training_trial, test_trial,
)


# ACV glomerular activation (Semmelhack & Wang 2009)
ACV_NORMAL_GLOM = ['DM1', 'DM2', 'DM3', 'DM4', 'DP1m', 'VA2', 'VM2']
ACV_HIGH_GLOM   = ACV_NORMAL_GLOM + ['DM5']


def make_network(config, ppl_indices=None, pam_indices=None, sign_mask=None):
    """Create and configure a MushroomBodyNetwork with compartmental plasticity."""
    net = MushroomBodyNetwork(config)
    w_rest = config.get('w_rest', 1.0)
    net.W_KC_DAN[:] = 0.0
    net.W_MBON_DAN[:] = 0.0
    net.W_DAN_DAN[:] = 0.0
    net.W_MBON_MBON[:] = 0.0
    wmax = net.W_KC_MBON.max()
    if wmax > 0:
        net.W_KC_MBON *= (w_rest / wmax)
    net.W_initial = net.W_KC_MBON.copy()
    if sign_mask is not None:
        net.set_compartmental_plasticity(ppl_indices, pam_indices, sign_mask)
    return net


def run_condition(label, glomeruli, config, ppl_indices, pam_indices,
                  sign_mask, mbon_groups, mbon_labels, prefix):
    """Run forward + backward conditioning for one ACV concentration.

    Returns dict with all results and generates 8 figures.
    """
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  Glomeruli: {', '.join(glomeruli)}")
    print(f"{'='*65}")

    # -- Networks with compartmental plasticity --------------------------------
    net_fwd   = make_network(config, ppl_indices, pam_indices, sign_mask)
    net_bwd   = make_network(config, ppl_indices, pam_indices, sign_mask)
    net_naive = make_network(config, ppl_indices, pam_indices, sign_mask)

    n_kc   = net_fwd.KCs.n
    n_mbon = net_fwd.MBONs.n
    n_dan  = net_fwd.DANs.n
    dt     = config['dt']
    w_rest = config.get('w_rest', 1.0)

    approach_idx = mbon_groups['approach']
    avoid_idx    = mbon_groups['avoid']

    # -- ACV odor pattern ----------------------------------------------------
    pn_kc_npz  = 'data/connectomes/processed/mb_circuit_right_acv_pn_to_kc_weights.npz'
    pn_ann_csv = 'data/connectomes/processed/mb_circuit_right_acv_pn_annotations.csv'

    print("\nBuilding ACV odor pattern ...")
    odor_pattern, odor_kcs = create_odor_from_glomeruli(
        active_glomeruli=glomeruli,
        pn_kc_weights_path=pn_kc_npz,
        pn_ann_path=pn_ann_csv,
        n_kc=n_kc, strength=15.0, sparsity=0.05,
    )
    print(f"ACV: {len(odor_kcs)} KCs active ({100 * len(odor_kcs) / n_kc:.1f}%)")

    # Shock stimulus
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

    for lbl, idx in [('Approach', approach_idx), ('Avoid', avoid_idx)]:
        dw_f = dW_fwd[np.ix_(idx, odor_kcs_sorted)].mean()
        dw_b = dW_bwd[np.ix_(idx, odor_kcs_sorted)].mean()
        print(f"  {lbl:8s} MBONs — Fwd ΔW: {dw_f:+.6f}, Bwd ΔW: {dw_b:+.6f}")

    t_vec   = test_fwd['t']
    cs_mask = (t_vec >= test_cs_on) & (t_vec < test_cs_off)

    app_vals, avd_vals = [], []
    for mbon_r in [test_naive['mbon_r'], test_fwd['mbon_r'], test_bwd['mbon_r']]:
        app_vals.append(mbon_r[cs_mask][:, approach_idx].mean())
        avd_vals.append(mbon_r[cs_mask][:, avoid_idx].mean())

    # -- MBON label colors ---------------------------------------------------
    ylabels, ycolors = [], []
    for i, lbl in enumerate(mbon_labels):
        ylabels.append(lbl)
        if i in approach_idx:
            ycolors.append('green')
        elif i in avoid_idx:
            ycolors.append('red')
        else:
            ycolors.append('black')

    # ================================================================
    # FIGURE 1 — Training vs Post-Training
    # ================================================================
    net_viz = make_network(config, ppl_indices, pam_indices, sign_mask)
    n_steps = int(trial_duration / dt)
    record_every = 10
    n_rec = n_steps // record_every
    zeros_kc  = np.zeros(n_kc)
    zeros_dan = np.zeros(n_dan)

    train_mbon_r = np.zeros((n_rec, n_mbon))
    t_train_vec  = np.zeros(n_rec)
    net_viz.reset_activity()
    ri = 0
    for step in range(n_steps):
        t_now = step * dt
        I_od = odor_pattern if fwd_cs_on <= t_now < fwd_cs_off else zeros_kc
        x_p  = ppl_stim    if fwd_us_on <= t_now < fwd_us_off else zeros_dan
        net_viz.step(I_od, x_p)
        if step % record_every == 0 and ri < n_rec:
            train_mbon_r[ri] = net_viz.MBONs.r
            t_train_vec[ri]  = t_now
            ri += 1

    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes1[0, 0]
    ax.axvspan(fwd_cs_on, fwd_cs_off, alpha=0.12, color='dodgerblue', label='CS')
    ax.axvspan(fwd_us_on, fwd_us_off, alpha=0.12, color='red', label='US')
    ax.plot(t_train_vec, train_mbon_r[:, approach_idx].mean(axis=1),
            lw=2, color='green', label='Approach')
    ax.plot(t_train_vec, train_mbon_r[:, avoid_idx].mean(axis=1),
            lw=2, color='red', label='Avoid')
    ax.set_ylabel('Mean firing rate (Hz)')
    ax.set_title('During Training (Forward, Trial 1)', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.25)

    ax = axes1[0, 1]
    ax.axvspan(fwd_cs_on, fwd_cs_off, alpha=0.12, color='dodgerblue')
    ax.axvspan(fwd_us_on, fwd_us_off, alpha=0.12, color='red')
    for j in approach_idx:
        ax.plot(t_train_vec, train_mbon_r[:, j], lw=0.5, alpha=0.4, color='green')
    for j in avoid_idx:
        ax.plot(t_train_vec, train_mbon_r[:, j], lw=0.5, alpha=0.4, color='red')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title('During Training — Individual MBONs', fontweight='bold')
    ax.grid(alpha=0.25)

    ax = axes1[1, 0]
    ax.axvspan(test_cs_on, test_cs_off, alpha=0.12, color='dodgerblue', label='CS')
    ax.plot(t_vec, test_fwd['mbon_r'][:, approach_idx].mean(axis=1),
            lw=2, color='green', label='Approach (fwd)')
    ax.plot(t_vec, test_fwd['mbon_r'][:, avoid_idx].mean(axis=1),
            lw=2, color='red', label='Avoid (fwd)')
    ax.plot(t_vec, test_naive['mbon_r'][:, approach_idx].mean(axis=1),
            lw=2, color='green', ls='--', alpha=0.5, label='Approach (naive)')
    ax.plot(t_vec, test_naive['mbon_r'][:, avoid_idx].mean(axis=1),
            lw=2, color='red', ls='--', alpha=0.5, label='Avoid (naive)')
    ax.set_ylabel('Mean firing rate (Hz)'); ax.set_xlabel('Time (ms)')
    ax.set_title('Post-Training Test (Forward, Odor Only)', fontweight='bold')
    ax.legend(fontsize=7, loc='upper right'); ax.grid(alpha=0.25)

    ax = axes1[1, 1]
    ax.axvspan(test_cs_on, test_cs_off, alpha=0.12, color='dodgerblue')
    for j in approach_idx:
        ax.plot(t_vec, test_fwd['mbon_r'][:, j], lw=0.5, alpha=0.4, color='green')
    for j in avoid_idx:
        ax.plot(t_vec, test_fwd['mbon_r'][:, j], lw=0.5, alpha=0.4, color='red')
    ax.set_ylabel('Firing rate (Hz)'); ax.set_xlabel('Time (ms)')
    ax.set_title('Post-Training Test — Individual MBONs', fontweight='bold')
    ax.grid(alpha=0.25)

    fig1.suptitle(f'Training vs Post-Training — {label}\n'
                  'Green = approach, Red = avoid', fontweight='bold', fontsize=12)
    fig1.tight_layout()
    fig1.savefig(f'results/{prefix}_train_vs_test.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"\n  Saved: results/{prefix}_train_vs_test.png")

    # ================================================================
    # FIGURES 2 & 3 — Weight-change heatmaps (separate)
    # ================================================================
    vmax = max(np.abs(dW_fwd[:, odor_kcs_sorted]).max(),
               np.abs(dW_bwd[:, odor_kcs_sorted]).max()) + 1e-9

    for dW, cond, suffix in [(dW_fwd, 'Forward', 'forward'),
                              (dW_bwd, 'Backward', 'backward')]:
        fig, ax = plt.subplots(figsize=(12, 9))
        im = ax.imshow(dW[:, odor_kcs_sorted], aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, interpolation='nearest')
        plt.colorbar(im, ax=ax, label='ΔW (weight change)')
        ax.set_xlabel(f'Odor-active KCs ({len(odor_kcs_sorted)} of {n_kc})')
        ax.set_ylabel('MBON')
        ax.set_title(f'{cond} Conditioning — KC→MBON Weight Change ({label})',
                     fontweight='bold')
        ax.set_yticks(range(n_mbon))
        ax.set_yticklabels(ylabels, fontsize=6)
        for tl, c in zip(ax.get_yticklabels(), ycolors):
            tl.set_color(c)
        fig.tight_layout()
        fig.savefig(f'results/{prefix}_dw_heatmap_{suffix}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: results/{prefix}_dw_heatmap_{suffix}.png")

    # ================================================================
    # FIGURES 4 & 5 — Per-MBON bar charts (separate)
    # ================================================================
    mean_dW_fwd = dW_fwd[:, odor_kcs_sorted].mean(axis=1)
    mean_dW_bwd = dW_bwd[:, odor_kcs_sorted].mean(axis=1)
    x_mbon = np.arange(n_mbon)

    for mean_dW, cond, color, suffix in [
        (mean_dW_fwd, 'Forward', 'steelblue', 'forward'),
        (mean_dW_bwd, 'Backward', 'firebrick', 'backward'),
    ]:
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.bar(x_mbon, mean_dW, color=color, alpha=0.8)
        ax.axhline(0, color='k', lw=0.8)
        for i in approach_idx:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color='green')
        for i in avoid_idx:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color='red')
        ax.set_xticks(x_mbon)
        ax.set_xticklabels(mbon_labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Mean ΔW (over odor-active KCs)')
        ax.set_title(f'{cond} Conditioning — Per-MBON Mean Weight Change ({label})',
                     fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(f'results/{prefix}_dw_barplot_{suffix}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: results/{prefix}_dw_barplot_{suffix}.png")

    # ================================================================
    # FIGURE 6 — Training dynamics (5×2 grid)
    # ================================================================
    fig6, axes6 = plt.subplots(5, 2, figsize=(16, 14), sharex='col')

    for col, (rec, cond_label, cs_on, cs_off, us_on, us_off) in enumerate([
        (rec_fwd, 'Forward', fwd_cs_on, fwd_cs_off, fwd_us_on, fwd_us_off),
        (rec_bwd, 'Backward', bwd_cs_on, bwd_cs_off, bwd_us_on, bwd_us_off),
    ]):
        t = rec['t']
        ax = axes6[0, col]
        ax.axvspan(cs_on, cs_off, alpha=0.3, color='dodgerblue', label='CS')
        ax.axvspan(us_on, us_off, alpha=0.3, color='red', label='US')
        ax.set_ylim(0, 1); ax.set_yticks([])
        ax.set_title(f'{cond_label} — {label}', fontweight='bold', fontsize=12)
        ax.legend(fontsize=8, loc='upper right')

        ax = axes6[1, col]
        ax.plot(t, rec['mean_kc_r'], color='green', lw=1.5)
        ax.set_ylabel('KC rate (Hz)', fontsize=9); ax.grid(alpha=0.25)

        ax = axes6[2, col]
        ax.plot(t, rec['mean_da'], color='purple', lw=1.5)
        ax.set_ylabel('DA to MBONs', fontsize=9); ax.grid(alpha=0.25)

        ax = axes6[3, col]
        ax.plot(t, rec['mean_d_up'], color='orange', lw=1.5, label='D_up')
        ax.plot(t, rec['mean_d_down'], color='darkred', lw=1.5, label='D_down')
        ax.set_ylabel('DA filter', fontsize=9)
        ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.25)

        ax = axes6[4, col]
        ax.plot(t, rec['mean_delta'], color='navy', lw=1.5, label='δ')
        ax_tw = ax.twinx()
        ax_tw.plot(t, rec['cum_mean_dw'], color='crimson', lw=1.5, ls='--',
                   label='Cum. ΔW')
        ax.set_ylabel('δ', fontsize=9, color='navy')
        ax_tw.set_ylabel('Cum. ΔW', fontsize=9, color='crimson')
        ax.set_xlabel('Time (ms)')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_tw.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
        ax.grid(alpha=0.25)

    fig6.suptitle(f'Training Dynamics (Trial 1) — {label}',
                  fontweight='bold', fontsize=12)
    fig6.tight_layout()
    fig6.savefig(f'results/{prefix}_dynamics.png', dpi=150, bbox_inches='tight')
    plt.close(fig6)
    print(f"  Saved: results/{prefix}_dynamics.png")

    # ================================================================
    # FIGURE 7 — Preference summary
    # ================================================================
    fig7, ax7 = plt.subplots(figsize=(8, 5))
    bar_x = np.arange(3)
    w7 = 0.35
    cond_names = ['Naive', 'Forward', 'Backward']
    colors = ['gray', 'steelblue', 'firebrick']
    ax7.bar(bar_x - w7/2, app_vals, w7, label='Approach',
            color=colors, alpha=0.85, edgecolor='k', linewidth=0.5)
    ax7.bar(bar_x + w7/2, avd_vals, w7, label='Avoid',
            color=colors, alpha=0.45, edgecolor='k', linewidth=0.5, hatch='///')
    ax7.set_xticks(bar_x)
    ax7.set_xticklabels(cond_names, fontsize=11)
    ax7.set_ylabel('Mean firing rate during CS (Hz)')
    ax7.set_title(f'{label} — Preference Summary\n'
                  '(solid = approach, hatched = avoid)', fontweight='bold')
    ax7.legend(fontsize=10); ax7.grid(axis='y', alpha=0.3)
    fig7.tight_layout()
    fig7.savefig(f'results/{prefix}_preference.png', dpi=150, bbox_inches='tight')
    plt.close(fig7)
    print(f"  Saved: results/{prefix}_preference.png")

    # ================================================================
    # FIGURE 8 — Side-by-side heatmaps
    # ================================================================
    fig8, axes8 = plt.subplots(1, 2, figsize=(16, 9))
    for ax, dW, title in zip(axes8, [dW_fwd, dW_bwd],
                              ['Forward', 'Backward']):
        im = ax.imshow(dW[:, odor_kcs_sorted], aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, interpolation='nearest')
        plt.colorbar(im, ax=ax, label='ΔW')
        ax.set_xlabel(f'Odor-active KCs ({len(odor_kcs_sorted)} of {n_kc})')
        ax.set_ylabel('MBON')
        ax.set_title(f'{title} — KC→MBON ΔW', fontweight='bold')
        ax.set_yticks(range(n_mbon))
        ax.set_yticklabels(ylabels, fontsize=6)
        for tl, c in zip(ax.get_yticklabels(), ycolors):
            tl.set_color(c)
    fig8.suptitle(f'KC→MBON Weight Changes After {n_trials} Trials — {label}\n'
                  'green = approach, red = avoid', fontweight='bold', fontsize=12)
    fig8.tight_layout()
    fig8.savefig(f'results/{prefix}_weight_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close(fig8)
    print(f"  Saved: results/{prefix}_weight_heatmaps.png")

    # ================================================================
    # Numerical summary
    # ================================================================
    print(f"\n  {'-'*50}")
    print(f"  {label.upper()} SUMMARY")
    print(f"  {'-'*50}")
    print(f"  Approach MBONs during CS (Hz):")
    print(f"    Naive:    {app_vals[0]:.2f}")
    print(f"    Forward:  {app_vals[1]:.2f}  (Δ = {app_vals[1]-app_vals[0]:+.2f})")
    print(f"    Backward: {app_vals[2]:.2f}  (Δ = {app_vals[2]-app_vals[0]:+.2f})")
    print(f"  Avoid MBONs during CS (Hz):")
    print(f"    Naive:    {avd_vals[0]:.2f}")
    print(f"    Forward:  {avd_vals[1]:.2f}  (Δ = {avd_vals[1]-avd_vals[0]:+.2f})")
    print(f"    Backward: {avd_vals[2]:.2f}  (Δ = {avd_vals[2]-avd_vals[0]:+.2f})")

    expected = (app_vals[1] < app_vals[0]) and (avd_vals[1] >= avd_vals[0])
    print(f"\n  Forward conditioning check:")
    print(f"    Approach depressed? {app_vals[1] < app_vals[0]}")
    print(f"    Avoid potentiated?  {avd_vals[1] >= avd_vals[0]}")
    if expected:
        print("    --> CORRECT")
    else:
        print("    --> UNEXPECTED")

    return {
        'app_vals': app_vals, 'avd_vals': avd_vals,
        'dW_fwd': dW_fwd, 'dW_bwd': dW_bwd,
        'n_active_kcs': len(odor_kcs),
    }


def run_v4g():
    print("=" * 65)
    print("V4g: Apple Cider Vinegar — Connectivity-Derived Valence + Compartmental Plasticity")
    print("    Gkanias DPR + Bennett MV + Huang Dynamics")
    print("    Semmelhack & Wang (2009) glomerular activation")
    print("=" * 65)

    # -- Config --------------------------------------------------------------
    with open('experiments/configs/config.yaml') as f:
        config = yaml.safe_load(f)
    config['dt'] = 1.0
    config['learning_rate'] = 5e-6
    config['kc_gating_only'] = True

    # -- Shared DAN/MBON classification --------------------------------------
    dan_ann_path = 'data/connectomes/processed/mb_circuit_right_dan_annotations.csv'
    ids_path     = 'data/connectomes/processed/mb_circuit_right_ids.npz'

    ppl_indices = load_ppl_indices(dan_ann_path, ids_path)
    pam_indices = load_pam_indices(dan_ann_path, ids_path)
    mbon_groups = classify_mbons()
    mbon_labels = load_mbon_labels(
        'data/connectomes/processed/mb_circuit_right_mbon_annotations.csv',
        ids_path,
    )

    print(f"PPL (aversive):   {len(ppl_indices)} DANs")
    print(f"PAM (appetitive): {len(pam_indices)} DANs")
    print(f"MBONs (manual): {len(mbon_groups['approach'])} approach, "
          f"{len(mbon_groups['avoid'])} avoid, "
          f"{len(mbon_groups['other'])} other")

    # -- Derive MBON sign from connectivity ----------------------------------
    probe = MushroomBodyNetwork(config)
    n_mbon = probe.MBONs.n
    manual_sign = build_mbon_sign_mask(n_mbon, mbon_groups)

    hard_sign, hard_ppl_frac = derive_mbon_sign_from_connectivity(
        probe.W_DAN_MBON, ppl_indices, pam_indices, mode='hard')

    print(f"\nConnectivity-derived (hard): "
          f"{int((hard_sign > 0).sum())} approach, "
          f"{int((hard_sign < 0).sum())} avoid, "
          f"{int((hard_sign == 0).sum())} neutral")

    print_mbon_sign_comparison(mbon_labels, manual_sign, hard_sign,
                                hard_ppl_frac, mbon_groups)

    # -- Run both concentrations with connectivity-derived sign ----------------
    res_normal = run_condition(
        label='ACV (Normal Concentration)',
        glomeruli=ACV_NORMAL_GLOM,
        config=config, ppl_indices=ppl_indices, pam_indices=pam_indices,
        sign_mask=hard_sign,
        mbon_groups=mbon_groups, mbon_labels=mbon_labels,
        prefix='v4g_acv_normal',
    )

    res_high = run_condition(
        label='ACV (High Concentration)',
        glomeruli=ACV_HIGH_GLOM,
        config=config, ppl_indices=ppl_indices, pam_indices=pam_indices,
        sign_mask=hard_sign,
        mbon_groups=mbon_groups, mbon_labels=mbon_labels,
        prefix='v4g_acv_high',
    )

    # -- Comparison figure ---------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bar_x = np.arange(3)
    w = 0.35
    cond_names = ['Naive', 'Forward', 'Backward']
    colors = ['gray', 'steelblue', 'firebrick']

    for ax, res, title in zip(axes,
                               [res_normal, res_high],
                               ['ACV Normal (7 glom)', 'ACV High (8 glom, +DM5)']):
        ax.bar(bar_x - w/2, res['app_vals'], w, label='Approach',
               color=colors, alpha=0.85, edgecolor='k', linewidth=0.5)
        ax.bar(bar_x + w/2, res['avd_vals'], w, label='Avoid',
               color=colors, alpha=0.45, edgecolor='k', linewidth=0.5, hatch='///')
        ax.set_xticks(bar_x)
        ax.set_xticklabels(cond_names, fontsize=11)
        ax.set_ylabel('Mean firing rate during CS (Hz)')
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

    fig.suptitle('ACV Concentration Comparison — Preference Summary\n'
                 '(solid = approach, hatched = avoid)',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    fig.savefig('results/v4g_acv_concentration_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: results/v4g_acv_concentration_comparison.png")

    print(f"\n{'='*65}")
    print("V4g Complete.")
    print(f"  Normal ACV: {res_normal['n_active_kcs']} KCs active")
    print(f"  High ACV:   {res_high['n_active_kcs']} KCs active")
    print(f"{'='*65}")


if __name__ == '__main__':
    run_v4g()
