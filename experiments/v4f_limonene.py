"""V4f: Biologically realistic limonene odor experiment with connectivity-derived valence.

Combines:
  - Gkanias et al. (2022) DPR — two-timescale dopaminergic plasticity rule
  - Bennett et al. (2021) MV model — compartmental valence-opponent DA signaling
  - Huang et al. (2024) rate dynamics
  - Connectivity-derived MBON valence from FlyWire DAN→MBON connectivity

Instead of manually hardcoded MBON labels, derives approach/avoid classification
from the ratio of PPL (aversive) vs PAM (appetitive) DAN input to each MBON,
then applies compartmental sign-flipped DA routing.

Runs both hard-threshold and continuous sign modes for comparison.

Produces 8 figures per sign mode (16 total) plus a comparison table.
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
from mbmodel.stimuli import create_odor_from_door, create_odor_from_door_fallback

from v4_conditioning import (
    load_ppl_indices, load_pam_indices, load_mbon_labels, classify_mbons,
    build_mbon_sign_mask, derive_mbon_sign_from_connectivity,
    print_mbon_sign_comparison,
    train_trial, record_training_trial, test_trial,
)


def make_network(config, ppl_indices, pam_indices, sign_mask, recurrent=False):
    """Create network with compartmental plasticity enabled.

    Parameters
    ----------
    recurrent : bool
        If True, keep recurrent pathways (MBON→DAN, MBON→MBON, DAN→DAN)
        from the connectome. If False (default), zero them out for isolated
        feedforward plasticity analysis.
    """
    net = MushroomBodyNetwork(config)
    w_rest = config.get('w_rest', 1.0)
    net.W_KC_DAN[:] = 0.0
    if not recurrent:
        net.W_MBON_DAN[:] = 0.0
        net.W_DAN_DAN[:] = 0.0
        net.W_MBON_MBON[:] = 0.0
    # else: keep connectome-derived MBON→DAN, MBON→MBON, DAN→DAN (0.5×)
    wmax = net.W_KC_MBON.max()
    if wmax > 0:
        net.W_KC_MBON *= (w_rest / wmax)
    net.W_initial = net.W_KC_MBON.copy()
    net.set_compartmental_plasticity(ppl_indices, pam_indices, sign_mask)
    return net


def run_one_sign_mode(mode_label, sign_mask, config, ppl_indices, pam_indices,
                      odor_pattern, odor_kcs, mbon_labels, mbon_groups, prefix,
                      recurrent=False):
    """Run full forward+backward conditioning with a given sign mask.

    Returns dict with results for later comparison.
    """
    print(f"\n{'='*65}")
    print(f"  Running: {mode_label}")
    if recurrent:
        print(f"  ** Recurrent pathways ENABLED (MBON->DAN, MBON->MBON, DAN->DAN)")
    print(f"{'='*65}")

    approach_idx = mbon_groups['approach']
    avoid_idx    = mbon_groups['avoid']
    n_kc   = config['_n_kc']
    n_mbon = config['_n_mbon']
    n_dan  = config['_n_dan']
    dt     = config['dt']

    # -- Networks with compartmental plasticity --------------------------------
    net_fwd       = make_network(config, ppl_indices, pam_indices, sign_mask, recurrent)
    net_bwd       = make_network(config, ppl_indices, pam_indices, sign_mask, recurrent)
    net_naive     = make_network(config, ppl_indices, pam_indices, sign_mask, recurrent)
    net_train_viz = make_network(config, ppl_indices, pam_indices, sign_mask, recurrent)

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
    print("  Running test trials (odor only) ...")
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
    print(f"\n  Forward  ΔW: mean={dW_fwd.mean():.6f}  "
          f"min={dW_fwd.min():.6f}  max={dW_fwd.max():.6f}")
    print(f"  Backward ΔW: mean={dW_bwd.mean():.6f}  "
          f"min={dW_bwd.min():.6f}  max={dW_bwd.max():.6f}")

    for label, idx in [('Approach', approach_idx), ('Avoid', avoid_idx)]:
        dw_f = dW_fwd[np.ix_(idx, odor_kcs_sorted)].mean()
        dw_b = dW_bwd[np.ix_(idx, odor_kcs_sorted)].mean()
        print(f"    {label:8s} MBONs — Fwd ΔW: {dw_f:+.6f}, Bwd ΔW: {dw_b:+.6f}")

    t_vec  = test_fwd['t']
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

    w_rest = config.get('w_rest', 1.0)

    # ====================================================================
    # FIGURE 1 — Training vs Post-Training MBON rates
    # ====================================================================
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
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.25)

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
    ax.set_ylabel('Mean firing rate (Hz)')
    ax.set_xlabel('Time (ms)')
    ax.set_title('Post-Training Test (Forward, Odor Only)', fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.25)

    ax = axes1[1, 1]
    ax.axvspan(test_cs_on, test_cs_off, alpha=0.12, color='dodgerblue')
    for j in approach_idx:
        ax.plot(t_vec, test_fwd['mbon_r'][:, j], lw=0.5, alpha=0.4, color='green')
    for j in avoid_idx:
        ax.plot(t_vec, test_fwd['mbon_r'][:, j], lw=0.5, alpha=0.4, color='red')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_xlabel('Time (ms)')
    ax.set_title('Post-Training Test — Individual MBONs', fontweight='bold')
    ax.grid(alpha=0.25)

    fig1.suptitle(f'Training vs Post-Training — Limonene ({mode_label})\n'
                  'Green = approach MBONs, Red = avoid MBONs',
                  fontweight='bold', fontsize=12)
    fig1.tight_layout()
    fig1.savefig(f'results/{prefix}_train_vs_test.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"\n--- Figures ({mode_label}) ---")
    print(f"  Saved: results/{prefix}_train_vs_test.png")

    # ====================================================================
    # FIGURES 2 & 3 — Weight-change heatmaps
    # ====================================================================
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
        ax.set_title(f'{cond} — KC→MBON ΔW — Limonene ({mode_label})',
                     fontweight='bold')
        ax.set_yticks(range(n_mbon))
        ax.set_yticklabels(ylabels, fontsize=6)
        for ticklabel, color in zip(ax.get_yticklabels(), ycolors):
            ticklabel.set_color(color)
        fig.tight_layout()
        fig.savefig(f'results/{prefix}_dw_heatmap_{suffix}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: results/{prefix}_dw_heatmap_{suffix}.png")

    # ====================================================================
    # FIGURES 4 & 5 — Per-MBON bar charts
    # ====================================================================
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
        ax.set_title(f'{cond} — Per-MBON ΔW — Limonene ({mode_label})',
                     fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(f'results/{prefix}_dw_barplot_{suffix}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: results/{prefix}_dw_barplot_{suffix}.png")

    # ====================================================================
    # FIGURE 6 — Training dynamics (5×2 grid)
    # ====================================================================
    fig6, axes6 = plt.subplots(5, 2, figsize=(16, 14), sharex='col')

    for col, (rec, cond_label, cs_on, cs_off, us_on, us_off) in enumerate([
        (rec_fwd, 'Forward', fwd_cs_on, fwd_cs_off, fwd_us_on, fwd_us_off),
        (rec_bwd, 'Backward', bwd_cs_on, bwd_cs_off, bwd_us_on, bwd_us_off),
    ]):
        t = rec['t']

        ax = axes6[0, col]
        ax.axvspan(cs_on, cs_off, alpha=0.3, color='dodgerblue', label='CS (odor)')
        ax.axvspan(us_on, us_off, alpha=0.3, color='red', label='US (PPL shock)')
        ax.set_ylim(0, 1); ax.set_yticks([])
        ax.set_title(f'{cond_label} — Limonene ({mode_label})',
                     fontweight='bold', fontsize=12)
        ax.legend(fontsize=8, loc='upper right')

        ax = axes6[1, col]
        ax.plot(t, rec['mean_kc_r'], color='green', lw=1.5)
        ax.set_ylabel('KC rate (Hz)', fontsize=9); ax.grid(alpha=0.25)

        ax = axes6[2, col]
        ax.plot(t, rec['mean_da'], color='purple', lw=1.5)
        ax.set_ylabel('DA to MBONs', fontsize=9); ax.grid(alpha=0.25)

        ax = axes6[3, col]
        ax.plot(t, rec['mean_d_up'], color='orange', lw=1.5, label='D_up (τ=200ms)')
        ax.plot(t, rec['mean_d_down'], color='darkred', lw=1.5, label='D_down (τ=2000ms)')
        ax.set_ylabel('DA filter', fontsize=9)
        ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.25)

        ax = axes6[4, col]
        ax.plot(t, rec['mean_delta'], color='navy', lw=1.5, label='δ = D_▽ − D_△')
        ax_tw = ax.twinx()
        ax_tw.plot(t, rec['cum_mean_dw'], color='crimson', lw=1.5, ls='--',
                   label='Cum. mean ΔW')
        ax.set_ylabel('δ', fontsize=9, color='navy')
        ax_tw.set_ylabel('Cum. ΔW', fontsize=9, color='crimson')
        ax.set_xlabel('Time (ms)')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_tw.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
        ax.grid(alpha=0.25)

    fig6.suptitle(f'Training Dynamics — Limonene ({mode_label}, Trial 1)\n'
                  'Gkanias DPR + Bennett MV compartmental routing',
                  fontweight='bold', fontsize=12)
    fig6.tight_layout()
    fig6.savefig(f'results/{prefix}_dynamics.png', dpi=150, bbox_inches='tight')
    plt.close(fig6)
    print(f"  Saved: results/{prefix}_dynamics.png")

    # ====================================================================
    # FIGURE 7 — Preference summary
    # ====================================================================
    fig7, ax7 = plt.subplots(figsize=(8, 5))
    bar_x = np.arange(3)
    width7 = 0.35
    cond_names = ['Naive', 'Forward', 'Backward']
    colors = ['gray', 'steelblue', 'firebrick']

    ax7.bar(bar_x - width7/2, app_vals, width7, label='Approach',
            color=colors, alpha=0.85, edgecolor='k', linewidth=0.5)
    ax7.bar(bar_x + width7/2, avd_vals, width7, label='Avoid',
            color=colors, alpha=0.45, edgecolor='k', linewidth=0.5, hatch='///')
    ax7.set_xticks(bar_x)
    ax7.set_xticklabels(cond_names, fontsize=11)
    ax7.set_ylabel('Mean firing rate during CS (Hz)')
    ax7.set_title(f'Limonene ({mode_label}) — Preference Summary\n'
                  '(solid = approach, hatched = avoid)', fontweight='bold')
    ax7.legend(fontsize=10)
    ax7.grid(axis='y', alpha=0.3)
    fig7.tight_layout()
    fig7.savefig(f'results/{prefix}_preference.png', dpi=150, bbox_inches='tight')
    plt.close(fig7)
    print(f"  Saved: results/{prefix}_preference.png")

    # ====================================================================
    # FIGURE 8 — Side-by-side heatmaps
    # ====================================================================
    fig8, axes8 = plt.subplots(1, 2, figsize=(16, 9))

    for ax, dW, title in zip(
        axes8, [dW_fwd, dW_bwd],
        ['Forward Conditioning', 'Backward Conditioning'],
    ):
        im = ax.imshow(dW[:, odor_kcs_sorted], aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, interpolation='nearest')
        plt.colorbar(im, ax=ax, label='ΔW (weight change)')
        ax.set_xlabel(f'Odor-active KCs ({len(odor_kcs_sorted)} of {n_kc})')
        ax.set_ylabel('MBON')
        ax.set_title(f'KC→MBON ΔW — {title}', fontweight='bold')
        ax.set_yticks(range(n_mbon))
        ax.set_yticklabels(ylabels, fontsize=6)
        for ticklabel, color in zip(ax.get_yticklabels(), ycolors):
            ticklabel.set_color(color)

    fig8.suptitle(f'KC→MBON Weight Changes — Limonene ({mode_label})\n'
                  'green = approach, red = avoid',
                  fontweight='bold', fontsize=12)
    fig8.tight_layout()
    fig8.savefig(f'results/{prefix}_weight_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close(fig8)
    print(f"  Saved: results/{prefix}_weight_heatmaps.png")

    # ====================================================================
    # Numerical summary
    # ====================================================================
    print(f"\n  {'-'*50}")
    print(f"  LIMONENE {mode_label.upper()} SUMMARY")
    print(f"  {'-'*50}")
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
        print("    --> CORRECT: aversive learning produces approach depression "
              "+ avoid potentiation")
    else:
        print("    --> UNEXPECTED: check model parameters or connectivity")

    return {'app_vals': app_vals, 'avd_vals': avd_vals,
            'dW_fwd': dW_fwd, 'dW_bwd': dW_bwd}


def run_v4f():
    print("=" * 65)
    print("V4f: Limonene — Connectivity-Derived Valence + Compartmental Plasticity")
    print("    Gkanias DPR + Bennett MV + Huang Dynamics")
    print("=" * 65)

    # -- Config --------------------------------------------------------------
    with open('experiments/configs/config.yaml') as f:
        config = yaml.safe_load(f)
    config['dt'] = 1.0
    config['learning_rate'] = 5e-6
    config['kc_gating_only'] = True

    # -- Probe network for dimensions ----------------------------------------
    probe = MushroomBodyNetwork(config)
    n_kc   = probe.KCs.n
    n_mbon = probe.MBONs.n
    n_dan  = probe.DANs.n
    config['_n_kc']   = n_kc
    config['_n_mbon'] = n_mbon
    config['_n_dan']  = n_dan
    print(f"\nNetwork: {n_kc} KCs, {n_mbon} MBONs, {n_dan} DANs")

    # -- DAN classification --------------------------------------------------
    dan_ann_path = 'data/connectomes/processed/mb_circuit_right_dan_annotations.csv'
    ids_path     = 'data/connectomes/processed/mb_circuit_right_ids.npz'

    ppl_indices = load_ppl_indices(dan_ann_path, ids_path)
    pam_indices = load_pam_indices(dan_ann_path, ids_path)
    print(f"PPL (aversive):   {len(ppl_indices)} DANs")
    print(f"PAM (appetitive): {len(pam_indices)} DANs")

    # -- MBON classification -------------------------------------------------
    mbon_groups = classify_mbons()
    approach_idx = mbon_groups['approach']
    avoid_idx    = mbon_groups['avoid']
    print(f"MBONs (manual): {len(approach_idx)} approach, {len(avoid_idx)} avoid, "
          f"{len(mbon_groups['other'])} other")

    mbon_labels = load_mbon_labels(
        'data/connectomes/processed/mb_circuit_right_mbon_annotations.csv',
        ids_path,
    )

    # -- Derive MBON sign from connectivity ----------------------------------
    W_DAN_MBON = probe.W_DAN_MBON  # (n_mbon, n_dan)
    manual_sign = build_mbon_sign_mask(n_mbon, mbon_groups)

    hard_sign, hard_ppl_frac = derive_mbon_sign_from_connectivity(
        W_DAN_MBON, ppl_indices, pam_indices, mode='hard')
    cont_sign, cont_ppl_frac = derive_mbon_sign_from_connectivity(
        W_DAN_MBON, ppl_indices, pam_indices, mode='continuous')

    print(f"\nConnectivity-derived (hard): "
          f"{int((hard_sign > 0).sum())} approach, "
          f"{int((hard_sign < 0).sum())} avoid, "
          f"{int((hard_sign == 0).sum())} neutral")
    print(f"Connectivity-derived (continuous): "
          f"{int((cont_sign > 0).sum())} positive, "
          f"{int((cont_sign < 0).sum())} negative, "
          f"{int((cont_sign == 0).sum())} zero")

    # Print comparison table
    print_mbon_sign_comparison(mbon_labels, manual_sign, hard_sign,
                                hard_ppl_frac, mbon_groups)

    # -- Limonene odor pattern -----------------------------------------------
    door_csv = 'data/odorants/XMGQYMWWDOXHJM-UHFFFAOYSA-N.csv'
    pn_kc_npz = 'data/connectomes/processed/mb_circuit_right_pn_to_kc_weights.npz'
    pn_ann_csv = 'data/connectomes/processed/mb_circuit_right_pn_annotations.csv'
    kc_ann_csv = 'data/connectomes/processed/mb_circuit_right_kc_annotations.csv'

    print("\nBuilding limonene odor pattern ...")
    if os.path.exists(pn_kc_npz) and os.path.exists(pn_ann_csv):
        print("  Using PN→KC connectivity from cache")
        odor_pattern, odor_kcs = create_odor_from_door(
            csv_path=door_csv,
            pn_kc_weights_path=pn_kc_npz,
            pn_ann_path=pn_ann_csv,
            n_kc=n_kc, strength=15.0, sparsity=0.05,
        )
    else:
        print("  PN→KC data not found — using fallback (KC subtype model)")
        odor_pattern, odor_kcs = create_odor_from_door_fallback(
            csv_path=door_csv,
            n_kc=n_kc,
            kc_ann_path=kc_ann_csv,
            strength=15.0, sparsity=0.05,
        )
    print(f"Limonene: {len(odor_kcs)} KCs active "
          f"({100 * len(odor_kcs) / n_kc:.1f}%)")

    # -- Run with hard threshold sign mask -----------------------------------
    res_hard = run_one_sign_mode(
        mode_label='Hard Threshold (connectivity-derived)',
        sign_mask=hard_sign, config=config,
        ppl_indices=ppl_indices, pam_indices=pam_indices,
        odor_pattern=odor_pattern, odor_kcs=odor_kcs,
        mbon_labels=mbon_labels, mbon_groups=mbon_groups,
        prefix='v4f_hard',
    )

    # -- Run with continuous sign mask ---------------------------------------
    res_cont = run_one_sign_mode(
        mode_label='Continuous (connectivity-derived)',
        sign_mask=cont_sign, config=config,
        ppl_indices=ppl_indices, pam_indices=pam_indices,
        odor_pattern=odor_pattern, odor_kcs=odor_kcs,
        mbon_labels=mbon_labels, mbon_groups=mbon_groups,
        prefix='v4f_cont',
    )

    # -- Optional: recurrent pathway exploration (Eschbach et al. 2020) ------
    if '--recurrent' in sys.argv:
        print("\n" + "#" * 65)
        print("# RECURRENT PATHWAY EXPLORATION")
        print("# Re-enabling MBON->DAN, MBON->MBON, DAN->DAN from connectome")
        print("# (Eschbach et al. 2020: recurrent MBON->DAN critical for")
        print("#  adaptive learning regulation)")
        print("#" * 65)

        res_recurrent = run_one_sign_mode(
            mode_label='Hard Threshold + Recurrent Pathways',
            sign_mask=hard_sign, config=config,
            ppl_indices=ppl_indices, pam_indices=pam_indices,
            odor_pattern=odor_pattern, odor_kcs=odor_kcs,
            mbon_labels=mbon_labels, mbon_groups=mbon_groups,
            prefix='v4f_recurrent',
            recurrent=True,
        )

        # Comparison summary
        print("\n" + "=" * 65)
        print("RECURRENT vs ISOLATED COMPARISON")
        print("=" * 65)
        print(f"                  {'Isolated':>12s}  {'Recurrent':>12s}")
        print(f"  Approach (fwd): {res_hard['app_vals'][1]:>12.2f}  "
              f"{res_recurrent['app_vals'][1]:>12.2f}")
        print(f"  Avoid (fwd):    {res_hard['avd_vals'][1]:>12.2f}  "
              f"{res_recurrent['avd_vals'][1]:>12.2f}")
        print(f"  Approach (bwd): {res_hard['app_vals'][2]:>12.2f}  "
              f"{res_recurrent['app_vals'][2]:>12.2f}")
        print(f"  Avoid (bwd):    {res_hard['avd_vals'][2]:>12.2f}  "
              f"{res_recurrent['avd_vals'][2]:>12.2f}")

    print("\n" + "=" * 65)
    print("V4f Complete.")
    print("=" * 65)


if __name__ == '__main__':
    run_v4f()
