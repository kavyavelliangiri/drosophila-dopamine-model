"""V4h: Appetitive conditioning with PAM sugar reward + connectivity-derived valence.

Combines:
  - Gkanias et al. (2022) DPR — two-timescale dopaminergic plasticity rule
  - Bennett et al. (2021) MV model — compartmental valence-opponent DA signaling
  - Huang et al. (2024) rate dynamics
  - Connectivity-derived MBON valence from FlyWire DAN→MBON connectivity

Appetitive conditioning with PAM sugar reward:
  With compartmental routing + PAM activation:
    Approach MBONs (sign > 0): da_eff = (+) * (0 - da_PAM) < 0 -> D_up < 0
                                -> delta = D_down - D_up > 0 -> potentiation
    Avoid MBONs (sign < 0):    da_eff = (-) * (0 - da_PAM) > 0 -> D_up > 0
                                -> delta = D_down - D_up < 0 -> depression

Expected: forward appetitive -> approach potentiated, avoid depressed.
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
from mbmodel.stimuli import create_sparse_odor

from v4_conditioning import (
    load_ppl_indices, load_pam_indices, load_mbon_labels, classify_mbons,
    build_mbon_sign_mask, derive_mbon_sign_from_connectivity,
    print_mbon_sign_comparison,
    build_pam_stimulus,
    train_trial, record_training_trial, test_trial,
)


def make_network_h(config, ppl_indices, pam_indices, sign_mask):
    """Create network with compartmental plasticity for appetitive conditioning."""
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
    net.set_compartmental_plasticity(ppl_indices, pam_indices, sign_mask)
    return net


def run_v4h():
    print("=" * 65)
    print("V4h: Appetitive Conditioning — Connectivity-Derived Valence")
    print("    Gkanias DPR + Bennett MV + Huang Dynamics + PAM Sugar Reward")
    print("=" * 65)

    # -- Config --------------------------------------------------------------
    with open('experiments/configs/config.yaml') as f:
        config = yaml.safe_load(f)
    config['dt'] = 1.0
    config['learning_rate'] = 5e-6
    config['kc_gating_only'] = True

    # -- Probe network for dimensions and connectivity -----------------------
    probe = MushroomBodyNetwork(config)
    n_kc   = probe.KCs.n
    n_mbon = probe.MBONs.n
    n_dan  = probe.DANs.n
    dt     = config['dt']
    print(f"\nNetwork: {n_kc} KCs, {n_mbon} MBONs, {n_dan} DANs")

    # -- DAN classification --------------------------------------------------
    dan_ann_path = 'data/connectomes/processed/mb_circuit_right_dan_annotations.csv'
    ids_path     = 'data/connectomes/processed/mb_circuit_right_ids.npz'

    pam_indices = load_pam_indices(dan_ann_path, ids_path)
    ppl_indices = load_ppl_indices(dan_ann_path, ids_path)
    print(f"PAM (appetitive): {len(pam_indices)} DANs")
    print(f"PPL (aversive):   {len(ppl_indices)} DANs")

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
    manual_sign = build_mbon_sign_mask(n_mbon, mbon_groups)
    hard_sign, hard_ppl_frac = derive_mbon_sign_from_connectivity(
        probe.W_DAN_MBON, ppl_indices, pam_indices, mode='hard')

    print(f"\nConnectivity-derived (hard): "
          f"{int((hard_sign > 0).sum())} approach, "
          f"{int((hard_sign < 0).sum())} avoid, "
          f"{int((hard_sign == 0).sum())} neutral")

    print_mbon_sign_comparison(mbon_labels, manual_sign, hard_sign,
                                hard_ppl_frac, mbon_groups)

    # -- Networks with compartmental plasticity --------------------------------
    net_fwd   = make_network_h(config, ppl_indices, pam_indices, hard_sign)
    net_bwd   = make_network_h(config, ppl_indices, pam_indices, hard_sign)
    net_naive = make_network_h(config, ppl_indices, pam_indices, hard_sign)

    # -- Odor pattern (random sparse, same as V4c) ---------------------------
    np.random.seed(7)
    odor_pattern, odor_kcs = create_sparse_odor(n_kc, sparsity=0.05, strength=15.0)
    print(f"Odor: {len(odor_kcs)} KCs active ({100*len(odor_kcs)/n_kc:.1f}%)")

    # Sugar reward stimulus: only PAM neurons (appetitive conditioning)
    pam_stim = build_pam_stimulus(n_dan, pam_indices, strength=80.0)

    # -- Trial parameters ----------------------------------------------------
    n_trials       = 5
    trial_duration = 4000

    fwd_cs_on, fwd_cs_off = 500, 2500
    fwd_us_on, fwd_us_off = 2000, 2500

    bwd_us_on, bwd_us_off = 500, 1000
    bwd_cs_on, bwd_cs_off = 1500, 3500

    test_cs_on, test_cs_off = 500, 2500

    print(f"\nForward:  CS {fwd_cs_on}-{fwd_cs_off} ms | "
          f"US {fwd_us_on}-{fwd_us_off} ms (sugar overlaps end of CS)")
    print(f"Backward: US {bwd_us_on}-{bwd_us_off} ms | "
          f"CS {bwd_cs_on}-{bwd_cs_off} ms (CS starts 500 ms after sugar ends)")
    print(f"Trials: {n_trials}\n")

    # -- Record trial 1 dynamics ---------------------------------------------
    print("  Recording trial 1 dynamics ...", end=' ', flush=True)
    rec_fwd = record_training_trial(
        net_fwd, odor_pattern, pam_stim, dt, trial_duration,
        fwd_cs_on, fwd_cs_off, fwd_us_on, fwd_us_off, odor_kcs)
    rec_bwd = record_training_trial(
        net_bwd, odor_pattern, pam_stim, dt, trial_duration,
        bwd_cs_on, bwd_cs_off, bwd_us_on, bwd_us_off, odor_kcs)
    print("done")

    # -- Remaining training trials -------------------------------------------
    for trial in range(1, n_trials):
        print(f"  Trial {trial + 1}/{n_trials} ...", end=' ', flush=True)
        train_trial(net_fwd, odor_pattern, pam_stim, dt, trial_duration,
                    fwd_cs_on, fwd_cs_off, fwd_us_on, fwd_us_off)
        train_trial(net_bwd, odor_pattern, pam_stim, dt, trial_duration,
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
    print(f"\nForward  dW: mean={dW_fwd.mean():.6f}  "
          f"min={dW_fwd.min():.6f}  max={dW_fwd.max():.6f}")
    print(f"Backward dW: mean={dW_bwd.mean():.6f}  "
          f"min={dW_bwd.min():.6f}  max={dW_bwd.max():.6f}")

    for label, idx in [('Approach', approach_idx), ('Avoid', avoid_idx)]:
        dw_f = dW_fwd[np.ix_(idx, odor_kcs_sorted)].mean()
        dw_b = dW_bwd[np.ix_(idx, odor_kcs_sorted)].mean()
        print(f"  {label:8s} MBONs -- Fwd dW: {dw_f:+.6f}, Bwd dW: {dw_b:+.6f}")

    t_vec  = test_fwd['t']
    cs_mask = (t_vec >= test_cs_on) & (t_vec < test_cs_off)

    # Compute preference values
    app_vals, avd_vals = [], []
    for mbon_r in [test_naive['mbon_r'], test_fwd['mbon_r'], test_bwd['mbon_r']]:
        app_vals.append(mbon_r[cs_mask][:, approach_idx].mean())
        avd_vals.append(mbon_r[cs_mask][:, avoid_idx].mean())

    # Helper: MBON label colors
    def mbon_label_colors():
        ylabels, ycolors = [], []
        for i, lbl in enumerate(mbon_labels):
            ylabels.append(lbl)
            if i in approach_idx:
                ycolors.append('green')
            elif i in avoid_idx:
                ycolors.append('red')
            else:
                ycolors.append('black')
        return ylabels, ycolors

    ylabels, ycolors = mbon_label_colors()

    # ====================================================================
    # FIGURE 1 -- Training vs Post-Training MBON rates
    # ====================================================================
    net_train_viz = make_network_h(config, ppl_indices, pam_indices, hard_sign)

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
        x_reward = pam_stim    if fwd_us_on <= t_now < fwd_us_off else zeros_dan
        net_train_viz.step(I_odor, x_reward)
        if step % record_every == 0 and rec_i < n_rec_train:
            train_mbon_r[rec_i] = net_train_viz.MBONs.r
            t_train_vec[rec_i]  = t_now
            rec_i += 1

    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes1[0, 0]
    ax.axvspan(fwd_cs_on, fwd_cs_off, alpha=0.12, color='dodgerblue', label='CS')
    ax.axvspan(fwd_us_on, fwd_us_off, alpha=0.12, color='orange', label='US (sugar)')
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
    ax.axvspan(fwd_us_on, fwd_us_off, alpha=0.12, color='orange')
    for j in approach_idx:
        ax.plot(t_train_vec, train_mbon_r[:, j], lw=0.5, alpha=0.4, color='green')
    for j in avoid_idx:
        ax.plot(t_train_vec, train_mbon_r[:, j], lw=0.5, alpha=0.4, color='red')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title('During Training -- Individual MBONs', fontweight='bold')
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
    ax.set_title('Post-Training Test -- Individual MBONs', fontweight='bold')
    ax.grid(alpha=0.25)

    fig1.suptitle('Training vs Post-Training MBON Activity -- Appetitive (PAM Sugar Reward)\n'
                  'Green = approach MBONs, Red = avoid MBONs',
                  fontweight='bold', fontsize=12)
    fig1.tight_layout()
    fig1.savefig('results/v4h_train_vs_test.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("\n--- Figures ---")
    print("  Saved: results/v4h_train_vs_test.png")

    # ====================================================================
    # FIGURE 2 -- Forward conditioning weight-change heatmap
    # ====================================================================
    vmax = max(np.abs(dW_fwd[:, odor_kcs_sorted]).max(),
               np.abs(dW_bwd[:, odor_kcs_sorted]).max()) + 1e-9

    fig2, ax2 = plt.subplots(figsize=(12, 9))
    im2 = ax2.imshow(dW_fwd[:, odor_kcs_sorted], aspect='auto', cmap='RdBu_r',
                     vmin=-vmax, vmax=vmax, interpolation='nearest')
    plt.colorbar(im2, ax=ax2, label='dW (weight change)')
    ax2.set_xlabel(f'Odor-active KCs ({len(odor_kcs_sorted)} of {n_kc})')
    ax2.set_ylabel('MBON')
    ax2.set_title('Forward Appetitive -- KC->MBON Weight Change (PAM Sugar)',
                  fontweight='bold')
    ax2.set_yticks(range(n_mbon))
    ax2.set_yticklabels(ylabels, fontsize=6)
    for ticklabel, color in zip(ax2.get_yticklabels(), ycolors):
        ticklabel.set_color(color)
    fig2.tight_layout()
    fig2.savefig('results/v4h_dw_heatmap_forward.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("  Saved: results/v4h_dw_heatmap_forward.png")

    # ====================================================================
    # FIGURE 3 -- Backward conditioning weight-change heatmap
    # ====================================================================
    fig3, ax3 = plt.subplots(figsize=(12, 9))
    im3 = ax3.imshow(dW_bwd[:, odor_kcs_sorted], aspect='auto', cmap='RdBu_r',
                     vmin=-vmax, vmax=vmax, interpolation='nearest')
    plt.colorbar(im3, ax=ax3, label='dW (weight change)')
    ax3.set_xlabel(f'Odor-active KCs ({len(odor_kcs_sorted)} of {n_kc})')
    ax3.set_ylabel('MBON')
    ax3.set_title('Backward Appetitive -- KC->MBON Weight Change (PAM Sugar)',
                  fontweight='bold')
    ax3.set_yticks(range(n_mbon))
    ax3.set_yticklabels(ylabels, fontsize=6)
    for ticklabel, color in zip(ax3.get_yticklabels(), ycolors):
        ticklabel.set_color(color)
    fig3.tight_layout()
    fig3.savefig('results/v4h_dw_heatmap_backward.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print("  Saved: results/v4h_dw_heatmap_backward.png")

    # ====================================================================
    # FIGURE 4 -- Forward conditioning per-MBON bar chart
    # ====================================================================
    mean_dW_fwd = dW_fwd[:, odor_kcs_sorted].mean(axis=1)
    mean_dW_bwd = dW_bwd[:, odor_kcs_sorted].mean(axis=1)
    x_mbon = np.arange(n_mbon)

    fig4, ax4 = plt.subplots(figsize=(16, 5))
    ax4.bar(x_mbon, mean_dW_fwd, color='darkorange', alpha=0.8)
    ax4.axhline(0, color='k', lw=0.8)
    for i in approach_idx:
        ax4.axvspan(i - 0.5, i + 0.5, alpha=0.08, color='green')
    for i in avoid_idx:
        ax4.axvspan(i - 0.5, i + 0.5, alpha=0.08, color='red')
    ax4.set_xticks(x_mbon)
    ax4.set_xticklabels(mbon_labels, rotation=45, ha='right', fontsize=7)
    ax4.set_ylabel('Mean dW (over odor-active KCs)')
    ax4.set_title('Forward Appetitive -- Per-MBON Mean Weight Change (PAM Sugar)',
                  fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    fig4.tight_layout()
    fig4.savefig('results/v4h_dw_barplot_forward.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print("  Saved: results/v4h_dw_barplot_forward.png")

    # ====================================================================
    # FIGURE 5 -- Backward conditioning per-MBON bar chart
    # ====================================================================
    fig5, ax5 = plt.subplots(figsize=(16, 5))
    ax5.bar(x_mbon, mean_dW_bwd, color='peru', alpha=0.8)
    ax5.axhline(0, color='k', lw=0.8)
    for i in approach_idx:
        ax5.axvspan(i - 0.5, i + 0.5, alpha=0.08, color='green')
    for i in avoid_idx:
        ax5.axvspan(i - 0.5, i + 0.5, alpha=0.08, color='red')
    ax5.set_xticks(x_mbon)
    ax5.set_xticklabels(mbon_labels, rotation=45, ha='right', fontsize=7)
    ax5.set_ylabel('Mean dW (over odor-active KCs)')
    ax5.set_title('Backward Appetitive -- Per-MBON Mean Weight Change (PAM Sugar)',
                  fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    fig5.tight_layout()
    fig5.savefig('results/v4h_dw_barplot_backward.png', dpi=150, bbox_inches='tight')
    plt.close(fig5)
    print("  Saved: results/v4h_dw_barplot_backward.png")

    # ====================================================================
    # FIGURE 6 -- Training dynamics (5x2 grid)
    # ====================================================================
    fig6, axes6 = plt.subplots(5, 2, figsize=(16, 14), sharex='col')

    for col, (rec, cond_label, cs_on, cs_off, us_on, us_off) in enumerate([
        (rec_fwd, 'Forward', fwd_cs_on, fwd_cs_off, fwd_us_on, fwd_us_off),
        (rec_bwd, 'Backward', bwd_cs_on, bwd_cs_off, bwd_us_on, bwd_us_off),
    ]):
        t = rec['t']

        ax = axes6[0, col]
        ax.axvspan(cs_on, cs_off, alpha=0.3, color='dodgerblue', label='CS (odor)')
        ax.axvspan(us_on, us_off, alpha=0.3, color='orange', label='US (PAM sugar)')
        ax.set_ylim(0, 1); ax.set_yticks([])
        ax.set_title(f'{cond_label} -- Appetitive (PAM Sugar)',
                     fontweight='bold', fontsize=12)
        ax.legend(fontsize=8, loc='upper right')

        ax = axes6[1, col]
        ax.plot(t, rec['mean_kc_r'], color='green', lw=1.5)
        ax.set_ylabel('KC rate (Hz)', fontsize=9); ax.grid(alpha=0.25)

        ax = axes6[2, col]
        ax.plot(t, rec['mean_da'], color='purple', lw=1.5)
        ax.set_ylabel('DA to MBONs', fontsize=9); ax.grid(alpha=0.25)

        ax = axes6[3, col]
        ax.plot(t, rec['mean_d_up'], color='orange', lw=1.5, label='D_up (t=200ms)')
        ax.plot(t, rec['mean_d_down'], color='darkred', lw=1.5, label='D_down (t=2000ms)')
        ax.set_ylabel('DA filter', fontsize=9)
        ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.25)

        ax = axes6[4, col]
        ax.plot(t, rec['mean_delta'], color='navy', lw=1.5,
                label='d = D_down - D_up (slow - fast)')
        ax_tw = ax.twinx()
        ax_tw.plot(t, rec['cum_mean_dw'], color='crimson', lw=1.5, ls='--',
                   label='Cum. mean dW')
        ax.set_ylabel('delta', fontsize=9, color='navy')
        ax_tw.set_ylabel('Cum. dW', fontsize=9, color='crimson')
        ax.set_xlabel('Time (ms)')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_tw.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
        ax.grid(alpha=0.25)

    fig6.suptitle('Training Dynamics -- Appetitive Conditioning (PAM Sugar Reward, Trial 1)\n'
                  'Gkanias DPR with Bennett KC-gating',
                  fontweight='bold', fontsize=12)
    fig6.tight_layout()
    fig6.savefig('results/v4h_dynamics.png', dpi=150, bbox_inches='tight')
    plt.close(fig6)
    print("  Saved: results/v4h_dynamics.png")

    # ====================================================================
    # FIGURE 7 -- Preference summary bar chart
    # ====================================================================
    fig7, ax7 = plt.subplots(figsize=(8, 5))
    bar_x = np.arange(3)
    width7 = 0.35
    cond_names = ['Naive', 'Forward', 'Backward']
    colors = ['gray', 'darkorange', 'peru']

    ax7.bar(bar_x - width7/2, app_vals, width7, label='Approach',
            color=colors, alpha=0.85, edgecolor='k', linewidth=0.5)
    ax7.bar(bar_x + width7/2, avd_vals, width7, label='Avoid',
            color=colors, alpha=0.45, edgecolor='k', linewidth=0.5, hatch='///')
    ax7.set_xticks(bar_x)
    ax7.set_xticklabels(cond_names, fontsize=11)
    ax7.set_ylabel('Mean firing rate during CS (Hz)')
    ax7.set_title('Appetitive (PAM Sugar) -- Preference Summary\n'
                  '(solid = approach, hatched = avoid)', fontweight='bold')
    ax7.legend(fontsize=10)
    ax7.grid(axis='y', alpha=0.3)
    fig7.tight_layout()
    fig7.savefig('results/v4h_preference.png', dpi=150, bbox_inches='tight')
    plt.close(fig7)
    print("  Saved: results/v4h_preference.png")

    # ====================================================================
    # FIGURE 8 -- Side-by-side FC vs BC heatmaps
    # ====================================================================
    fig8, axes8 = plt.subplots(1, 2, figsize=(16, 9))

    for ax, dW, title in zip(
        axes8,
        [dW_fwd, dW_bwd],
        ['Forward Appetitive', 'Backward Appetitive'],
    ):
        im = ax.imshow(dW[:, odor_kcs_sorted], aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, interpolation='nearest')
        plt.colorbar(im, ax=ax, label='dW (weight change)')
        ax.set_xlabel(f'Odor-active KCs ({len(odor_kcs_sorted)} of {n_kc})')
        ax.set_ylabel('MBON')
        ax.set_title(f'KC->MBON dW -- {title}', fontweight='bold')
        ax.set_yticks(range(n_mbon))
        ax.set_yticklabels(ylabels, fontsize=6)
        for ticklabel, color in zip(ax.get_yticklabels(), ycolors):
            ticklabel.set_color(color)

    fig8.suptitle(f'KC->MBON Synaptic Weight Changes After {n_trials} Trials -- '
                  'Appetitive (PAM Sugar)\n'
                  'Gkanias DPR Model (green = approach, red = avoid)',
                  fontweight='bold', fontsize=12)
    fig8.tight_layout()
    fig8.savefig('results/v4h_weight_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close(fig8)
    print("  Saved: results/v4h_weight_heatmaps.png")

    # ====================================================================
    # Numerical summary
    # ====================================================================
    print("\n" + "-" * 50)
    print("APPETITIVE (PAM SUGAR) MODEL SUMMARY")
    print("-" * 50)
    print(f"  Approach MBONs during CS (Hz):")
    print(f"    Naive:    {app_vals[0]:.2f}")
    print(f"    Forward:  {app_vals[1]:.2f}  (d = {app_vals[1]-app_vals[0]:+.2f})")
    print(f"    Backward: {app_vals[2]:.2f}  (d = {app_vals[2]-app_vals[0]:+.2f})")
    print(f"  Avoid MBONs during CS (Hz):")
    print(f"    Naive:    {avd_vals[0]:.2f}")
    print(f"    Forward:  {avd_vals[1]:.2f}  (d = {avd_vals[1]-avd_vals[0]:+.2f})")
    print(f"    Backward: {avd_vals[2]:.2f}  (d = {avd_vals[2]-avd_vals[0]:+.2f})")

    expected_app = (app_vals[1] >= app_vals[0]) and (avd_vals[1] < avd_vals[0])
    print(f"\n  Forward conditioning check (appetitive):")
    print(f"    Approach potentiated from naive? {app_vals[1] >= app_vals[0]}")
    print(f"    Avoid depressed from naive?      {avd_vals[1] < avd_vals[0]}")
    if expected_app:
        print("    --> CORRECT: appetitive learning produces approach potentiation "
              "+ avoid depression")
    else:
        print("    --> UNEXPECTED: check model parameters or connectivity")

    print("\n" + "=" * 65)
    print("V4h Complete.")
    print("=" * 65)


if __name__ == '__main__':
    run_v4h()
