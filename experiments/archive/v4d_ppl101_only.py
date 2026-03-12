"""V4d: Forward vs Backward Conditioning — PPL101 (γ1pedc) only.

Same as v4c but activates ONLY PPL101 (index 34) during shock, testing
whether compartment-specific DA delivery produces selective depression of
γ1pedc-innervated MBONs rather than broad depression of all MBONs.

PPL101 is identified as PPL1-γ1pedc in the literature (Aso et al. 2014):
the single DAN per hemisphere that innervates the γ1pedc MB compartment
and serves as the canonical aversive teaching signal for that compartment.
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


def run_v4d():
    print("=" * 65)
    print("V4d: Forward vs Backward — PPL101 (γ1pedc) Only")
    print("=" * 65)

    # -- Config --------------------------------------------------------------
    with open('experiments/configs/config.yaml') as f:
        config = yaml.safe_load(f)
    config['dt'] = 1.0
    config['learning_rate'] = 5e-6

    # -- Networks ------------------------------------------------------------
    net_fwd   = MushroomBodyNetwork(config)
    net_bwd   = MushroomBodyNetwork(config)
    net_naive = MushroomBodyNetwork(config)

    for net in (net_fwd, net_bwd, net_naive):
        net.W_KC_DAN[:] = 0.0
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

    # -- PPL101 identification -----------------------------------------------
    # PPL101 = index 34 in the DAN weight matrix (γ1pedc compartment DAN)
    ppl101_index = 34
    ppl101_stim = np.zeros(n_dan)
    ppl101_stim[ppl101_index] = 80.0
    print(f"Activating ONLY PPL101 (index {ppl101_index}) during US")

    # Also load all PPL indices for comparison reporting
    all_ppl = load_ppl_indices(
        'data/connectomes/processed/mb_circuit_right_dan_annotations.csv',
        'data/connectomes/processed/mb_circuit_right_ids.npz',
    )
    print(f"(All PPL indices for reference: {all_ppl})")

    # -- Check PPL101 → MBON connectivity ------------------------------------
    ppl101_to_mbon = net_fwd.W_DAN_MBON[:, ppl101_index]
    mbon_labels = load_mbon_labels(
        'data/connectomes/processed/mb_circuit_right_mbon_annotations.csv',
        'data/connectomes/processed/mb_circuit_right_ids.npz',
    )
    mbon_groups = classify_mbons()
    approach_idx = mbon_groups['approach']
    avoid_idx    = mbon_groups['avoid']

    print(f"\nPPL101 → MBON connectivity (W_DAN_MBON[:, 34]):")
    print(f"  Nonzero targets: {np.count_nonzero(ppl101_to_mbon)} / {n_mbon} MBONs")
    print(f"  Approach MBONs mean weight: {ppl101_to_mbon[approach_idx].mean():.6f}")
    print(f"  Avoid MBONs mean weight:    {ppl101_to_mbon[avoid_idx].mean():.6f}")
    top_targets = np.argsort(ppl101_to_mbon)[::-1][:10]
    print(f"  Top 10 targets:")
    for idx in top_targets:
        if ppl101_to_mbon[idx] > 0:
            group = ('APP' if idx in approach_idx else
                     'AVD' if idx in avoid_idx else 'OTH')
            print(f"    MBON {idx:2d} ({mbon_labels[idx]:20s}) [{group}]: "
                  f"{ppl101_to_mbon[idx]:.6f}")

    # -- Odor pattern --------------------------------------------------------
    np.random.seed(7)
    odor_pattern, odor_kcs = create_sparse_odor(n_kc, sparsity=0.05, strength=15.0)
    print(f"\nOdor: {len(odor_kcs)} KCs active ({100*len(odor_kcs)/n_kc:.1f}%)")

    # -- Trial parameters (same as v4c) --------------------------------------
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

    # -- Record trial 1, then train remaining --------------------------------
    print("  Recording trial 1 dynamics ...", end=' ', flush=True)
    rec_fwd = record_training_trial(
        net_fwd, odor_pattern, ppl101_stim, dt, trial_duration,
        fwd_cs_on, fwd_cs_off, fwd_us_on, fwd_us_off, odor_kcs)
    rec_bwd = record_training_trial(
        net_bwd, odor_pattern, ppl101_stim, dt, trial_duration,
        bwd_cs_on, bwd_cs_off, bwd_us_on, bwd_us_off, odor_kcs)
    print("done")

    for trial in range(1, n_trials):
        print(f"  Trial {trial + 1}/{n_trials} ...", end=' ', flush=True)
        train_trial(net_fwd, odor_pattern, ppl101_stim, dt, trial_duration,
                    fwd_cs_on, fwd_cs_off, fwd_us_on, fwd_us_off)
        train_trial(net_bwd, odor_pattern, ppl101_stim, dt, trial_duration,
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

    print(f"\nForward  ΔW: mean={dW_fwd.mean():.6f}  "
          f"min={dW_fwd.min():.6f}  max={dW_fwd.max():.6f}")
    print(f"Backward ΔW: mean={dW_bwd.mean():.6f}  "
          f"min={dW_bwd.min():.6f}  max={dW_bwd.max():.6f}")

    # -- Per-group weight change analysis ------------------------------------
    odor_kcs_sorted = np.sort(odor_kcs)
    for label, idx in [('Approach', approach_idx), ('Avoid', avoid_idx)]:
        dw_f = dW_fwd[np.ix_(idx, odor_kcs_sorted)].mean()
        dw_b = dW_bwd[np.ix_(idx, odor_kcs_sorted)].mean()
        print(f"  {label:8s} MBONs — Fwd ΔW: {dw_f:.6f}, Bwd ΔW: {dw_b:.6f}")

    # ========================================================================
    # FIGURE 1 — Approach vs Avoid rates (PPL101 only)
    # ========================================================================
    t_vec = test_fwd['t']

    fig1, (ax_app, ax_avd) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for ax, idx, group_name in [
        (ax_app, approach_idx, 'Approach MBONs (glutamatergic)'),
        (ax_avd, avoid_idx,   'Avoid MBONs (GABAergic)'),
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
    fig1.suptitle('PPL101 (γ1pedc) Only — Approach vs Avoid MBON Responses\n'
                  'Single-DAN activation: does compartment specificity emerge?',
                  fontweight='bold', fontsize=11)
    fig1.tight_layout()
    fig1.savefig('results/v4d_ppl101_approach_avoid.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: results/v4d_ppl101_approach_avoid.png")

    # ========================================================================
    # FIGURE 2 — Per-MBON weight change (PPL101 only vs all-PPL comparison)
    # ========================================================================
    mean_dW_fwd = dW_fwd[:, odor_kcs_sorted].mean(axis=1)
    mean_dW_bwd = dW_bwd[:, odor_kcs_sorted].mean(axis=1)

    x = np.arange(n_mbon)
    width = 0.4

    fig2, ax2 = plt.subplots(figsize=(16, 5))
    ax2.bar(x - width/2, mean_dW_fwd, width, label='Forward (PPL101 only)',
            color='steelblue', alpha=0.8)
    ax2.bar(x + width/2, mean_dW_bwd, width, label='Backward (PPL101 only)',
            color='firebrick', alpha=0.8)
    ax2.axhline(0, color='k', lw=0.8)

    # Mark approach/avoid MBONs
    for i in approach_idx:
        ax2.axvspan(i - 0.5, i + 0.5, alpha=0.08, color='green')
    for i in avoid_idx:
        ax2.axvspan(i - 0.5, i + 0.5, alpha=0.08, color='red')

    ax2.set_xticks(x)
    ax2.set_xticklabels(mbon_labels, rotation=45, ha='right', fontsize=7)
    ax2.set_ylabel('Mean ΔW (over odor-active KCs)', fontsize=10)
    ax2.set_title('Per-MBON Weight Change: PPL101 (γ1pedc) Only\n'
                  'Green shading = approach MBONs, Red shading = avoid MBONs',
                  fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    fig2.tight_layout()
    fig2.savefig('results/v4d_ppl101_dw_per_mbon.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/v4d_ppl101_dw_per_mbon.png")

    # ========================================================================
    # FIGURE 3 — Training dynamics (PPL101 only)
    # ========================================================================
    fig3, axes3 = plt.subplots(5, 2, figsize=(16, 14), sharex='col')

    for col, (rec, cond_label, cs_on, cs_off, us_on, us_off) in enumerate([
        (rec_fwd, 'Forward', fwd_cs_on, fwd_cs_off, fwd_us_on, fwd_us_off),
        (rec_bwd, 'Backward', bwd_cs_on, bwd_cs_off, bwd_us_on, bwd_us_off),
    ]):
        t = rec['t']

        ax = axes3[0, col]
        ax.axvspan(cs_on, cs_off, alpha=0.3, color='dodgerblue', label='CS (odor)')
        ax.axvspan(us_on, us_off, alpha=0.3, color='red', label='US (PPL101)')
        ax.set_ylim(0, 1); ax.set_yticks([])
        ax.set_title(f'{cond_label} — PPL101 Only', fontweight='bold', fontsize=12)
        ax.legend(fontsize=8, loc='upper right')

        ax = axes3[1, col]
        ax.plot(t, rec['mean_kc_r'], color='green', lw=1.5)
        ax.set_ylabel('KC rate (Hz)', fontsize=9); ax.grid(alpha=0.25)

        ax = axes3[2, col]
        ax.plot(t, rec['mean_da'], color='purple', lw=1.5)
        ax.set_ylabel('DA to MBONs', fontsize=9); ax.grid(alpha=0.25)

        ax = axes3[3, col]
        ax.plot(t, rec['mean_d_up'], color='orange', lw=1.5, label='D_up (τ=200ms)')
        ax.plot(t, rec['mean_d_down'], color='darkred', lw=1.5, label='D_down (τ=2000ms)')
        ax.set_ylabel('DA filter', fontsize=9)
        ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.25)

        ax = axes3[4, col]
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

    fig3.suptitle('Training Dynamics — Trial 1 (PPL101 / γ1pedc Only)',
                  fontweight='bold', fontsize=12)
    fig3.tight_layout()
    fig3.savefig('results/v4d_ppl101_training_dynamics.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/v4d_ppl101_training_dynamics.png")

    # ========================================================================
    # FIGURE 4 — Preference summary bar chart
    # ========================================================================
    cs_mask = (t_vec >= test_cs_on) & (t_vec < test_cs_off)

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    bar_x = np.arange(3)
    width4 = 0.35
    cond_names = ['Naive', 'Forward', 'Backward']
    colors = ['gray', 'steelblue', 'firebrick']

    app_vals, avd_vals = [], []
    for cond_name, mbon_r in [('Naive', test_naive['mbon_r']),
                               ('Forward', test_fwd['mbon_r']),
                               ('Backward', test_bwd['mbon_r'])]:
        app_vals.append(mbon_r[cs_mask][:, approach_idx].mean())
        avd_vals.append(mbon_r[cs_mask][:, avoid_idx].mean())

    ax4.bar(bar_x - width4/2, app_vals, width4, label='Approach',
            color=colors, alpha=0.85, edgecolor='k', linewidth=0.5)
    ax4.bar(bar_x + width4/2, avd_vals, width4, label='Avoid',
            color=colors, alpha=0.45, edgecolor='k', linewidth=0.5, hatch='///')
    ax4.set_xticks(bar_x)
    ax4.set_xticklabels(cond_names, fontsize=11)
    ax4.set_ylabel('Mean firing rate during CS (Hz)', fontsize=10)
    ax4.set_title('PPL101 (γ1pedc) Only — Preference Summary\n'
                  '(solid = approach, hatched = avoid)', fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    fig4.tight_layout()
    fig4.savefig('results/v4d_ppl101_preference_summary.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/v4d_ppl101_preference_summary.png")

    plt.show()
    print("\n" + "=" * 65)
    print("V4d Complete.")
    print("=" * 65)


if __name__ == '__main__':
    run_v4d()
