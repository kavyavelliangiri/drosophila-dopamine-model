"""V3: Full-scale model.

Research question: Are biological weights optimal?
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
from mbmodel.models import MushroomBodyNetwork
from mbmodel.stimuli import create_sparse_odor


def run_v3():
    """Run full MB model."""
    
    print("="*60)
    print("V3: Full Mushroom Body Model")
    print("="*60)
    
    # Load config
    with open('experiments/configs/config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Create network
    net = MushroomBodyNetwork(config)
    
    print(f"\nNetwork size:")
    print(f"  {net.KCs.n} KCs -> {net.MBONs.n} MBONs")
    print(f"  {net.DANs.n} DANs")
    print(f"  Weights: {net.W_KC_MBON.shape}")
    print(f"  Initial mean weight: {np.mean(net.W_KC_MBON):.3f}\n")
    
    # Training
    n_trials = 5
    trial_duration = 3000
    dt = config['dt']
    
    print(f"Training: {n_trials} trials\n")
    
    # Fixed odor pattern (same KCs each trial)
    np.random.seed(42)
    odor_pattern, odor_kcs = create_sparse_odor(
        net.KCs.n, sparsity=0.1, strength=50.0)
    
    for trial in range(n_trials):
        print(f"Trial {trial+1}/{n_trials}...")
        
        net.reset_activity()
        n_steps = int(trial_duration / dt)
        
        for step in range(n_steps):
            t = step * dt
            
            # Odor (500-1000ms)
            I_odor = odor_pattern if 500 <= t < 1000 else np.zeros(net.KCs.n)

            # Shock (1500-1600ms): x_punish drives DANs via w_punish
            x_punish = 1.0 if 1500 <= t < 1600 else 0.0

            net.step(I_odor, x_punish)
    
    # Analyze
    dW = net.get_weight_change()
    
    print(f"\nWeight analysis:")
    print(f"  Mean change: {np.mean(dW):.4f}")
    print(f"  Max increase: {np.max(dW):.4f}")
    print(f"  Max decrease: {np.min(dW):.4f}")
    print(f"  % changed >0.1: {100*np.mean(np.abs(dW) > 0.1):.1f}%")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Initial weights
    im0 = axes[0].imshow(net.W_initial, aspect='auto', cmap='viridis')
    axes[0].set_title('Initial (Biological)', fontweight='bold')
    axes[0].set_xlabel('KC')
    axes[0].set_ylabel('MBON')
    plt.colorbar(im0, ax=axes[0])
    
    # Final weights
    im1 = axes[1].imshow(net.W_KC_MBON, aspect='auto', cmap='viridis')
    axes[1].set_title('After Learning', fontweight='bold')
    axes[1].set_xlabel('KC')
    axes[1].set_ylabel('MBON')
    plt.colorbar(im1, ax=axes[1])
    
    # Change
    im2 = axes[2].imshow(dW, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    axes[2].set_title('Weight Change (ΔW)', fontweight='bold')
    axes[2].set_xlabel('KC')
    axes[2].set_ylabel('MBON')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('results/v3_full_model_real_connectivity.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved: results/v3_full_model_real_connectivity .png")
    plt.show()
    
    print("\n" + "="*60)
    print("V3 Complete! Ready for research.")
    print("="*60)

def run_v3_single_trial():
    """Single trial: observe MBON activity and 2nd messenger dynamics over time."""

    print("="*60)
    print("V3: Single Trial Visualization")
    print("="*60)

    with open('experiments/configs/config.yaml') as f:
        config = yaml.safe_load(f)

    net = MushroomBodyNetwork(config)

    dt = config['dt']
    trial_duration = 3000
    n_steps = int(trial_duration / dt)
    t_vec = np.arange(n_steps) * dt

    np.random.seed(42)
    approach_idx = 26  # MBON11 right hemisphere (approach, PPL1-targeted)
    avoid_idx    = 0   # MBON01 right hemisphere (avoid, PAM-target
    
    # In run_v3_single_trial(), replace create_sparse_odor with:
    w_approach = net.W_KC_MBON[approach_idx, :]  # KC inputs to MBON11
    w_avoid    = net.W_KC_MBON[avoid_idx, :]     # KC inputs to MBON01

    # Find KCs connected to both
    both_connected = np.where((w_approach > 0) & (w_avoid > 0))[0]
    print(f"KCs connected to both MBONs: {len(both_connected)}")

    # Take all shared KCs, or up to 10% whichever is smaller
    n_odor = min(len(both_connected), max(10, int(0.1 * len(both_connected))))
    odor_kcs = np.random.choice(both_connected, n_odor, replace=False)
    odor_pattern = np.zeros(net.KCs.n)
    odor_pattern[odor_kcs] = 50.0

    # --- Identify one approach and one avoid MBON ---
    # Assumes net.MBONs has a 'types' or 'labels' attribute.
    # Adjust index lookup to match your model's conventions.
    approach_idx = 26  # MBON11 right hemisphere (approach, PPL1-targeted)
    avoid_idx    = 0   # MBON01 right hemisphere (avoid, PAM-targeted)
    print(f"Using MBON11 (idx {approach_idx}) as approach, "
      f"MBON01 (idx {avoid_idx}) as avoid.")

    # --- Storage ---
    approach_r   = np.zeros(n_steps)
    avoid_r      = np.zeros(n_steps)

    # D_△ and D_▽ traces at approach MBON
    ca_trace   = np.zeros(n_steps)
    camp_trace = np.zeros(n_steps)

    # Weight change per step for the two MBONs (sum over their KC inputs)
    approach_dW = np.zeros(n_steps)
    avoid_dW    = np.zeros(n_steps)
    W_approach_prev = net.W_KC_MBON[approach_idx, :].copy()
    W_avoid_prev    = net.W_KC_MBON[avoid_idx, :].copy()

    # --- Simulate ---
    net.reset_activity()

    for step in range(n_steps):
        t = step * dt

        I_odor   = odor_pattern if 500 <= t < 1000 else np.zeros(net.KCs.n)
        x_punish = 1.0 if 1500 <= t < 1600 else 0.0

        net.step(I_odor, x_punish)

        # MBON firing rates
        approach_r[step] = net.MBONs.r[approach_idx]
        avoid_r[step]    = net.MBONs.r[avoid_idx]

        # Dopaminergic DPR components at the approach MBON
        d_up   = net.plasticity.D_up
        d_down = net.plasticity.D_down
        ca_trace[step]   = d_up[approach_idx]   if d_up   is not None else 0.0
        camp_trace[step] = d_down[approach_idx] if d_down is not None else 0.0

        # Instantaneous weight change magnitude (mean over all KCs for that MBON)
        W_approach_now = net.W_KC_MBON[approach_idx, :]
        W_avoid_now    = net.W_KC_MBON[avoid_idx, :]
        approach_dW[step] = np.mean(W_approach_now - W_approach_prev)
        avoid_dW[step]    = np.mean(W_avoid_now    - W_avoid_prev)
        W_approach_prev = W_approach_now.copy()
        W_avoid_prev    = W_avoid_now.copy()

    # ----------------------------------------------------------------
    # Figure 1: MBON activity + cumulative weight change
    # ----------------------------------------------------------------
    fig1, axes1 = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Shade stimulus periods
    for ax in axes1:
        ax.axvspan(500,  1000, alpha=0.15, color='dodgerblue',  label='Odor (CS)')
        ax.axvspan(1500, 1600, alpha=0.25, color='tomato',       label='Shock (US)')

    # Approach MBON
    axes1[0].plot(t_vec, approach_r, color='steelblue', lw=1.5)
    axes1[0].set_ylabel('Firing rate (Hz)', fontsize=10)
    axes1[0].set_title(f'Approach MBON (idx {approach_idx})', fontweight='bold')
    axes1[0].legend(loc='upper right', fontsize=8)

    # Avoid MBON
    axes1[1].plot(t_vec, avoid_r, color='firebrick', lw=1.5)
    axes1[1].set_ylabel('Firing rate (Hz)', fontsize=10)
    axes1[1].set_title(f'Avoid MBON (idx {avoid_idx})', fontweight='bold')

    # Cumulative weight change
    axes1[2].plot(t_vec, np.cumsum(approach_dW), color='steelblue',
                  lw=1.5, label='Approach MBON')
    axes1[2].plot(t_vec, np.cumsum(avoid_dW),    color='firebrick',
                  lw=1.5, label='Avoid MBON')
    axes1[2].axhline(0, color='k', lw=0.8, ls='--')
    axes1[2].set_ylabel('Cumul. ΔW (mean over KCs)', fontsize=10)
    axes1[2].set_xlabel('Time (ms)', fontsize=10)
    axes1[2].set_title('Cumulative KC→MBON Weight Change', fontweight='bold')
    axes1[2].legend(fontsize=8)

    fig1.tight_layout()
    fig1.savefig('results/v3_mbon_single_trial_real_connectivity.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/v3_mbon_single_trial_real_connectivity.png")

    # ----------------------------------------------------------------
    # Figure 2: Ca2+ and cAMP dynamics in probe KC
    # ----------------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    for ax in axes2:
        ax.axvspan(500,  1000, alpha=0.15, color='dodgerblue')
        ax.axvspan(1500, 1600, alpha=0.25, color='tomato')

    axes2[0].plot(t_vec, ca_trace, color='mediumorchid', lw=1.5)
    axes2[0].set_ylabel('D_△ (a.u.)', fontsize=10)
    axes2[0].set_title(f'MBON {approach_idx} — D_△ potentiation component (fast DA tracker)',
                       fontweight='bold')

    axes2[1].plot(t_vec, camp_trace, color='darkorange', lw=1.5)
    axes2[1].set_ylabel('D_▽ (a.u.)', fontsize=10)
    axes2[1].set_xlabel('Time (ms)', fontsize=10)
    axes2[1].set_title(f'MBON {approach_idx} — D_▽ depression component (slow DA tracker)',
                       fontweight='bold')

    fig2.tight_layout()
    fig2.savefig('results/v3_ca_camp_dynamics_real_connectivity.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/v3_ca_camp_dynamics_real_connectivity.png")

    plt.show()
    print("\n" + "="*60)
    print("Single Trial Analysis Complete.")
    print("="*60)


if __name__ == '__main__':
    # run_v3()
    run_v3_single_trial()
