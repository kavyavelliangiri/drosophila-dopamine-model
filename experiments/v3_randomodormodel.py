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
            I_KC = odor_pattern if 500 <= t < 1000 else np.zeros(net.KCs.n)
            
            # Shock (1500-1600ms)
            I_DAN = np.full(net.DANs.n, 80.0) if 1500 <= t < 1600 else np.zeros(net.DANs.n)
            
            net.step(I_KC, I_DAN)
    
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
    plt.savefig('results/v3_full_model.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved: results/v3_full_model.png")
    plt.show()
    
    print("\n" + "="*60)
    print("V3 Complete! Ready for research.")
    print("="*60)


if __name__ == '__main__':
    run_v3()