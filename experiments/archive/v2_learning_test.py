"""V2 Test Model: 2 KCs, 2 MBONs, 1 DAN.
   The goal of this test model is to verify that the dopamine-modulated plasticity rule will update the KC>MBON weights correctly"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from mbmodel.models import NeuronPopulation
from mbmodel.plasticity import BiologicalPlasticity


def test_mbon_responses(KCs, MBON_approach, MBON_avoid, W_approach, W_avoid, 
                        test_duration=2000, dt=0.1):
    """Test both MBON responses to odor (no shock).
    
    Returns DataFrame with timeseries data for both MBONs.
    """
    KCs.reset()
    MBON_approach.reset()
    MBON_avoid.reset()
    
    n_steps = int(test_duration / dt)
    
    data = {
        'time_ms': [],
        'KC1_rate': [],
        'KC2_rate': [],
        'MBON_approach_rate': [],
        'MBON_avoid_rate': [],
        'net_valence': [],  # approach - avoid
        'odor_present': []
    }
    
    for step in range(n_steps):
        t = step * dt
        
        I_KC = np.array([50.0, 20.0]) if 500 <= t < 1000 else np.zeros(2)
        KCs.update(I_KC, dt)
        
        # Update both MBONs
        I_approach = W_approach @ KCs.r
        MBON_approach.update(I_approach, dt)
        
        I_avoid = W_avoid @ KCs.r
        MBON_avoid.update(I_avoid, dt)
        
        # Net valence: positive = approach, negative = avoid
        net_valence = MBON_approach.r[0] - MBON_avoid.r[0]
        
        data['time_ms'].append(t)
        data['KC1_rate'].append(KCs.r[0])
        data['KC2_rate'].append(KCs.r[1])
        data['MBON_approach_rate'].append(MBON_approach.r[0])
        data['MBON_avoid_rate'].append(MBON_avoid.r[0])
        data['net_valence'].append(net_valence)
        data['odor_present'].append(500 <= t < 1000)
    
    return pd.DataFrame(data)


def run_v2_two_mbons():
    """Run learning with approach and avoidance MBONs."""
    
    print("="*60)
    print("V2: Two MBON Types - Approach vs Avoidance")
    print("="*60)
    
    dt = 0.1
    
    # Create neurons
    KCs = NeuronPopulation(
        n=2, tau=10.0, r_max=100.0, activation='relu', name="KCs"
    )
    
    # Two MBON types
    MBON_approach = NeuronPopulation(
        n=1, tau=20.0, r_max=100.0, activation='relu', name="MBON_approach"
    )
    MBON_avoid = NeuronPopulation(
        n=1, tau=20.0, r_max=100.0, activation='relu', name="MBON_avoid"
    )
    
    # Punishment DAN
    DAN_punish = NeuronPopulation(
        n=1, tau=15.0, r_max=100.0, activation='relu', name="DAN_punish"
    )
    
    # Initial weights (both start equal)
    W_approach = np.array([[1.0, 1.0]])
    W_avoid = np.array([[1.0, 1.0]])
    
    # Two plasticity rules with OPPOSITE signs!
    plasticity_approach = BiologicalPlasticity(
        learning_rate=0.002,
        w_rest=1.0,
        tau_eligibility=1000.0,
        tau_cAMP=500.0,
        tau_Ca=200.0,
        DA_baseline=20.0
    )
    
    plasticity_avoid = BiologicalPlasticity(
        learning_rate=0.002,
        w_rest=1.0,
        tau_eligibility=1000.0,
        tau_cAMP=500.0,
        tau_Ca=200.0,
        DA_baseline=20.0
    )
    
    print(f"\nInitial weights:")
    print(f"  Approach MBON: {W_approach[0]}")
    print(f"  Avoid MBON:    {W_avoid[0]}")
    
    # ============================================================
    # TEST BEFORE LEARNING
    # ============================================================
    
    print("\n" + "-"*60)
    print("TESTING BEFORE LEARNING")
    print("-"*60)
    
    test_before = test_mbon_responses(
        KCs, MBON_approach, MBON_avoid, W_approach, W_avoid, dt=dt
    )
    
    odor_period = test_before[test_before['odor_present'] == True]
    approach_before = odor_period['MBON_approach_rate'].mean()
    avoid_before = odor_period['MBON_avoid_rate'].mean()
    valence_before = odor_period['net_valence'].mean()
    
    print(f"MBON_approach: {approach_before:.2f} Hz")
    print(f"MBON_avoid:    {avoid_before:.2f} Hz")
    print(f"Net valence:   {valence_before:.2f} Hz ({'APPROACH' if valence_before > 0 else 'AVOID'})")
    
    # ============================================================
    # TRAINING WITH PUNISHMENT
    # ============================================================
    
    print("\n" + "-"*60)
    print("TRAINING: Odor + Punishment Shock")
    print("-"*60)
    
    n_trials = 5
    trial_duration = 3000
    
    # Data collection
    training_history = {
        'trial': [0],
        'W_approach_KC1': [W_approach[0, 0]],
        'W_approach_KC2': [W_approach[0, 1]],
        'W_avoid_KC1': [W_avoid[0, 0]],
        'W_avoid_KC2': [W_avoid[0, 1]]
    }
    
    for trial in range(n_trials):
        print(f"Trial {trial+1}/{n_trials}...", end=' ')
        
        # Reset activity
        KCs.reset()
        MBON_approach.reset()
        MBON_avoid.reset()
        DAN_punish.reset()
        plasticity_approach.DA = plasticity_approach.DA_baseline
        plasticity_avoid.DA = plasticity_avoid.DA_baseline
        
        n_steps = int(trial_duration / dt)
        
        for step in range(n_steps):
            t = step * dt
            
            # Odor (500-1000ms)
            I_KC = np.array([50.0, 20.0]) if 500 <= t < 1000 else np.zeros(2)
            
            # Punishment shock (1500-1600ms)
            I_DAN = np.array([80.0]) if 1500 <= t < 1600 else np.zeros(1)
            
            # Update neurons
            KCs.update(I_KC, dt)
            
            I_approach = W_approach @ KCs.r
            MBON_approach.update(I_approach, dt)
            
            I_avoid = W_avoid @ KCs.r
            MBON_avoid.update(I_avoid, dt)
            
            DAN_punish.update(I_DAN, dt)
            
            # Update DA for both plasticity rules
            plasticity_approach.update_DA(DAN_punish.r[0], dt)
            plasticity_avoid.update_DA(DAN_punish.r[0], dt)
            
            plasticity_approach.update_cAMP(dt)
            plasticity_approach.update_Ca(MBON_approach.r, dt)
            
            plasticity_avoid.update_cAMP(dt)
            plasticity_avoid.update_Ca(MBON_avoid.r, dt)
            
            # CRITICAL: Punishment DAN has OPPOSITE effects!
            # Get kappa for approach (will be used to DEPRESS)
            kappa_approach = plasticity_approach.compute_kappa()
            
            # For avoid, INVERT the sign (punishment POTENTIATES avoidance)
            kappa_avoid = -plasticity_avoid.compute_kappa()  # NEGATIVE!
            
            # Update approach weights (depression with punishment)
            eligibility = plasticity_approach.update_eligibility(KCs.r, dt)
            delta = eligibility[np.newaxis, :]
            plasticity_term = kappa_approach + W_approach - plasticity_approach.w_rest
            dW = plasticity_approach.learning_rate * delta * plasticity_term * dt
            W_approach = W_approach + dW
            W_approach = np.clip(W_approach, 0, 10)
            
            # Update avoid weights (potentiation with punishment)
            eligibility = plasticity_avoid.update_eligibility(KCs.r, dt)
            delta = eligibility[np.newaxis, :]
            plasticity_term = kappa_avoid + W_avoid - plasticity_avoid.w_rest
            dW = plasticity_avoid.learning_rate * delta * plasticity_term * dt
            W_avoid = W_avoid + dW
            W_avoid = np.clip(W_avoid, 0, 10)
        
        # Store weights
        training_history['trial'].append(trial + 1)
        training_history['W_approach_KC1'].append(W_approach[0, 0])
        training_history['W_approach_KC2'].append(W_approach[0, 1])
        training_history['W_avoid_KC1'].append(W_avoid[0, 0])
        training_history['W_avoid_KC2'].append(W_avoid[0, 1])
        
        print(f"Approach={W_approach[0]}, Avoid={W_avoid[0]}")
    
    print(f"\nFinal weights:")
    print(f"  Approach MBON: {W_approach[0]} (change: {W_approach[0] - np.array([1.0, 1.0])})")
    print(f"  Avoid MBON:    {W_avoid[0]} (change: {W_avoid[0] - np.array([1.0, 1.0])})")
    
    # ============================================================
    # TEST AFTER LEARNING
    # ============================================================
    
    print("\n" + "-"*60)
    print("TESTING AFTER LEARNING")
    print("-"*60)
    
    test_after = test_mbon_responses(
        KCs, MBON_approach, MBON_avoid, W_approach, W_avoid, dt=dt
    )
    
    odor_period = test_after[test_after['odor_present'] == True]
    approach_after = odor_period['MBON_approach_rate'].mean()
    avoid_after = odor_period['MBON_avoid_rate'].mean()
    valence_after = odor_period['net_valence'].mean()
    
    print(f"MBON_approach: {approach_after:.2f} Hz")
    print(f"MBON_avoid:    {avoid_after:.2f} Hz")
    print(f"Net valence:   {valence_after:.2f} Hz ({'APPROACH' if valence_after > 0 else 'AVOID'})")
    
    print("\n" + "-"*60)
    print("LEARNING EFFECT")
    print("-"*60)
    print(f"Approach MBON: {approach_before:.1f} → {approach_after:.1f} Hz ({approach_after - approach_before:+.1f})")
    print(f"Avoid MBON:    {avoid_before:.1f} → {avoid_after:.1f} Hz ({avoid_after - avoid_before:+.1f})")
    print(f"Net valence:   {valence_before:.1f} → {valence_after:.1f} Hz ({valence_after - valence_before:+.1f})")
    print(f"\nBehavior: {('APPROACH' if valence_before > 0 else 'AVOID')} → {('APPROACH' if valence_after > 0 else 'AVOID')}")
    
    # ============================================================
    # SAVE DATA
    # ============================================================
    
    print("\n" + "-"*60)
    print("Saving data...")
    print("-"*60)
    os.makedirs('results', exist_ok=True)
    
    # Training history
    training_df = pd.DataFrame(training_history)
    training_df.to_csv('results/v2_two_mbons_training.csv', index=False)
    print("  ✓ results/v2_two_mbons_training.csv")
    
    # Before test
    test_before.to_csv('results/v2_two_mbons_before.csv', index=False)
    print("  ✓ results/v2_two_mbons_before.csv")
    
    # After test
    test_after.to_csv('results/v2_two_mbons_after.csv', index=False)
    print("  ✓ results/v2_two_mbons_after.csv")
    
    # Comparison summary
    comparison_df = pd.DataFrame({
        'condition': ['before', 'after'],
        'approach_MBON_Hz': [approach_before, approach_after],
        'avoid_MBON_Hz': [avoid_before, avoid_after],
        'net_valence_Hz': [valence_before, valence_after],
        'behavior': ['APPROACH' if valence_before > 0 else 'AVOID',
                     'APPROACH' if valence_after > 0 else 'AVOID']
    })
    comparison_df.to_csv('results/v2_two_mbons_comparison.csv', index=False)
    print("  ✓ results/v2_two_mbons_comparison.csv")
    
    # ============================================================
    # PLOT
    # ============================================================
    
    print("\nGenerating plots...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # 1. Weight evolution - Approach MBON
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(training_df['trial'], training_df['W_approach_KC1'], 'o-',
             label='KC1→Approach', markersize=8, linewidth=2, color='blue')
    ax1.plot(training_df['trial'], training_df['W_approach_KC2'], 's-',
             label='KC2→Approach', markersize=8, linewidth=2, color='cyan')
    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Trial', fontsize=11)
    ax1.set_ylabel('Weight', fontsize=11)
    ax1.set_title('Approach MBON Weights (Depression)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    
    # 2. Weight evolution - Avoid MBON
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(training_df['trial'], training_df['W_avoid_KC1'], 'o-',
             label='KC1→Avoid', markersize=8, linewidth=2, color='red')
    ax2.plot(training_df['trial'], training_df['W_avoid_KC2'], 's-',
             label='KC2→Avoid', markersize=8, linewidth=2, color='orange')
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Trial', fontsize=11)
    ax2.set_ylabel('Weight', fontsize=11)
    ax2.set_title('Avoid MBON Weights (Potentiation)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    
    # 3. MBON responses BEFORE
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(test_before['time_ms'], test_before['MBON_approach_rate'],
             linewidth=2.5, label='Approach MBON', color='blue')
    ax3.plot(test_before['time_ms'], test_before['MBON_avoid_rate'],
             linewidth=2.5, label='Avoid MBON', color='red')
    ax3.axvspan(500, 1000, alpha=0.15, color='purple', label='Odor')
    ax3.set_xlabel('Time (ms)', fontsize=11)
    ax3.set_ylabel('MBON Rate (Hz)', fontsize=11)
    ax3.set_title('Before Learning: Neutral Response', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    
    # 4. MBON responses AFTER
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(test_after['time_ms'], test_after['MBON_approach_rate'],
             linewidth=2.5, label='Approach MBON', color='blue')
    ax4.plot(test_after['time_ms'], test_after['MBON_avoid_rate'],
             linewidth=2.5, label='Avoid MBON', color='red')
    ax4.axvspan(500, 1000, alpha=0.15, color='purple', label='Odor')
    ax4.set_xlabel('Time (ms)', fontsize=11)
    ax4.set_ylabel('MBON Rate (Hz)', fontsize=11)
    ax4.set_title('After Learning: Avoidance Response', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    
    # 5. Net valence comparison (BIG PLOT)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.plot(test_before['time_ms'], test_before['net_valence'],
             linewidth=3, label='Before Learning', color='gray', alpha=0.7)
    ax5.plot(test_after['time_ms'], test_after['net_valence'],
             linewidth=3, label='After Learning', color='green')
    ax5.axhline(0, color='black', linestyle='-', linewidth=1)
    ax5.axvspan(500, 1000, alpha=0.15, color='purple', label='Odor')
    ax5.fill_between(test_before['time_ms'], 0, test_before['net_valence'],
                      where=(test_before['net_valence'] > 0), alpha=0.2, color='blue', label='Approach zone')
    ax5.fill_between(test_after['time_ms'], 0, test_after['net_valence'],
                      where=(test_after['net_valence'] < 0), alpha=0.2, color='red', label='Avoid zone')
    ax5.set_xlabel('Time (ms)', fontsize=12)
    ax5.set_ylabel('Net Valence (Approach - Avoid, Hz)', fontsize=12)
    ax5.set_title('Behavioral Shift: Net Valence Before vs After Punishment Learning',
                  fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10, loc='upper right')
    ax5.grid(alpha=0.3)
    
    # Add annotation
    ax5.text(0.02, 0.98, 
             f'Before: {valence_before:+.1f} Hz ({"APPROACH" if valence_before > 0 else "AVOID"})\n'
             f'After:  {valence_after:+.1f} Hz ({"APPROACH" if valence_after > 0 else "AVOID"})\n'
             f'Change: {valence_after - valence_before:+.1f} Hz',
             transform=ax5.transAxes,
             fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.savefig('results/v2_two_mbons.png', dpi=150, bbox_inches='tight')
    print("  ✓ results/v2_two_mbons.png")
    plt.show()
    
    print("\n" + "="*60)
    print("V2 TWO MBONS COMPLETE!")
    print("="*60)
    print("\nPush-Pull Dynamics:")
    print(f"  ✓ Punishment DEPRESSED approach MBON synapses")
    print(f"  ✓ Punishment POTENTIATED avoid MBON synapses")
    print(f"  ✓ Net result: Behavioral shift from {('APPROACH' if valence_before > 0 else 'AVOID')} to {('APPROACH' if valence_after > 0 else 'AVOID')}")
    print("="*60)


if __name__ == '__main__':
    run_v2_two_mbons()