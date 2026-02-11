"""V1 Test Model: 2 KCs, 1 MBON, 1 DAN.
   The goal of this test model is to verify the rate equations work as expected."""
   
import numpy as np
from matplotlib import pyplot as plt
from mbmodel.models import NeuronPopulation
from mbmodel.utils import Recorder

print("="*40)
print("Starting V1 Test Model Simulation")
print("="*40)

# Simulation parameters
dt = 0.1 # ms
sim_time = 1000 # ms
n_steps = int(sim_time / dt)

# Create neuron populations
kc = NeuronPopulation(n=2, tau=10.0, r_max=100.0, activation='relu', name='KCs')
mbon = NeuronPopulation(n=1, tau=20.0, r_max=100.0, activation='relu', name='MBON')
dan = NeuronPopulation(n=1, tau=15.0, r_max=100.0, activation='relu', name='DAN')

# Connectivity 
w_kc_mbon = np.array([[0.5], [0.5]])  # 2 KCs to 1 MBON

rec = Recorder(record_every=10)

print(f"\nNetwork: {kc.n} KCs -> {mbon.n} MBON, {dan.n} DAN")
print(f"Simulation: {sim_time}ms, dt={dt}ms")
print(f"Weights: {w_kc_mbon[0]}\n")

# Main simulation loop

for step in range(n_steps):
   t = step * dt
   
   # odor onset: at t = 200 ms, activate KCs. duration of 60 seconds. 
   if 200 <= t < 260:
       kc_input = np.array([80.0, 20.0])  # strong input to one KC, weak to the other
   else:
       kc_input = np.array([0.0, 0.0])

   # shock input to DAN at t = 700 ms for 120 ms. 
   if 700 <= t < 820:
       dan_input = np.array([100.0])  # strong shock input
   else:
       dan_input = np.array([0.0])
       
   # update neurons 
   kc.update(kc_input, dt)
   
   # mbons receive input from kcs
   mbon_input = kc.r @ w_kc_mbon # matrix multiplication of kc rates and weights
   mbon.update(mbon_input.flatten(), dt)
   
   dan.update(dan_input, dt)
   
   # record data
   rec.record(time=t, r_KC=kc.r, r_MBON=mbon.r[0], r_DAN=dan.r[0])

print("Simulation complete.\n")

# Plot results
data = rec.get_arrays()
time = data['time']
    
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
# KCs
axes[0].plot(time, data['r_KC'][:, 0], label='KC1', linewidth=2)
axes[0].plot(time, data['r_KC'][:, 1], label='KC2', linewidth=2)
axes[0].axvspan(200, 260, alpha=0.2, color='blue')
axes[0].set_ylabel('Rate (Hz)')
axes[0].set_title('Kenyon Cells', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)
    
# MBON
axes[1].plot(time, data['r_MBON'], color='green', linewidth=2)
axes[1].axvspan(200, 260, alpha=0.2, color='blue', label='Odor')
axes[1].set_ylabel('Rate (Hz)')
axes[1].set_title('MBON', fontweight='bold')
axes[1].grid(alpha=0.3)
    
# DAN
axes[2].plot(time, data['r_DAN'], color='red', linewidth=2)
axes[2].axvspan(700, 820, alpha=0.2, color='red', label='Shock')
axes[2].set_ylabel('Rate (Hz)')
axes[2].set_xlabel('Time (ms)')
axes[2].set_title('Dopamine Neuron', fontweight='bold')
axes[2].grid(alpha=0.3)

plt.savefig('results/v1_minimal.png', dpi=150, bbox_inches='tight')
print("\n✓ Plot saved: results/v1_minimal.png")
plt.show()
    
print("\n" + "="*60)
print("V1 Complete! Rate dynamics verified.")
print("Next: Add plasticity (V2)")
print("="*60)

   