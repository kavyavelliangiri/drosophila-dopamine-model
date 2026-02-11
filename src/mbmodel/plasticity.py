__author__ = "Kavya Velliangiri"
__credits__ = ["Kavya Velliangiri"]
__license__ = "MIT"
__version__ = "v2"
__maintainer__ = "Kavya Velliangiri"

"""A module that implements a dopaminergic plasticity rule for the KC>MBON synapses based on the activity of the DANs:
Implements the detailed plasticity rule from Gkanias et al. with:
- Dopamine receptor signaling
- ER-cAMP concentration dynamics
- Ca2+ concentration dynamics
- Three-factor weight update rule
"""

import numpy as np
from mbmodel.utils import euler_step


"""Dopamine-modulated plasticity with cAMP and Ca2+ dynamics.

FIXED VERSION with proper biological signaling.
"""

import numpy as np
from mbmodel.utils import euler_step


class BiologicalPlasticity:
    """Biologically detailed dopamine-modulated plasticity.
    
    Implements the weight update rule:
    ΔW_k2m(t) = δ(t)[κ^i(t) + W_k2m(t) - w_rest]
    
    Parameters
    ----------
    learning_rate : float
        Global learning rate (default: 0.002, increased from 0.001)
    w_rest : float
        Resting weight value (default: 1.0)
    tau_eligibility : float
        Eligibility trace decay time constant in ms (default: 1000)
    tau_cAMP : float
        cAMP dynamics time constant in ms (default: 500)
    tau_Ca : float
        Ca2+ dynamics time constant in ms (default: 200)
    DA_baseline : float
        Baseline dopamine level (default: 20.0 Hz)
    cAMP_baseline : float
        Baseline cAMP concentration (default: 0.5)
    Ca_baseline : float
        Baseline Ca2+ concentration (default: 0.1)
    kappa_scale : float
        Scaling factor for kappa (default: 10.0)
    w_min : float
        Minimum weight value (default: 0.0)
    w_max : float
        Maximum weight value (default: 10.0)
    """
    
    def __init__(self, 
                 learning_rate=0.002,  # Increased slightly
                 w_rest=1.0,
                 tau_eligibility=1000.0,
                 tau_cAMP=500.0,
                 tau_Ca=200.0,
                 DA_baseline=20.0,
                 cAMP_baseline=0.5,
                 Ca_baseline=0.1,
                 kappa_scale=10.0,  # NEW: scale kappa appropriately
                 w_min=0.0,
                 w_max=10.0):
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.w_rest = w_rest
        self.kappa_scale = kappa_scale
        
        # Time constants
        self.tau_eligibility = tau_eligibility
        self.tau_cAMP = tau_cAMP
        self.tau_Ca = tau_Ca
        
        # Baseline concentrations
        self.DA_baseline = DA_baseline
        self.cAMP_baseline = cAMP_baseline
        self.Ca_baseline = Ca_baseline
        
        # Weight bounds
        self.w_min = w_min
        self.w_max = w_max
        
        # State variables
        self.eligibility = None
        self.cAMP = cAMP_baseline
        self.Ca = Ca_baseline
        self.DA = DA_baseline
        
    def update_DA(self, r_DAN, dt):
        """Update dopamine concentration from DAN activity."""
        r_mean = np.mean(r_DAN) if isinstance(r_DAN, np.ndarray) else r_DAN
        tau_DA = 100.0  # Fast DA clearance
        self.DA = euler_step(self.DA, r_mean, tau_DA, dt)
    
    def update_cAMP(self, dt):
        """Update cAMP concentration based on dopamine.
        
        FIXED: cAMP should INCREASE with positive DA signal for LTP.
        """
        # Dopamine signal (deviation from baseline)
        DA_signal = self.DA - self.DA_baseline
        
        # cAMP production is RECTIFIED - only increases with positive DA
        # This prevents negative kappa during the odor presentation
        if DA_signal > 0:
            # Positive DA -> increase cAMP above baseline
            cAMP_target = self.cAMP_baseline + 0.5 * DA_signal  # Increased gain
        else:
            # Negative DA -> return to baseline (but don't go below)
            cAMP_target = self.cAMP_baseline
        
        cAMP_target = np.clip(cAMP_target, self.cAMP_baseline, 3.0)
        
        # Update cAMP
        self.cAMP = euler_step(self.cAMP, cAMP_target, self.tau_cAMP, dt)
    
    def update_Ca(self, r_post, dt):
        """Update Ca2+ concentration based on postsynaptic activity."""
        mean_post_rate = np.mean(r_post)
        
        # Ca2+ influx during MBON activity
        Ca_target = self.Ca_baseline + 0.02 * mean_post_rate  # Increased gain
        Ca_target = np.clip(Ca_target, 0, 2.0)
        
        # Fast Ca2+ dynamics
        self.Ca = euler_step(self.Ca, Ca_target, self.tau_Ca, dt)
    
    def update_eligibility(self, r_pre, dt):
        """Update eligibility traces for each synapse."""
        if self.eligibility is None:
            self.eligibility = np.zeros_like(r_pre)
        
        # Eligibility increases with presynaptic activity
        target = r_pre / 100.0
        self.eligibility = euler_step(
            self.eligibility, target, self.tau_eligibility, dt
        )
        
        return self.eligibility
    
    def compute_kappa(self):
        """Compute intracellular signaling term κ^i(t).
        
        FIXED: Use multiplicative interaction that's always positive
        when both signals are elevated.
        
        Returns
        -------
        kappa : float
            Intracellular signaling strength
        """
        # Normalized deviations from baseline
        cAMP_norm = (self.cAMP - self.cAMP_baseline) / self.cAMP_baseline
        Ca_norm = (self.Ca - self.Ca_baseline) / self.Ca_baseline
        
        # Multiplicative interaction (Hebbian-like)
        # Both must be elevated for strong potentiation
        kappa = self.kappa_scale * cAMP_norm * Ca_norm
        
        return kappa
    
    def update_weights(self, W, r_pre, r_post, dt):
        """Update synaptic weights using biological plasticity rule.
        
        Implements: ΔW_k2m(t) = η * δ(t)[κ^i(t) + W_k2m(t) - w_rest]
        """
        # Update all biological variables
        eligibility = self.update_eligibility(r_pre, dt)
        kappa = self.compute_kappa()
        
        # Broadcast eligibility across MBONs
        delta = eligibility[np.newaxis, :]  # Shape: (1, n_pre)
        
        # Weight update with homeostatic term
        plasticity_term = kappa + W - self.w_rest
        
        # Weight change
        dW = self.learning_rate * delta * plasticity_term * dt
        
        # Apply weight change
        W = W + dW
        
        # Enforce bounds
        W = np.clip(W, self.w_min, self.w_max)
        
        return W
    
    def reset(self):
        """Reset all state variables to baseline."""
        self.DA = self.DA_baseline
        self.cAMP = self.cAMP_baseline
        self.Ca = self.Ca_baseline
        self.eligibility = None


class DopaminePlasticity:
    """Simple dopamine-modulated plasticity (original version)."""
    
    def __init__(self, learning_rate=0.001, DA_baseline=20.0,
                 tau_DA=500.0, w_min=0.0, w_max=10.0):
        self.eta = learning_rate
        self.DA_baseline = DA_baseline
        self.tau_DA = tau_DA
        self.w_min = w_min
        self.w_max = w_max
        self.DA = DA_baseline
    
    def update_DA(self, r_DAN, dt):
        """Update dopamine concentration."""
        r_mean = np.mean(r_DAN) if isinstance(r_DAN, np.ndarray) else r_DAN
        self.DA = euler_step(self.DA, r_mean, self.tau_DA, dt)
    
    def update_weights(self, W, r_pre, r_post, dt):
        """Update synaptic weights (simple rule)."""
        DA_signal = self.DA - self.DA_baseline
        dW = self.eta * np.outer(r_post, r_pre) * DA_signal * dt
        W = W + dW
        W = np.clip(W, self.w_min, self.w_max)
        return W
    
    def reset(self):
        """Reset dopamine to baseline."""
        self.DA = self.DA_baseline