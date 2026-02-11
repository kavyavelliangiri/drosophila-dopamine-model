__author__ = "Kavya Velliangiri"
__credits__ = ["Kavya Velliangiri"]
__license__ = "MIT"
__version__ = "v2"
__maintainer__ = "Kavya Velliangiri"

"""A module that implements a dopaminergic plasticity rule for the KC>MBON synapses based on the activity of the DANs"""

import numpy as np
from mbmodel.utils import euler_step

class DopaminergicPlasticity:
    """A class implementing dopaminergic plasticity for KC>MBON synapses. Derived from Gkanias et al. 2024. 
    Three factor learning rule: 
    - presynaptic activity (KC) 
    - postsynaptic activity (MBON)
    - modulatory signal (DAN)

    Parameters:
    learning rate: float 
        The learning rate for synaptic updates, based on synaptic connectivity. 
    DAN_baseline: float
        Baseline dopamine activity level (Hz)
    tau_DAN: float
        Time constant for dopamine signal decay (ms) (default = 16 ms (from Huang et al 2019))
    w_min: float
        Minimum synaptic weight (default = 0.0)
    w_max: float
        Maximum synaptic weight (default = 1.0)
    """
    def __init__(self, learning_rate = 0.001, DAN_baseline = 20, tau_DAN=16.0, w_min=0.0, w_max=1.0):
        self.eta = learning_rate
        self.DAN_baseline = DAN_baseline
        self.tau_DAN = tau_DAN
        self.w_min = w_min
        self.w_max = w_max 

        # Current State 
        self.DA = DAN_baseline  # Current dopamine level

    def update_DA(self, r_DAN, dt):
        """Update dopamine concentration.
        
        DA follows DAN activity with exponential decay:
        τ_DA * dDA/dt = -DA + r_DAN
        
        Parameters
        ----------
        r_DAN : float or ndarray
            DAN firing rate(s) - if array, uses mean
        dt : float
            Timestep (ms)
        """
        # Use mean if multiple DANs
        r_mean = np.mean(r_DAN) if isinstance(r_DAN, np.ndarray) else r_DAN
        
        # Update DA concentration
        self.DA = euler_step(self.DA, r_mean, self.tau_DA, dt)

    def update_weights(self, W, r_pre, r_post, dt):
        """Update synaptic weights.
        
        Parameters
        ----------
        W : ndarray, shape (n_post, n_pre)
            Weight matrix
        r_pre : ndarray, shape (n_pre,)
            Presynaptic rates (KCs)
        r_post : ndarray, shape (n_post,)
            Postsynaptic rates (MBONs)
        dt : float
            Timestep (ms)
        
        Returns
        -------
        W : ndarray
            Updated weights
        """
        # Dopamine signal (deviation from baseline)
        DA_signal = self.DA - self.DA_baseline
        
        # Weight change: outer product of post × pre, modulated by DA
        # dW = η * r_post ⊗ r_pre * DA_signal
        dW = self.eta * np.outer(r_post, r_pre) * DA_signal * dt
        
        # Apply weight change
        W = W + dW
        
        # Enforce bounds
        W = np.clip(W, self.w_min, self.w_max)
        
        return W
    
    def reset(self):
        """Reset dopamine to baseline."""
        self.DA = self.DA_baseline

