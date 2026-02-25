"""Odor stimulus generation."""

import numpy as np


def create_sparse_odor(n_kcs, sparsity=0.1, strength=50.0, noise=0.1):
    """Create sparse odor input pattern.
    
    Parameters
    ----------
    n_kcs : int
        Number of KCs
    sparsity : float
        Fraction of KCs activated (0-1)
    strength : float
        Mean input strength
    noise : float
        Noise level (std relative to strength)
    
    Returns
    -------
    I : ndarray, shape (n_kcs,)
        Input currents
    active_indices : ndarray
        Which KCs are active
    """
    n_active = int(n_kcs * sparsity)
    active_indices = np.random.choice(n_kcs, n_active, replace=False)
    
    I = np.zeros(n_kcs)
    I[active_indices] = strength * (1 + noise * np.random.randn(n_active))
    I = np.maximum(I, 0)  # No negative inputs
    
    return I, active_indices


def temporal_odor_profile(duration, dt, rise_time=50, fall_time=100):
    """Create temporal profile for odor presentation.
    
    Returns envelope that rises, sustains, then falls.
    
    Parameters
    ----------
    duration : float
        Total duration (ms)
    dt : float
        Timestep (ms)
    rise_time : float
        Time to reach max (ms)
    fall_time : float
        Time to decay (ms)
    
    Returns
    -------
    envelope : ndarray
        Temporal profile (0 to 1)
    """
    n_steps = int(duration / dt)
    time = np.arange(n_steps) * dt
    envelope = np.zeros(n_steps)
    
    for i, t in enumerate(time):
        if t < rise_time:
            envelope[i] = t / rise_time
        elif t < duration - fall_time:
            envelope[i] = 1.0
        else:
            time_in_fall = t - (duration - fall_time)
            envelope[i] = 1.0 - (time_in_fall / fall_time)
    
    return np.clip(envelope, 0, 1)



