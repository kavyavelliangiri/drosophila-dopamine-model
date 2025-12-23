import numpy as np

def relu(x, max_val = None):
    """Computes the rectified linear activation function.

    Args:
        x: A numeric input (int, float, or array-like).
        max_val: An optional maximum value to cap the output.

    Returns:
        The result of applying the ReLU function to the input.
    """
    result = max(0, x)
    if max_val is not None:
        result = min(result, max_val)
    return result

def sigmoid(x, theta=0.0, gain=1.0):
    """Computes the sigmoid activation function.

    Args:
        x: A numeric input (int, float, or array-like).
        theta: The threshold parameter that shifts the sigmoid function.
        gain: The gain parameter that controls the steepness of the sigmoid function.

    Returns:
        The result of applying the sigmoid function to the input.
    """
    return 1.0 / (1.0 + np.exp(-gain * (x - theta)))

def euler_step(r, target, tau, dt):
    """Performs a single Euler integration step.

    Solves: τ * dr/dt = -r + target

    Args:
        r: The current neural firing rate
        target: The target value towards which to integrate.
        tau: The decay time constant for the firing rate.
        dt: The time step for the integration.

    Returns:
        The updated value after one Euler step.
    """
    dr = (-r + target) / tau
    r += dr * dt
    return r

class Recorder: 
    """A recorder class that logs values over n defined timestep. 
    
    Attributes: 
        record_every: int
            The timestep at which to record values. Defaults to 10. 
    """
    def __init__(self, record_every=10):
        self.record_every = record_every
        self.data = []
        self.step_count = 0

    def record(self, **kwargs):
        """Records the provided keyword arguments at defined intervals.

        Args:
            **kwargs: Key-value pairs to record.
            
        Usage:
            recorder.record(r_KC=kc.r, r_MBON=mbon.r, time=t)
            would record the spiking rates of KCs and MBONs along with the current time.
        """
        if self.step_count % self.record_every == 0:
            self.data.append(kwargs)
        self.step_count += 1

    def get_arrays(self): 
        """Converts the recorded data lists into a dictionary of numpy arrays.
        
        """
        return {key: np.array(val) for key, val in self.data.items()}
    
    def reset(self):
        """Resets recorder data clears step count."""
        self.data = []
        self.step_count = 0