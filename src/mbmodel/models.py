__author__ = "Kavya Velliangiri"
__credits__ = ["Kavya Velliangiri"]
__license__ = "MIT"
__version__ = "v1"
__maintainer__ = "Kavya Velliangiri"

import numpy as np 
from mbmodel.utils import relu, sigmoid, euler_step, Recorder

class NeuronPopulation:
    """
    A class representing a rate-based neuron population. 
    
    Parameters:
        n: int
            The number of neurons in the population.
        tau: float
            The decay time constant for the firing rate.
        r_max: str
            The maximum firing rate for the neurons.
        activation: str
            The activation function to use ('relu' or 'sigmoid').
        name: str, optional
            The name of the neuron population for debugging purposes.
    """
    
    def __init__(self, n, tau, r_max, activation, name=None):
        self.n = n
        self.tau = tau
        self.r_max = r_max
        self.activation = activation
        self.name = name or f"NeuronPopulation{self.n}"
        
        self.r = np.zeros(n)  # Initialize firing rates to zero
    
    def update(self, input_current, dt):
        """
        Update the firing rates for one timestep.

        Parameters:
            input_current: ndarray
                input current
            dt: float
                timestep (ms)
        """
        if self.activation == 'relu':
            target = relu(input_current, self.r_max)
        elif self.activation == 'sigmoid':
            target = sigmoid(input_current)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

        self.r = euler_step(self.r, target, self.tau, dt)
        
        return self.r
    
    def reset(self):
        """Reset the firing rates to zero."""
        self.r = np.zeros(self.n)
        
    def __repr__(self):
        return f"<{self.name} mean rate: {np.mean(self.r):.2f} Hz>"
    
    