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
    
class MushroomBodyNetwork:
    """Full mushroom body circuit.
    
    Parameters
    ----------
    config : dict
        Configuration with network parameters
    """
    
    def __init__(self, config):
        from mbmodel.connectivity import load_flywire_connectivity, create_random_sparse
        from mbmodel.plasticity import DopaminePlasticity
        
        # Extract config
        self.config = config
        self.dt = config.get('dt', 0.1)
        
        # Create populations
        self.KCs = NeuronPopulation(
            n=config['n_KCs'],
            tau=config.get('tau_KC', 10.0),
            r_max=config.get('r_max_KC', 100.0),
            name="KCs"
        )
        
        self.MBONs = NeuronPopulation(
            n=config['n_MBONs'],
            tau=config.get('tau_MBON', 20.0),
            r_max=config.get('r_max_MBON', 100.0),
            name="MBONs"
        )
        
        self.DANs = NeuronPopulation(
            n=config['n_DANs'],
            tau=config.get('tau_DAN', 15.0),
            r_max=config.get('r_max_DAN', 100.0),
            name="DANs"
        )
        
        # Load connectivity
        if 'connectome_path' in config:
            self.W_KC_MBON, _ = load_flywire_connectivity(
                config['connectome_path'])
        else:
            self.W_KC_MBON = create_random_sparse(
                config['n_KCs'], config['n_MBONs'], sparsity=0.1)
        
        # Store initial weights
        self.W_initial = self.W_KC_MBON.copy()
        
        # Plasticity
        self.plasticity = DopaminePlasticity(
            learning_rate=config.get('learning_rate', 0.001),
            DA_baseline=config.get('DA_baseline', 20.0),
            tau_DA=config.get('tau_DA', 500.0)
        )
    
    def step(self, I_KC, I_DAN):
        """Run one timestep.
        
        Parameters
        ----------
        I_KC : ndarray, shape (n_KCs,)
        I_DAN : ndarray, shape (n_DANs,)
        """
        # Update neurons
        self.KCs.update(I_KC, self.dt)
        
        I_MBON = self.W_KC_MBON @ self.KCs.r
        self.MBONs.update(I_MBON, self.dt)
        
        self.DANs.update(I_DAN, self.dt)
        
        # Update plasticity
        self.plasticity.update_DA(self.DANs.r, self.dt)
        self.W_KC_MBON = self.plasticity.update_weights(
            self.W_KC_MBON, self.KCs.r, self.MBONs.r, self.dt)
    
    def reset_activity(self):
        """Reset firing rates, keep weights."""
        self.KCs.reset()
        self.MBONs.reset()
        self.DANs.reset()
        self.plasticity.reset()
    
    def reset_weights(self):
        """Reset to initial weights."""
        self.W_KC_MBON = self.W_initial.copy()
    
    def get_weight_change(self):
        """Compute weight change from initial."""
        return self.W_KC_MBON - self.W_initial

    
    