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