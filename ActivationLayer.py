from collections.abc import Callable

import numpy as np

from Layer import Layer


class _ActivationLayer(Layer):
    """
    Base class for activation function layers.

    This class provides the common interface for all activation layers,
    implementing both the forward and backward passes.

    Parameters
    ----------
    activation : Callable[[np.ndarray], np.ndarray]
        The activation function to apply element-wise.
    activation_prime : Callable[[np.ndarray], np.ndarray]
        The derivative (gradient) of the activation function, used during backpropagation.

    Attributes
    ----------
    input : np.ndarray
        Cached input from the previous layer.
    output : np.ndarray
        Output after applying the activation function.

    Examples
    --------
    >>> import numpy as np
    >>> from _ActivationLayer import _ActivationLayer
    >>> relu = _ActivationLayer(lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float))
    >>> x = np.array([-1, 0, 2])
    >>> relu.forward(x)
    array([0., 0., 2.])
    >>> grad = relu.backward(np.array([1, 1, 1]), learning_rate=0.01)
    >>> grad
    array([0., 0., 1.])
    """

    def __init__(self, activation: Callable, activation_prime: Callable) -> None:
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through the activation function.

        Applies the activation function element-wise to the input tensor
        and stores the input for later use during backpropagation.

        Parameters
        ----------
        inputs : np.ndarray
            Input tensor from the previous layer.

        Returns
        -------
        np.ndarray
            Output tensor after applying the activation function.

        Examples
        --------
        >>> relu = _ActivationLayer(lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float))
        >>> relu.forward(np.array([-1, 2]))
        array([0., 2.])
        """
        self.input = inputs
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass through the activation function.

        Computes the gradient of the loss with respect to the input
        by multiplying the upstream gradient with the derivative of
        the activation function.

        Parameters
        ----------
        output_gradient : np.ndarray
            Gradient of the loss with respect to the layer's output.
        learning_rate : float
            Learning rate (not used in activations, included for API consistency).

        Returns
        -------
        np.ndarray
            Gradient of the loss with respect to the layer's input.

        Examples
        --------
        >>> act = _ActivationLayer(np.tanh, lambda x: 1 - np.tanh(x) ** 2)
        >>> x = np.array([-1.0, 0.0, 1.0])
        >>> act.forward(x)
        array([-0.76159416, 0., 0.76159416])
        >>> grad_out = np.array([1., 1., 1.])
        >>> act.backward(grad_out, learning_rate=0.01)
        array([0.41997434, 1., 0.41997434])
        """
        return np.multiply(output_gradient, self.activation_prime(self.input))
