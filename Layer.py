from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Layer(ABC):
    """
    Abstract base class for all neural network layers.

    This class defines the basic structure for a layer in a neural network,
    including forward and backward passes.

    Attributes
    ----------
    input : np.ndarray or None
        Cached input data from the forward pass.
    output : np.ndarray or None
        Output data computed during the forward pass.
    """

    def __init__(self) -> None:
        self.input: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Abstract Methods
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass through the layer.

        Parameters
        ----------
        inputs : np.ndarray
            Input data of shape `(n_features, n_samples)` or as defined by the model.

        Returns
        -------
        np.ndarray
            Output of the layer after computation.
        """
        pass

    @abstractmethod
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Perform the backward pass and compute gradients.

        Parameters
        ----------
        output_gradient : np.ndarray
            Gradient of the loss with respect to the layer's output.
        learning_rate : float
            Learning rate used for parameter updates.

        Returns
        -------
        np.ndarray
            Gradient of the loss with respect to the layer's input.
        """
        pass

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        Return a concise string representation for debugging.
        """
        cls_name = self.__class__.__name__
        input_shape = getattr(self.input, "shape", None)
        output_shape = getattr(self.output, "shape", None)
        return f"{cls_name}(input_shape={input_shape}, output_shape={output_shape})"
