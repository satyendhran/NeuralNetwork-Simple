from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):
    """
    Abstract base class for all loss functions.

    This class defines the required interface for all loss functions.
    Subclasses must implement both `forward` and `backward` methods.

    Methods
    -------
    forward(y_true, y_pred)
        Compute the scalar loss value.
    backward(y_true, y_pred)
        Compute the gradient of the loss with respect to the predictions.
    """

    @abstractmethod
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the loss value.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth target values.
        y_pred : np.ndarray
            Predicted values from the model.

        Returns
        -------
        float
            Scalar loss value.
        """
        pass

    @abstractmethod
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss with respect to predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth target values.
        y_pred : np.ndarray
            Predicted values from the model.

        Returns
        -------
        np.ndarray
            Gradient of the loss with respect to `y_pred`.
        """
        pass

    # Optional: universal representation for debugging or doc printing
    def __repr__(self) -> str:
        """
        Return a string representation of the loss function with key attributes.
        """
        attrs = []
        if hasattr(self, "epsilon"):
            attrs.append(f"epsilon={self.epsilon}")
        if hasattr(self, "delta"):
            attrs.append(f"delta={self.delta}")
        params = ", ".join(attrs) if attrs else "no parameters"
        return f"{self.__class__.__name__}({params})"
