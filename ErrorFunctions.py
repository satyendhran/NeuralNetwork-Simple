import numpy as np

from ErrorLayer import LossFunction


class MSE(LossFunction):
    """
    Mean Squared Error (MSE) loss function.

    Formula
    -------
    .. math::
        L = \\frac{1}{n} \\sum_i (y_i - \\hat{y}_i)^2

    Attributes
    ----------
    None
    """

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the mean squared error.

        Parameters
        ----------
        y_true : np.ndarray
            True target values.
        y_pred : np.ndarray
            Predicted values.

        Returns
        -------
        float
            The mean squared error.
        """
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of MSE with respect to predictions.

        Returns
        -------
        np.ndarray
            Gradient of the loss.
        """
        return 2.0 * (y_pred - y_true) / y_true.size


class MAE(LossFunction):
    """
    Mean Absolute Error (MAE) loss function.

    Formula
    -------
    .. math::
        L = \\frac{1}{n} \\sum_i |y_i - \\hat{y}_i|
    """

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.sign(y_pred - y_true) / y_true.size


class Huber(LossFunction):
    """
    Huber loss (Smooth L1 Loss).

    Provides a balance between MSE and MAE, being less sensitive to outliers.

    Parameters
    ----------
    delta : float, optional
        Threshold for switching between quadratic and linear loss.
        Defaults to 1.0.

    Formula
    -------
    .. math::
        L = \\begin{cases}
        0.5 (y - \\hat{y})^2, & |y - \\hat{y}| \\le \\delta \\\\
        \\delta (|y - \\hat{y}| - 0.5 \\delta), & \\text{otherwise}
        \\end{cases}
    """

    def __init__(self, delta: float = 1.0) -> None:
        self.delta = delta

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        error = y_true - y_pred
        abs_error = np.abs(error)
        is_small = abs_error <= self.delta
        squared_loss = 0.5 * error**2
        linear_loss = self.delta * (abs_error - 0.5 * self.delta)
        return np.mean(np.where(is_small, squared_loss, linear_loss))

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        error = y_pred - y_true
        abs_error = np.abs(error)
        is_small = abs_error <= self.delta
        grad = np.where(is_small, error, self.delta * np.sign(error))
        return grad / y_true.size


class BinaryCrossEntropy(LossFunction):
    """
    Binary Cross-Entropy (BCE) loss with numerical stability.

    Suitable for binary classification tasks.

    Parameters
    ----------
    epsilon : float, optional
        Small constant for numerical stability.
        Defaults to 1e-15.

    Formula
    -------
    .. math::
        L = -\\frac{1}{n} \\sum_i [y_i \\log(\\hat{y}_i) + (1 - y_i) \\log(1 - \\hat{y}_i)]
    """

    def __init__(self, epsilon: float = 1e-15) -> None:
        self.epsilon = epsilon

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, self.epsilon, 1.0 - self.epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, self.epsilon, 1.0 - self.epsilon)
        grad = ((1.0 - y_true) / (1.0 - y_pred) - y_true / y_pred) / y_true.size
        return grad


class CategoricalCrossEntropy(LossFunction):
    """
    Categorical Cross-Entropy (CCE) loss for multi-class classification.

    Assumes one-hot encoded `y_true`.

    Parameters
    ----------
    epsilon : float, optional
        Small constant for numerical stability.
        Defaults to 1e-15.

    Formula
    -------
    .. math::
        L = -\\frac{1}{n} \\sum_i \\sum_j y_{ij} \\log(\\hat{y}_{ij})
    """

    def __init__(self, epsilon: float = 1e-15) -> None:
        self.epsilon = epsilon

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, self.epsilon, 1.0 - self.epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=0))

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, self.epsilon, 1.0 - self.epsilon)
        return -y_true / (y_pred * y_true.size)
