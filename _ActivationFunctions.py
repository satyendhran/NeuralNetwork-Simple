import numpy as np

from ActivationLayer import _ActivationLayer


class _Sigmoid(_ActivationLayer):
    """
    Sigmoid activation function.

    Computes:
        f(x) = 1 / (1 + exp(-x))

    Returns values in the range (0, 1).

    Notes
    -----
    This function is numerically stabilized using clipping at [-500, 500].

    Examples
    --------
    >>> act = _Sigmoid()
    >>> x = np.array([-1, 0, 1])
    >>> act._activation(x)
    array([0.26894142, 0.5, 0.73105858])
    """

    def __init__(self) -> None:
        super().__init__(self._activation, self._activation_prime)

    def _activation(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _activation_prime(self, x: np.ndarray) -> np.ndarray:
        s = self._activation(x)
        return s * (1.0 - s)


class _Tanh(_ActivationLayer):
    """
    Hyperbolic tangent activation function.

    Computes:
        f(x) = tanh(x)

    Output range: (-1, 1).

    Examples
    --------
    >>> act = _Tanh()
    >>> act._activation(np.array([-1, 0, 1]))
    array([-0.76159416,  0.,  0.76159416])
    """

    def __init__(self) -> None:
        super().__init__(np.tanh, lambda x: 1.0 - np.tanh(x) ** 2)


class _ReLU(_ActivationLayer):
    """
    Rectified Linear Unit (ReLU).

    Computes:
        f(x) = max(0, x)

    Returns 0 for negative inputs and x otherwise.

    Examples
    --------
    >>> act = _ReLU()
    >>> act._activation(np.array([-1, 0, 2]))
    array([0., 0., 2.])
    """

    def __init__(self) -> None:
        super().__init__(lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float))


class _LeakyReLU(_ActivationLayer):
    """
    Leaky Rectified Linear Unit (Leaky ReLU).

    Computes:
        f(x) = x if x > 0 else alpha * x

    Parameters
    ----------
    alpha : float, optional
        Slope for negative values, by default 0.01.

    Examples
    --------
    >>> act = _LeakyReLU(alpha=0.1)
    >>> act._activation(np.array([-1, 0, 1]))
    array([-0.1, 0., 1.])
    """

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha
        super().__init__(
            lambda x: np.where(x > 0, x, self.alpha * x),
            lambda x: np.where(x > 0, 1.0, self.alpha),
        )


class _ELU(_ActivationLayer):
    """
    Exponential Linear Unit (ELU).

    Computes:
        f(x) = x if x > 0 else alpha * (exp(x) - 1)

    Parameters
    ----------
    alpha : float, optional
        Scaling factor for negative region, by default 1.0.

    Notes
    -----
    Inputs are clipped to [-500, 500] to prevent overflow.

    Examples
    --------
    >>> act = _ELU(alpha=1.0)
    >>> act._activation(np.array([-1, 0, 1]))
    array([-0.63212056, 0., 1.])
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        super().__init__(
            lambda x: np.where(
                x > 0, x, self.alpha * (np.exp(np.clip(x, -500, 500)) - 1)
            ),
            lambda x: np.where(x > 0, 1.0, self.alpha * np.exp(np.clip(x, -500, 500))),
        )


class _SELU(_ActivationLayer):
    """
    Scaled Exponential Linear Unit (SELU).

    Computes:
        f(x) = scale * (x if x > 0 else alpha * (exp(x) - 1))

    Notes
    -----
    SELU is self-normalizing and helps maintain zero mean and unit variance
    during training.

    Examples
    --------
    >>> act = _SELU()
    >>> act._activation(np.array([-1, 0, 1]))
    array([-1.11133074, 0., 1.05070099])
    """

    def __init__(self) -> None:
        self.alpha = 1.6732632423543772
        self.scale = 1.0507009873554805
        super().__init__(
            lambda x: self.scale
            * np.where(x > 0, x, self.alpha * (np.exp(np.clip(x, -500, 500)) - 1)),
            lambda x: self.scale
            * np.where(x > 0, 1.0, self.alpha * np.exp(np.clip(x, -500, 500))),
        )


class _Swish(_ActivationLayer):
    """
    Swish activation function.

    Computes:
        f(x) = x * sigmoid(x)

    Examples
    --------
    >>> act = _Swish()
    >>> act._activation(np.array([-1, 0, 1]))
    array([-0.26894142, 0., 0.73105858])
    """

    def __init__(self) -> None:
        super().__init__(self._activation, self._activation_prime)

    def _activation(self, x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _activation_prime(self, x: np.ndarray) -> np.ndarray:
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return sigmoid + x * sigmoid * (1.0 - sigmoid)


class _Mish(_ActivationLayer):
    """
    Mish activation function.

    Computes:
        f(x) = x * tanh(softplus(x))
        where softplus(x) = log(1 + exp(x))

    Examples
    --------
    >>> act = _Mish()
    >>> act._activation(np.array([-1, 0, 1]))
    array([-0.303401, 0., 0.865098])
    """

    def __init__(self) -> None:
        super().__init__(self._activation, self._activation_prime)

    def _activation(self, x: np.ndarray) -> np.ndarray:
        return x * np.tanh(np.log(1.0 + np.exp(np.clip(x, -500, 500))))

    def _activation_prime(self, x: np.ndarray) -> np.ndarray:
        omega = (
            4.0 * (x + 1.0)
            + 4.0 * np.exp(2.0 * np.clip(x, -500, 500))
            + np.exp(3.0 * np.clip(x, -500, 500))
            + np.exp(np.clip(x, -500, 500)) * (4.0 * x + 6.0)
        )
        delta = (
            2.0 * np.exp(np.clip(x, -500, 500))
            + np.exp(2.0 * np.clip(x, -500, 500))
            + 2.0
        )
        return np.exp(np.clip(x, -500, 500)) * omega / (delta**2)


class _ParametricReLU(_ActivationLayer):
    """
    Parametric Rectified Linear Unit (PReLU).

    Computes:
        f(x) = x if x >= 0 else alpha * x

    Parameters
    ----------
    alpha : float, optional
        Learnable or fixed negative slope, by default 0.25.

    Examples
    --------
    >>> act = _ParametricReLU(alpha=0.2)
    >>> act._activation(np.array([-1, 0, 1]))
    array([-0.2, 0., 1.])
    """

    def __init__(self, alpha: float = 0.25) -> None:
        self.alpha = alpha
        super().__init__(
            lambda x: np.where(x >= 0, x, self.alpha * x),
            lambda x: np.where(x >= 0, 1.0, self.alpha),
        )


class _HardSigmoid(_ActivationLayer):
    """
    Hard (piecewise linear) sigmoid approximation.

    Computes:
        f(x) = clip(0.2 * x + 0.5, 0, 1)

    Examples
    --------
    >>> act = _HardSigmoid()
    >>> act._activation(np.array([-3, 0, 3]))
    array([0., 0.5, 1.])
    """

    def __init__(self) -> None:
        super().__init__(
            lambda x: np.clip(0.2 * x + 0.5, 0.0, 1.0),
            lambda x: np.where((x > -2.5) & (x < 2.5), 0.2, 0.0),
        )


class _Softplus(_ActivationLayer):
    """
    Softplus activation function.

    Computes:
        f(x) = log(1 + exp(x))

    Smooth approximation of ReLU.

    Examples
    --------
    >>> act = _Softplus()
    >>> act._activation(np.array([-1, 0, 1]))
    array([0.31326169, 0.69314718, 1.31326169])
    """

    def __init__(self) -> None:
        super().__init__(
            lambda x: np.log(1.0 + np.exp(np.clip(x, -500, 500))),
            lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
        )


class _Power(_ActivationLayer):
    """
    Power activation function.

    Computes:
        f(x) = x^n

    Parameters
    ----------
    n : float, optional
        Exponent for the power function, by default 2.

    Examples
    --------
    >>> act = _Power(n=3)
    >>> act._activation(np.array([-2, 0, 2]))
    array([-8, 0, 8])
    """

    def __init__(self, n: float = 2) -> None:
        self.n = n
        super().__init__(lambda x: x**self.n, lambda x: self.n * x ** (self.n - 1))


class _Absolute(_ActivationLayer):
    """
    Absolute value activation function.

    Computes:
        f(x) = |x|

    Examples
    --------
    >>> act = _Absolute()
    >>> act._activation(np.array([-2, 0, 2]))
    array([2, 0, 2])
    """

    def __init__(self) -> None:
        super().__init__(np.abs, lambda x: np.where(x >= 0, 1.0, -1.0))


class _Linear(_ActivationLayer):
    """
    Linear activation function.

    Computes:
        f(x) = slope * x + intercept

    Parameters
    ----------
    slope : float, optional
        Linear slope, by default 1.0.
    intercept : float, optional
        Linear intercept, by default 0.0.

    Examples
    --------
    >>> act = _Linear(slope=2.0, intercept=1.0)
    >>> act._activation(np.array([-1, 0, 1]))
    array([-1., 1., 3.])
    """

    def __init__(self, slope: float = 1.0, intercept: float = 0.0) -> None:
        self.slope = slope
        self.intercept = intercept
        super().__init__(
            lambda x: self.slope * x + self.intercept,
            lambda x: np.full_like(x, self.slope),
        )


class _BinaryStep(_ActivationLayer):
    """
    Binary step activation function.

    Computes:
        f(x) = 1 if x >= threshold else 0

    Parameters
    ----------
    threshold : float, optional
        Cutoff threshold, by default 0.0.

    Notes
    -----
    The derivative is zero everywhere (not suitable for gradient-based optimization).

    Examples
    --------
    >>> act = _BinaryStep(threshold=0.5)
    >>> act._activation(np.array([0.2, 0.7]))
    array([0., 1.])
    """

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold
        super().__init__(
            lambda x: np.where(x >= self.threshold, 1.0, 0.0),
            lambda x: np.zeros_like(x),
        )
