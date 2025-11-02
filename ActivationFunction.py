from _ActivationFunctions import (
    _ELU,
    _SELU,
    _Absolute,
    _BinaryStep,
    _HardSigmoid,
    _LeakyReLU,
    _Linear,
    _Mish,
    _ParametricReLU,
    _Power,
    _ReLU,
    _Sigmoid,
    _Softplus,
    _Swish,
    _Tanh,
)
from Dense import Dense
from Layer import Layer


def _Activation(activation_cls):
    """
    Factory function for creating (Dense, Activation) layer pairs.

    This function wraps a given activation class and returns a callable factory.
    When called with layer dimensions, it creates a `Dense` layer followed by
    the specified activation layer.

    Parameters
    ----------
    activation_cls : type
        Activation layer class (e.g., `_ReLU`, `_Sigmoid`, `_Tanh`, etc.).

    Returns
    -------
    Callable[[int, int, ...], tuple[Dense, Layer]]
        A factory function that, when called, returns a tuple `(Dense, Activation)`.

    Examples
    --------
    >>> Sigmoid = _Activation(_Sigmoid)
    >>> dense_layer, activation_layer = Sigmoid(4, 3)
    >>> type(dense_layer).__name__
    'Dense'
    >>> type(activation_layer).__name__
    '_Sigmoid'
    """

    def factory(input_size: int, output_size: int, **kwargs) -> tuple[Dense, Layer]:
        """
        Creates a Dense layer followed by the given activation layer.

        Parameters
        ----------
        input_size : int
            Number of input neurons.
        output_size : int
            Number of output neurons.
        **kwargs : dict, optional
            Additional keyword arguments for activation layer initialization.

        Returns
        -------
        tuple of (Dense, Layer)
            Tuple containing a `Dense` layer and an activation layer instance.

        Examples
        --------
        >>> ReLU = _Activation(_ReLU)
        >>> dense, act = ReLU(8, 4)
        >>> isinstance(dense, Dense)
        True
        >>> isinstance(act, Layer)
        True
        """
        return Dense(input_size, output_size), activation_cls(**kwargs)

    return factory


# Activation factories
Sigmoid = _Activation(_Sigmoid)
HardSigmoid = _Activation(_HardSigmoid)
Tanh = _Activation(_Tanh)
ReLU = _Activation(_ReLU)
LeakyReLU = _Activation(_LeakyReLU)
ELU = _Activation(_ELU)
SELU = _Activation(_SELU)
Swish = _Activation(_Swish)
Mish = _Activation(_Mish)
ParametricReLU = _Activation(_ParametricReLU)
Softplus = _Activation(_Softplus)
Power = _Activation(_Power)
Linear = _Activation(_Linear)
BinaryStep = _Activation(_BinaryStep)
Absolute = _Activation(_Absolute)
