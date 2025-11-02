import numpy as np

from Layer import Layer


class Dense(Layer):
    """
    Fully connected (dense) layer with configurable weight initialization.

    Parameters
    ----------
    input_size : int
        Number of input features.
    output_size : int
        Number of output features.
    weight_init : {'xavier', 'he', 'lecun'}, optional
        Weight initialization method. Defaults to 'xavier'.

    Attributes
    ----------
    weights : np.ndarray
        Weight matrix of shape (output_size, input_size).
    bias : np.ndarray
        Bias vector of shape (output_size, 1).
    input : np.ndarray
        Cached input from forward pass.
    output : np.ndarray
        Output of the layer after forward pass.
    """

    def __init__(
        self, input_size: int, output_size: int, weight_init: str = "xavier"
    ) -> None:
        super().__init__()

        if weight_init not in ("xavier", "he", "lecun"):
            raise ValueError(
                f"Invalid weight_init '{weight_init}'. Choose from 'xavier', 'he', 'lecun'."
            )

        self.weights = self._initialize_weights(
            input_size, output_size, method=weight_init
        )
        self.bias = np.zeros((output_size, 1))

    @staticmethod
    def _initialize_weights(
        input_size: int, output_size: int, method: str
    ) -> np.ndarray:
        """Return initialized weight matrix."""
        if method == "he":
            std = np.sqrt(2.0 / input_size)
            return np.random.randn(output_size, input_size) * std
        elif method == "lecun":
            std = np.sqrt(1.0 / input_size)
            return np.random.randn(output_size, input_size) * std
        else:  # Xavier
            limit = np.sqrt(6.0 / (input_size + output_size))
            return np.random.uniform(-limit, limit, (output_size, input_size))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass.

        Parameters
        ----------
        inputs : np.ndarray
            Input data of shape (input_size, batch_size).

        Returns
        -------
        np.ndarray
            Output of shape (output_size, batch_size).
        """
        self.input = inputs
        self.output = np.dot(self.weights, self.input) + self.bias
        return self.output

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Compute backward pass and update weights and bias.

        Parameters
        ----------
        output_gradient : np.ndarray
            Gradient from the next layer of shape (output_size, batch_size).
        learning_rate : float
            Learning rate for parameter updates.

        Returns
        -------
        np.ndarray
            Gradient for the previous layer of shape (input_size, batch_size).
        """
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)

        # In-place updates for performance
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * np.sum(output_gradient, axis=1, keepdims=True)

        return input_gradient

    def __repr__(self) -> str:
        return f"Dense(input={self.weights.shape[1]}, output={self.weights.shape[0]}, weights={self.weights.shape}, bias={self.bias.shape})"

    def __str__(self) -> str:
        return f"Dense Layer: {self.weights.shape[1]} → {self.weights.shape[0]}"
