from collections.abc import Sequence

import numpy as np

from ActivationFunction import *
from ErrorFunctions import MSE
from ErrorLayer import LossFunction
from Layer import Layer
from LearnLayer import ConstantLR, LearningRateScheduler, OneCycleLR
from Visualise import *


class NeuralNetwork:
    """
    A modular feedforward neural network implementation with support for
    custom layers, activation functions, and learning rate schedulers.

    This class supports sequential layer stacking, forward and backward propagation,
    training with configurable learning rate schedulers, validation tracking,
    and optional early stopping.

    Parameters
    ----------
    loss_function : LossFunction, optional
        Loss function to use during training and evaluation.
        Defaults to :class:`MSE`.

    Attributes
    ----------
    layers : list of Layer
        List of neural network layers in sequential order.
    loss_function : LossFunction
        Loss function used for computing training and evaluation losses.
    """

    def __init__(self, loss_function: LossFunction | None = None) -> None:
        self.layers: list[Layer] = []
        self.loss_function = loss_function if loss_function else MSE()

    # -------------------------------------------------------------------------
    def add(self, layer: Layer | Sequence[Layer]) -> None:
        """
        Add a layer or a sequence of layers to the network.

        Parameters
        ----------
        layer : Layer or Sequence[Layer]
            A single layer instance or a sequence of layers to append to the network.
        """
        if isinstance(layer, Layer):
            self.layers.append(layer)
        elif isinstance(layer, Sequence):
            for x in layer:
                self.add(x)

    # -------------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform a forward pass through the network to generate predictions.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape ``(n_samples, n_features)`` or ``(n_features,)``.

        Returns
        -------
        np.ndarray
            The network output after applying all layers sequentially.
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    # -------------------------------------------------------------------------
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        learning_rate: LearningRateScheduler | float = 0.01,
        verbose: bool = True,
        verbose_period: int = 1000,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        early_stopping_patience: int | None = None,
        min_delta: float = 1e-7,
    ) -> dict:
        """
        Train the neural network using the specified dataset and learning rate schedule.

        Supports optional validation data tracking and early stopping.

        Parameters
        ----------
        X : np.ndarray
            Training input samples of shape ``(n_samples, n_features, ...)``.
        y : np.ndarray
            Target output values of shape compatible with network output.
        epochs : int
            Number of training epochs.
        learning_rate : float or LearningRateScheduler, optional
            Either a fixed learning rate or an instance of a learning rate scheduler.
            Defaults to ``0.01``.
        verbose : bool, optional
            Whether to print progress during training. Defaults to ``True``.
        verbose_period : int, optional
            Number of epochs between progress prints. Defaults to ``1000``.
        validation_data : tuple of (np.ndarray, np.ndarray), optional
            Tuple containing validation inputs and targets. Defaults to ``None``.
        early_stopping_patience : int, optional
            Stop training if validation loss does not improve for this many epochs.
            Defaults to ``None`` (disabled).
        min_delta : float, optional
            Minimum change in validation loss to qualify as an improvement.
            Defaults to ``1e-7``.

        Returns
        -------
        dict
            Dictionary containing training history:
            - `'loss'`: list of training losses.
            - `'val_loss'`: list of validation losses (if validation data provided).
        """
        if not isinstance(learning_rate, LearningRateScheduler):
            learning_rate = ConstantLR(learning_rate)

        history = {"loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            lr = learning_rate.get_lr(epoch)
            total_loss = 0.0

            for i in range(len(X)):
                output = self.predict(X[i])

                loss = self.loss_function.forward(y[i], output)
                total_loss += loss

                gradient = self.loss_function.backward(y[i], output)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, lr)

            avg_loss = total_loss / len(X)
            history["loss"].append(avg_loss)

            val_loss = None
            if validation_data is not None:
                X_val, y_val = validation_data
                val_loss = self.evaluate(X_val, y_val)
                history["val_loss"].append(val_loss)

                if early_stopping_patience is not None:
                    if val_loss < best_val_loss - min_delta:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            if verbose:
                                print(f"Early stopping at epoch {epoch + 1}")
                            break

            if verbose and (epoch + 1) % verbose_period == 0:
                msg = f"Epoch {epoch + 1}/{epochs}, LR: {lr:.6f}, Loss: {avg_loss:.6f}"
                if validation_data is not None:
                    msg += f", Val Loss: {val_loss:.6f}"
                print(msg)

        final_accuracy = max(0, 100.0 * (1.0 - avg_loss))
        print(f"Training Complete - Final Score: {final_accuracy:.2f}%")

        return history

    # -------------------------------------------------------------------------
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the average loss of the network on the given dataset.

        Parameters
        ----------
        X : np.ndarray
            Input samples of shape ``(n_samples, n_features, ...)``.
        y : np.ndarray
            True target values.

        Returns
        -------
        float
            The mean loss over the dataset.
        """
        total_loss = 0.0
        for i in range(len(X)):
            prediction = self.predict(X[i])
            loss = self.loss_function.forward(y[i], prediction)
            total_loss += loss
        return total_loss / len(X)

    # -------------------------------------------------------------------------
    def get_accuracy(
        self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5
    ) -> float:
        """
        Compute classification accuracy for binary outputs.

        Parameters
        ----------
        X : np.ndarray
            Input samples.
        y : np.ndarray
            True binary labels.
        threshold : float, optional
            Decision threshold for classifying outputs. Defaults to ``0.5``.

        Returns
        -------
        float
            Classification accuracy as a value between 0 and 1.
        """
        correct = 0
        for i in range(len(X)):
            prediction = self.predict(X[i])
            predicted_class = (prediction > threshold).astype(int)
            if np.array_equal(predicted_class, y[i]):
                correct += 1
        return correct / len(X)

    # -------------------------------------------------------------------------
    def get_mse_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute a regression-style accuracy metric (based on MSE).

        Accuracy is defined as ``100 * (1 - MSE)``, clipped to a minimum of 0.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        y : np.ndarray
            True target values.

        Returns
        -------
        float
            Accuracy percentage (0–100% scale).
        """
        mse = self.evaluate(X, y)
        return max(0.0, 100.0 * (1.0 - mse))

