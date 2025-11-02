"""
Neural Network Visualization Utilities
======================================

This module provides advanced visualization tools using `pyqtgraph` to analyze
neural network training and prediction behavior. It supports visualizing
training history, prediction accuracy, and error analysis for regression-style
models.

Each function launches an interactive PyQtGraph window for detailed
exploration of model results.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

# =====================================================================
# Internal Helper
# =====================================================================


def _predict_scalar(nn, x):
    """
    Predict a scalar output from a neural network model.

    Ensures that the network's output is converted to a single float value,
    regardless of the output shape.

    Parameters
    ----------
    nn : object
        Neural network object implementing a `predict` method that returns
        NumPy-compatible arrays.
    x : array-like
        Input data sample. Can be 1D or 2D.

    Returns
    -------
    float
        Scalar prediction result.
    """
    pred = nn.predict(np.asarray(x))
    try:
        return float(pred)
    except Exception:
        return float(np.asarray(pred).flatten()[0])


# =====================================================================
# Plotting Functions
# =====================================================================


def plot_training_history(history: dict, title: str = "Training History") -> None:
    """
    Plot training and validation loss across epochs.

    Parameters
    ----------
    history : dict
        Dictionary containing loss history. Must contain the key `'loss'`,
        and optionally `'val_loss'`.
    title : str, optional
        Window title for the plot (default is `"Training History"`).
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title=title)
    win.resize(800, 600)
    win.setWindowTitle(title)

    plot = win.addPlot(title="Loss over Epochs")
    plot.setLabel("left", "Loss")
    plot.setLabel("bottom", "Epoch")
    plot.addLegend()
    plot.showGrid(x=True, y=True, alpha=0.3)

    epochs = np.arange(len(history["loss"]))
    plot.plot(epochs, history["loss"], pen=pg.mkPen("b", width=2), name="Training Loss")

    if "val_loss" in history and len(history["val_loss"]) > 0:
        plot.plot(
            epochs,
            history["val_loss"],
            pen=pg.mkPen("r", width=2),
            name="Validation Loss",
        )

    QtWidgets.QApplication.instance().exec()


def plot_predictions(
    nn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    title: str = "Neural Network Predictions",
) -> None:
    """
    Plot neural network predictions for both training and test data.

    Parameters
    ----------
    nn : object
        Neural network model implementing a `predict` method.
    X_train : np.ndarray
        Training input data.
    y_train : np.ndarray
        Ground truth training outputs.
    X_test : np.ndarray
        Test input data.
    y_test : np.ndarray
        Ground truth test outputs.
    title : str, optional
        Title for the plot window.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title=title)
    win.resize(1200, 600)
    win.setWindowTitle(title)

    # Sort training data
    X_train_flat = X_train.flatten()
    y_train_flat = y_train.flatten()
    train_sort_idx = np.argsort(X_train_flat)
    X_train_sorted = X_train_flat[train_sort_idx]
    y_train_sorted = y_train_flat[train_sort_idx]

    train_preds = np.array([_predict_scalar(nn, X_train[i]) for i in train_sort_idx])

    # Sort test data
    X_test_flat = X_test.flatten()
    y_test_flat = y_test.flatten()
    test_sort_idx = np.argsort(X_test_flat)
    X_test_sorted = X_test_flat[test_sort_idx]
    y_test_sorted = y_test_flat[test_sort_idx]

    test_preds = np.array([_predict_scalar(nn, X_test[i]) for i in test_sort_idx])

    # Training plot
    train_plot = win.addPlot(title="Training Data")
    train_plot.setLabel("left", "Output")
    train_plot.setLabel("bottom", "Input")
    train_plot.addLegend()
    train_plot.showGrid(x=True, y=True, alpha=0.3)

    train_plot.plot(
        X_train_sorted, y_train_sorted, pen=pg.mkPen("b", width=2), name="True Values"
    )
    train_plot.plot(
        X_train_sorted,
        train_preds,
        pen=pg.mkPen("r", width=2, style=QtCore.Qt.PenStyle.DashLine),
        name="Predictions",
    )

    # Test plot
    win.nextRow()
    test_plot = win.addPlot(title="Test Data")
    test_plot.setLabel("left", "Output")
    test_plot.setLabel("bottom", "Input")
    test_plot.addLegend()
    test_plot.showGrid(x=True, y=True, alpha=0.3)

    test_plot.plot(
        X_test_sorted, y_test_sorted, pen=pg.mkPen("b", width=2), name="True Values"
    )
    test_plot.plot(
        X_test_sorted,
        test_preds,
        pen=pg.mkPen("r", width=2, style=QtCore.Qt.PenStyle.DashLine),
        name="Predictions",
    )

    QtWidgets.QApplication.instance().exec()


def plot_combined_predictions(
    nn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    title: str = "Neural Network - Train & Test Predictions",
) -> None:
    """
    Plot combined train/test predictions with a smooth model curve.

    Parameters
    ----------
    nn : object
        Neural network implementing `predict`.
    X_train : np.ndarray
        Training input data.
    y_train : np.ndarray
        Training ground truth data.
    X_test : np.ndarray
        Test input data.
    y_test : np.ndarray
        Test ground truth data.
    title : str, optional
        Window title.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title=title)
    win.resize(1000, 700)
    win.setWindowTitle(title)

    plot = win.addPlot(title="Predictions vs Ground Truth")
    plot.setLabel("left", "Output Value")
    plot.setLabel("bottom", "Input Value")
    plot.addLegend()
    plot.showGrid(x=True, y=True, alpha=0.3)

    # Sort data
    X_train_flat = X_train.flatten()
    y_train_flat = y_train.flatten()
    train_sort_idx = np.argsort(X_train_flat)
    X_train_sorted = X_train_flat[train_sort_idx]
    y_train_sorted = y_train_flat[train_sort_idx]

    X_test_flat = X_test.flatten()
    y_test_flat = y_test.flatten()
    test_sort_idx = np.argsort(X_test_flat)
    X_test_sorted = X_test_flat[test_sort_idx]
    y_test_sorted = y_test_flat[test_sort_idx]

    # Smooth curve
    X_min = float(min(X_train_flat.min(), X_test_flat.min()))
    X_max = float(max(X_train_flat.max(), X_test_flat.max()))
    X_smooth = np.linspace(X_min, X_max, 500).reshape(-1, 1).astype(np.float32)
    y_smooth = np.array([_predict_scalar(nn, x.reshape(1, -1)) for x in X_smooth])

    plot.plot(
        X_smooth.flatten(), y_smooth, pen=pg.mkPen("g", width=3), name="NN Prediction"
    )
    plot.plot(
        X_train_sorted,
        y_train_sorted,
        pen=None,
        symbol="o",
        symbolSize=6,
        symbolBrush=(100, 100, 255, 150),
        name="Train Data",
    )
    plot.plot(
        X_test_sorted,
        y_test_sorted,
        pen=None,
        symbol="t",
        symbolSize=8,
        symbolBrush=(255, 100, 100, 200),
        name="Test Data",
    )

    QtWidgets.QApplication.instance().exec()


def plot_error_distribution(
    nn,
    X_test: np.ndarray,
    y_test: np.ndarray,
    title: str = "Prediction Error Distribution",
) -> None:
    """
    Visualize the distribution of prediction errors on test data.

    Parameters
    ----------
    nn : object
        Neural network implementing `predict`.
    X_test : np.ndarray
        Test input data.
    y_test : np.ndarray
        True test labels.
    title : str, optional
        Plot window title.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title=title)
    win.resize(800, 600)
    win.setWindowTitle(title)

    predictions = np.array([_predict_scalar(nn, x) for x in X_test])
    errors = predictions - y_test.flatten()

    plot = win.addPlot(title="Error Distribution")
    plot.setLabel("left", "Frequency")
    plot.setLabel("bottom", "Error")
    plot.showGrid(x=True, y=True, alpha=0.3)

    counts, bin_edges = np.histogram(errors, bins=30)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    plot.plot(
        bin_centers, counts, stepMode=False, fillLevel=0, brush=(100, 100, 255, 150)
    )

    text = pg.TextItem(
        f"Mean Error: {np.mean(errors):.4f}\n"
        f"Std Dev: {np.std(errors):.4f}\n"
        f"MAE: {np.mean(np.abs(errors)):.4f}",
        anchor=(0, 1),
    )
    text.setPos(bin_centers[0], counts.max() if counts.size else 0)
    plot.addItem(text)

    QtWidgets.QApplication.instance().exec()


def plot_error_vs_input(
    nn,
    X_test: np.ndarray,
    y_test: np.ndarray,
    title: str = "Error vs Input Value",
) -> None:
    """
    Plot prediction errors as a function of input values.

    Parameters
    ----------
    nn : object
        Neural network implementing `predict`.
    X_test : np.ndarray
        Input data.
    y_test : np.ndarray
        True output values.
    title : str, optional
        Window title.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title=title)
    win.resize(1000, 600)
    win.setWindowTitle(title)

    X_test_flat = X_test.flatten()
    y_test_flat = y_test.flatten()
    predictions = np.array([_predict_scalar(nn, x) for x in X_test])
    errors = predictions - y_test_flat

    sort_idx = np.argsort(X_test_flat)
    X_sorted = X_test_flat[sort_idx]
    errors_sorted = errors[sort_idx]

    plot = win.addPlot(title="Prediction Error vs Input")
    plot.setLabel("left", "Error (Pred - True)")
    plot.setLabel("bottom", "Input Value")
    plot.showGrid(x=True, y=True, alpha=0.3)

    plot.plot(X_sorted, errors_sorted, pen=pg.mkPen("b", width=2))
    plot.plot(
        [X_sorted.min(), X_sorted.max()],
        [0, 0],
        pen=pg.mkPen("r", width=2, style=QtCore.Qt.PenStyle.DashLine),
    )

    QtWidgets.QApplication.instance().exec()


def plot_all_metrics(
    nn,
    history: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    title: str = "Neural Network Complete Analysis",
) -> None:
    """
    Display all key plots for neural network analysis.

    Combines training loss, predictions, and error visualization into a single
    PyQtGraph window layout.

    Parameters
    ----------
    nn : object
        Neural network implementing `predict`.
    history : dict
        Training history containing `loss` and optionally `val_loss`.
    X_train : np.ndarray
        Training input data.
    y_train : np.ndarray
        Training target values.
    X_test : np.ndarray
        Test input data.
    y_test : np.ndarray
        Test target values.
    title : str, optional
        Window title.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title=title)
    win.resize(1400, 900)
    win.setWindowTitle(title)
    # --- First row: training history ---
    loss_plot = win.addPlot(title="Training Loss")
    loss_plot.setLabel("left", "Loss")
    loss_plot.setLabel("bottom", "Epoch")
    loss_plot.addLegend()
    loss_plot.showGrid(x=True, y=True, alpha=0.3)
    epochs = np.arange(len(history["loss"]))
    loss_plot.plot(
        epochs, history["loss"], pen=pg.mkPen("b", width=2), name="Train Loss"
    )
    if "val_loss" in history:
        loss_plot.plot(
            epochs, history["val_loss"], pen=pg.mkPen("r", width=2), name="Val Loss"
        )

    # --- Second plot: predictions ---
    pred_plot = win.addPlot(title="Predictions vs Ground Truth")
    pred_plot.setLabel("left", "Output")
    pred_plot.setLabel("bottom", "Input")
    pred_plot.addLegend()
    pred_plot.showGrid(x=True, y=True, alpha=0.3)

    X_train_flat = X_train.flatten()
    y_train_flat = y_train.flatten()
    train_sort_idx = np.argsort(X_train_flat)
    X_train_sorted = X_train_flat[train_sort_idx]
    y_train_sorted = y_train_flat[train_sort_idx]

    X_test_flat = X_test.flatten()
    y_test_flat = y_test.flatten()
    test_sort_idx = np.argsort(X_test_flat)
    X_test_sorted = X_test_flat[test_sort_idx]
    y_test_sorted = y_test_flat[test_sort_idx]

    X_min = float(min(X_train_flat.min(), X_test_flat.min()))
    X_max = float(max(X_train_flat.max(), X_test_flat.max()))
    X_smooth = np.linspace(X_min, X_max, 500).reshape(-1, 1).astype(np.float32)
    y_smooth = np.array([_predict_scalar(nn, x.reshape(1, -1)) for x in X_smooth])

    pred_plot.plot(
        X_smooth.flatten(), y_smooth, pen=pg.mkPen("g", width=3), name="Prediction"
    )
    pred_plot.plot(
        X_train_sorted,
        y_train_sorted,
        pen=None,
        symbol="o",
        symbolSize=5,
        symbolBrush=(100, 100, 255, 100),
        name="Train",
    )
    pred_plot.plot(
        X_test_sorted,
        y_test_sorted,
        pen=None,
        symbol="t",
        symbolSize=7,
        symbolBrush=(255, 100, 100, 150),
        name="Test",
    )

    # --- Third row: errors ---
    win.nextRow()
    error_plot = win.addPlot(title="Prediction Error Distribution")
    error_plot.setLabel("left", "Frequency")
    error_plot.setLabel("bottom", "Error")
    error_plot.showGrid(x=True, y=True, alpha=0.3)

    predictions = np.array([_predict_scalar(nn, x) for x in X_test])
    errors = predictions - y_test_flat
    counts, bin_edges = np.histogram(errors, bins=30)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    error_plot.plot(
        bin_centers, counts, stepMode=False, fillLevel=0, brush=(100, 255, 100, 150)
    )

    error_input_plot = win.addPlot(title="Error vs Input Value")
    error_input_plot.setLabel("left", "Error (Pred - True)")
    error_input_plot.setLabel("bottom", "Input Value")
    error_input_plot.showGrid(x=True, y=True, alpha=0.3)

    sort_idx = np.argsort(X_test_flat)
    X_test_sorted_err = X_test_flat[sort_idx]
    errors_sorted = errors[sort_idx]

    error_input_plot.plot(X_test_sorted_err, errors_sorted, pen=pg.mkPen("b", width=2))
    error_input_plot.plot(
        [X_test_sorted_err.min(), X_test_sorted_err.max()],
        [0, 0],
        pen=pg.mkPen("r", width=2, style=QtCore.Qt.PenStyle.DashLine),
    )

    QtWidgets.QApplication.instance().exec()


def visualize_nn_results(nn, history, X_train, y_train, X_test, y_test, mode="all"):
    """
    High-level visualization entry point for neural network analysis.

    Parameters
    ----------
    nn : object
        Neural network implementing `predict`.
    history : dict
        Dictionary containing training loss metrics.
    X_train : np.ndarray
        Training input data.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test input data.
    y_test : np.ndarray
        Test labels.
    mode : {'all', 'predictions', 'history', 'errors', 'separate'}, optional
        Visualization mode:
        - 'all': Display full analysis (loss, predictions, and errors)
        - 'predictions': Combined train/test prediction visualization
        - 'history': Training/validation loss
        - 'errors': Error histogram
        - 'separate': Train/test prediction comparison
    """
    if mode == "all":
        plot_all_metrics(nn, history, X_train, y_train, X_test, y_test)
    elif mode == "predictions":
        plot_combined_predictions(nn, X_train, y_train, X_test, y_test)
    elif mode == "history":
        plot_training_history(history)
    elif mode == "errors":
        plot_error_distribution(nn, X_test, y_test)
    elif mode == "separate":
        plot_predictions(nn, X_train, y_train, X_test, y_test)
    else:
        print(
            f"Unknown mode: {mode}. Use 'all', 'predictions', 'history', 'errors', or 'separate'."
        )
