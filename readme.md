# Neural Network Framework

A modular, lightweight feedforward neural network implementation built with NumPy, featuring custom layers, activation functions, and advanced training utilities.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
  - [Core Classes](#core-classes)
  - [NeuralNetwork API](#neuralnetwork-api)
  - [Layer Types](#layer-types)
  - [Loss Functions](#loss-functions)
  - [Learning Rate Schedulers](#learning-rate-schedulers)
- [Usage Examples](#usage-examples)
- [Architecture Details](#architecture-details)
- [Advanced Topics](#advanced-topics)
- [Contributing](#contributing)
- [License](#license)

## Overview

This neural network framework provides a flexible, educational implementation of feedforward neural networks with a focus on clarity and extensibility. It supports arbitrary network architectures through layer composition and includes modern training techniques such as learning rate scheduling and early stopping.

## Features

**Core Functionality**
- Sequential layer stacking with arbitrary depth
- Multiple built-in activation functions (ReLU, Sigmoid, Tanh, Swish)
- Pluggable loss function architecture
- Batch-free training with sample-by-sample gradient updates

**Training Utilities**
- Learning rate scheduling (Constant, OneCycleLR, custom schedulers)
- Validation data tracking during training
- Early stopping with configurable patience
- Training history logging
- Verbose training progress with configurable intervals

**Evaluation Metrics**
- Mean Squared Error (MSE) loss
- Classification accuracy for binary outputs
- Regression accuracy metrics
- Custom loss function support

## Requirements

### Python Version
- Python 3.8 or higher

### Dependencies
- NumPy >= 1.20.0

### Project Modules
The following modules must be present in your project directory:
- `ActivationFunction.py` - Activation function implementations
- `ErrorFunctions.py` - Loss function definitions
- `ErrorLayer.py` - Loss function base class
- `Layer.py` - Base layer class
- `LearnLayer.py` - Learning rate scheduler implementations
- `Visualise.py` - Visualization utilities

## Installation

### Step 1: Install NumPy

```bash
pip install numpy
```

### Step 2: Clone or Download Project Files

Ensure all project modules are in the same directory as your main script.

### Step 3: Import and Use

```python
from NeuralNetwork import NeuralNetwork
from ActivationFunction import ReLU, Sigmoid
from ErrorFunctions import MSE
```

## Quick Start

```python
import numpy as np
from NeuralNetwork import NeuralNetwork
from ActivationFunction import ReLU, Sigmoid
from ErrorFunctions import MSE

# Initialize network
nn = NeuralNetwork(loss_function=MSE())

# Build architecture
nn.add(ReLU(input_size=1, output_size=64))
nn.add(ReLU(input_size=64, output_size=32))
nn.add(Sigmoid(input_size=32, output_size=1))

# Prepare training data
X_train = np.random.randn(1000, 1, 1).astype(np.float32)
y_train = np.random.randn(1000, 1, 1).astype(np.float32)

# Train the network
history = nn.train(
    X_train, 
    y_train,
    epochs=1000,
    learning_rate=0.01,
    verbose=True,
    verbose_period=100
)

# Make predictions
predictions = nn.predict(X_train)

# Evaluate performance
test_loss = nn.evaluate(X_train, y_train)
accuracy = nn.get_mse_accuracy(X_train, y_train)

print(f"Test Loss: {test_loss:.6f}")
print(f"Accuracy: {accuracy:.2f}%")
```

## Documentation

### Core Classes

#### NeuralNetwork

The main class for creating and managing neural networks.

**Initialization**

```python
NeuralNetwork(loss_function=None)
```

**Parameters:**
- `loss_function` (LossFunction, optional): Loss function for training. If `None`, defaults to `MSE()`.

**Attributes:**
- `layers` (list[Layer]): Sequential list of network layers
- `loss_function` (LossFunction): Active loss function

**Example:**

```python
from ErrorFunctions import MSE
nn = NeuralNetwork(loss_function=MSE())
```

---

### NeuralNetwork API

#### add

Add one or more layers to the network.

```python
add(layer: Layer | Sequence[Layer]) -> None
```

**Parameters:**
- `layer`: Single `Layer` instance or sequence of `Layer` instances

**Usage:**

```python
# Single layer
nn.add(ReLU(10, 20))

# Multiple layers
nn.add([
    ReLU(10, 20),
    Sigmoid(20, 10),
    Tanh(10, 1)
])
```

---

#### predict

Perform forward propagation through the network.

```python
predict(X: np.ndarray) -> np.ndarray
```

**Parameters:**
- `X` (np.ndarray): Input data with shape `(n_samples, n_features)` or `(n_features,)`

**Returns:**
- `np.ndarray`: Network output after passing through all layers

**Example:**

```python
X_test = np.array([[0.5], [1.0], [1.5]]).reshape(-1, 1, 1)
predictions = nn.predict(X_test)
```

---

#### train

Train the neural network using backpropagation.

```python
train(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    learning_rate: LearningRateScheduler | float = 0.01,
    verbose: bool = True,
    verbose_period: int = 1000,
    validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    early_stopping_patience: int | None = None,
    min_delta: float = 1e-7
) -> dict
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | np.ndarray | Required | Training input samples |
| `y` | np.ndarray | Required | Target output values |
| `epochs` | int | Required | Number of training epochs |
| `learning_rate` | float or LearningRateScheduler | 0.01 | Learning rate or scheduler instance |
| `verbose` | bool | True | Whether to print training progress |
| `verbose_period` | int | 1000 | Number of epochs between progress updates |
| `validation_data` | tuple or None | None | Tuple of (X_val, y_val) for validation |
| `early_stopping_patience` | int or None | None | Stop after N epochs without improvement |
| `min_delta` | float | 1e-7 | Minimum change to qualify as improvement |

**Returns:**
- `dict`: Training history containing:
  - `'loss'` (list[float]): Training loss per epoch
  - `'val_loss'` (list[float]): Validation loss per epoch (if validation data provided)

**Example:**

```python
history = nn.train(
    X_train, y_train,
    epochs=5000,
    learning_rate=0.01,
    verbose=True,
    verbose_period=500,
    validation_data=(X_val, y_val),
    early_stopping_patience=1000,
    min_delta=1e-6
)

# Access training history
train_losses = history['loss']
val_losses = history['val_loss']
```

---

#### evaluate

Calculate the average loss on a dataset.

```python
evaluate(X: np.ndarray, y: np.ndarray) -> float
```

**Parameters:**
- `X` (np.ndarray): Input samples
- `y` (np.ndarray): True target values

**Returns:**
- `float`: Mean loss across all samples

**Example:**

```python
test_loss = nn.evaluate(X_test, y_test)
train_loss = nn.evaluate(X_train, y_train)
print(f"Test Loss: {test_loss:.6f}, Train Loss: {train_loss:.6f}")
```

---

#### get_accuracy

Compute classification accuracy for binary outputs.

```python
get_accuracy(X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> float
```

**Parameters:**
- `X` (np.ndarray): Input samples
- `y` (np.ndarray): True binary labels (0 or 1)
- `threshold` (float): Decision boundary for classification (default: 0.5)

**Returns:**
- `float`: Accuracy as a decimal between 0.0 and 1.0

**Example:**

```python
accuracy = nn.get_accuracy(X_test, y_test, threshold=0.5)
print(f"Classification Accuracy: {accuracy * 100:.2f}%")
```

---

#### get_mse_accuracy

Compute regression accuracy based on MSE.

```python
get_mse_accuracy(X: np.ndarray, y: np.ndarray) -> float
```

**Parameters:**
- `X` (np.ndarray): Input data
- `y` (np.ndarray): True target values

**Returns:**
- `float`: Accuracy percentage (0-100), calculated as `max(0, 100 * (1 - MSE))`

**Example:**

```python
accuracy = nn.get_mse_accuracy(X_test, y_test)
print(f"MSE-based Accuracy: {accuracy:.2f}%")
```

---

### Layer Types

Layers must inherit from the `Layer` base class and implement `forward()` and `backward()` methods.

**Built-in Activation Layers:**
- `ReLU(input_size, output_size)` - Rectified Linear Unit
- `Sigmoid(input_size, output_size)` - Sigmoid activation
- `Tanh(input_size, output_size)` - Hyperbolic tangent
- `Swish(input_size, output_size)` - Self-gated activation

**Example:**

```python
from ActivationFunction import ReLU, Sigmoid, Tanh, Swish

nn = NeuralNetwork()
nn.add(ReLU(1, 128))
nn.add(Swish(128, 64))
nn.add(Tanh(64, 32))
nn.add(Sigmoid(32, 1))
```

---

### Loss Functions

Loss functions must implement the `LossFunction` interface with `forward()` and `backward()` methods.

**Built-in Loss Functions:**
- `MSE()` - Mean Squared Error

**Custom Loss Function Example:**

```python
from ErrorLayer import LossFunction

class CustomLoss(LossFunction):
    def forward(self, y_true, y_pred):
        # Compute loss
        return loss_value
    
    def backward(self, y_true, y_pred):
        # Compute gradient
        return gradient

nn = NeuralNetwork(loss_function=CustomLoss())
```

---

### Learning Rate Schedulers

Schedulers must inherit from `LearningRateScheduler` and implement `get_lr(epoch)`.

**Built-in Schedulers:**

**ConstantLR**
```python
from LearnLayer import ConstantLR
lr = ConstantLR(learning_rate=0.01)
```

**OneCycleLR**
```python
from LearnLayer import OneCycleLR
lr = OneCycleLR(
    max_lr=0.05,
    total_epochs=10000,
    pct_start=0.3  # Percentage of epochs for warm-up
)
```

**Custom Scheduler Example:**

```python
from LearnLayer import LearningRateScheduler

class ExponentialDecayLR(LearningRateScheduler):
    def __init__(self, initial_lr=0.1, decay_rate=0.95):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
    
    def get_lr(self, epoch):
        return self.initial_lr * (self.decay_rate ** epoch)

lr_schedule = ExponentialDecayLR(initial_lr=0.1, decay_rate=0.95)
history = nn.train(X, y, epochs=1000, learning_rate=lr_schedule)
```

---

## Usage Examples

### Example 1: Basic Regression

Train a neural network to approximate a simple function.

```python
import numpy as np
from NeuralNetwork import NeuralNetwork
from ActivationFunction import ReLU, Sigmoid
from ErrorFunctions import MSE

# Generate synthetic data
np.random.seed(42)
X = np.linspace(-5, 5, 500).reshape(-1, 1, 1).astype(np.float32)
y = (np.sin(X) * 0.5 + 0.5).astype(np.float32)

# Build network
nn = NeuralNetwork(loss_function=MSE())
nn.add(ReLU(1, 32))
nn.add(ReLU(32, 16))
nn.add(Sigmoid(16, 1))

# Train
history = nn.train(X, y, epochs=1000, learning_rate=0.01, verbose_period=200)

# Evaluate
accuracy = nn.get_mse_accuracy(X, y)
print(f"Final Accuracy: {accuracy:.2f}%")
```

---

### Example 2: Using Learning Rate Schedulers

Leverage OneCycleLR for better convergence.

```python
from LearnLayer import OneCycleLR

# Create scheduler
lr_schedule = OneCycleLR(
    max_lr=0.05,
    total_epochs=5000,
    pct_start=0.3
)

# Train with scheduler
history = nn.train(
    X_train, y_train,
    epochs=5000,
    learning_rate=lr_schedule,
    verbose_period=500
)
```

---

### Example 3: Validation and Early Stopping

Prevent overfitting with validation monitoring.

```python
# Split data
split_idx = int(0.8 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# Train with early stopping
history = nn.train(
    X_train, y_train,
    epochs=10000,
    learning_rate=0.01,
    validation_data=(X_val, y_val),
    early_stopping_patience=1000,
    min_delta=1e-6,
    verbose_period=500
)

# Plot training curves
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(history['loss'], label='Training Loss', alpha=0.7)
plt.plot(history['val_loss'], label='Validation Loss', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History')
plt.grid(True, alpha=0.3)
plt.show()
```

---

### Example 4: Complex Function Approximation

Full example demonstrating advanced features.

```python
import numpy as np
from NeuralNetwork import NeuralNetwork
from ActivationFunction import ReLU, Swish, Tanh, Sigmoid
from ErrorFunctions import MSE
from LearnLayer import OneCycleLR

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("Neural Network Training - Complex Sine Wave Approximation")
print("=" * 70)

# Generate complex synthetic data
X = np.linspace(-np.pi, np.pi, 2000).reshape(-1, 1, 1).astype(np.float32)
y = np.tanh(3 * np.sin(X**2) + np.cos(5 * X)) + np.sign(np.sin(7 * X))
y = (y + 1.0) / 2.0  # Normalize to [0, 1]

# Shuffle data
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print("=" * 70)

# Build deep network
nn = NeuralNetwork(loss_function=MSE())
nn.add(ReLU(1, 64))
nn.add(Swish(64, 128))
nn.add(Tanh(128, 64))
nn.add(Sigmoid(64, 1))

# Configure learning rate schedule
lr_schedule = OneCycleLR(
    max_lr=0.05,
    total_epochs=10000,
    pct_start=0.3
)

# Train with all features
history = nn.train(
    X_train, y_train,
    epochs=10000,
    learning_rate=lr_schedule,
    verbose=True,
    verbose_period=500,
    validation_data=(X_test, y_test),
    early_stopping_patience=1000,
    min_delta=1e-6
)

# Evaluate final performance
train_loss = nn.evaluate(X_train, y_train)
test_loss = nn.evaluate(X_test, y_test)
train_acc = nn.get_mse_accuracy(X_train, y_train)
test_acc = nn.get_mse_accuracy(X_test, y_test)

print("=" * 70)
print(f"Final Training Loss: {train_loss:.6f}")
print(f"Final Test Loss: {test_loss:.6f}")
print(f"Training Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")
print("=" * 70)

# Display sample predictions
print("\nSample Predictions:")
print("-" * 70)
print(f"{'Input':>10} | {'Prediction':>12} | {'True Value':>12} | {'Error':>10}")
print("-" * 70)

for i in range(0, len(X_test), len(X_test) // 5):
    x_val = X_test[i]
    pred = nn.predict(x_val)
    true_val = y_test[i]
    error = abs(pred[0, 0] - true_val[0, 0])
    print(f"{x_val[0, 0]:10.4f} | {pred[0, 0]:12.6f} | {true_val[0, 0]:12.6f} | {error:10.6f}")

print("-" * 70)
```

---

### Example 5: Binary Classification

Classify data into two classes.

```python
import numpy as np
from NeuralNetwork import NeuralNetwork
from ActivationFunction import ReLU, Sigmoid
from ErrorFunctions import MSE

# Generate binary classification data
np.random.seed(42)
n_samples = 1000

# Class 0: centered at (-2, -2)
X_class0 = np.random.randn(n_samples // 2, 2) * 0.5 + np.array([-2, -2])
y_class0 = np.zeros((n_samples // 2, 1))

# Class 1: centered at (2, 2)
X_class1 = np.random.randn(n_samples // 2, 2) * 0.5 + np.array([2, 2])
y_class1 = np.ones((n_samples // 2, 1))

# Combine and reshape
X = np.vstack([X_class0, X_class1]).reshape(-1, 2, 1).astype(np.float32)
y = np.vstack([y_class0, y_class1]).reshape(-1, 1, 1).astype(np.float32)

# Shuffle
indices = np.random.permutation(len(X))
X, y = X[indices], y[indices]

# Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build classifier
nn = NeuralNetwork(loss_function=MSE())
nn.add(ReLU(2, 16))
nn.add(ReLU(16, 8))
nn.add(Sigmoid(8, 1))

# Train
history = nn.train(
    X_train, y_train,
    epochs=2000,
    learning_rate=0.01,
    verbose=True,
    verbose_period=200
)

# Evaluate
accuracy = nn.get_accuracy(X_test, y_test, threshold=0.5)
print(f"Classification Accuracy: {accuracy * 100:.2f}%")
```

---

## Architecture Details

### Network Structure

The neural network implements a sequential feedforward architecture:

```
Input Layer → Hidden Layer 1 → ... → Hidden Layer N → Output Layer
```

Each layer performs:
1. Linear transformation: `z = Wx + b`
2. Activation function: `a = f(z)`

### Forward Propagation

Data flows sequentially through each layer:

```python
def predict(self, X):
    output = X
    for layer in self.layers:
        output = layer.forward(output)
    return output
```

### Backward Propagation

Gradients are computed in reverse order:

```python
gradient = loss_function.backward(y_true, y_pred)
for layer in reversed(self.layers):
    gradient = layer.backward(gradient, learning_rate)
```

### Training Loop

The training process for each epoch:

1. **Forward pass**: Compute predictions for each sample
2. **Loss calculation**: Measure prediction error
3. **Backward pass**: Compute gradients via backpropagation
4. **Parameter update**: Adjust weights using gradients and learning rate
5. **Validation**: Optionally evaluate on validation set
6. **Early stopping check**: Monitor for convergence

### Memory Considerations

This implementation uses sample-by-sample training rather than batch processing:
- Lower memory footprint
- More frequent weight updates
- Potentially slower training for large datasets

For large-scale applications, consider implementing mini-batch processing.

---

## Advanced Topics

### Custom Layer Implementation

Create custom layers by inheriting from the `Layer` base class:

```python
from Layer import Layer
import numpy as np

class CustomLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.input = None
    
    def forward(self, input_data):
        self.input = input_data
        return np.dot(input_data, self.weights) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        input_gradient = np.dot(output_gradient, self.weights.T)
        weights_gradient = np.dot(self.input.T, output_gradient)
        
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        
        return input_gradient
```

### Implementing Dropout

Add regularization through dropout:

```python
class Dropout(Layer):
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True
    
    def forward(self, input_data):
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, 
                                          size=input_data.shape)
            return input_data * self.mask / (1 - self.dropout_rate)
        return input_data
    
    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.mask / (1 - self.dropout_rate)
```

### Weight Initialization Strategies

Improve convergence with proper initialization:

```python
# Xavier/Glorot initialization
def xavier_init(input_size, output_size):
    limit = np.sqrt(6 / (input_size + output_size))
    return np.random.uniform(-limit, limit, (input_size, output_size))

# He initialization (for ReLU)
def he_init(input_size, output_size):
    std = np.sqrt(2 / input_size)
    return np.random.randn(input_size, output_size) * std
```

### Saving and Loading Models

Implement model persistence:

```python
import pickle

# Save model
def save_model(nn, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(nn, f)

# Load model
def load_model(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# Usage
save_model(nn, 'trained_model.pkl')
loaded_nn = load_model('trained_model.pkl')
```

### Hyperparameter Tuning

Systematic approach to finding optimal parameters:

```python
def grid_search(X_train, y_train, X_val, y_val):
    learning_rates = [0.001, 0.01, 0.1]
    architectures = [
        [32, 16],
        [64, 32, 16],
        [128, 64, 32]
    ]
    
    best_accuracy = 0
    best_params = None
    
    for lr in learning_rates:
        for arch in architectures:
            nn = NeuralNetwork()
            
            # Build architecture
            prev_size = X_train.shape[1]
            for size in arch:
                nn.add(ReLU(prev_size, size))
                prev_size = size
            nn.add(Sigmoid(prev_size, 1))
            
            # Train
            nn.train(X_train, y_train, epochs=1000, 
                    learning_rate=lr, verbose=False)
            
            # Evaluate
            accuracy = nn.get_mse_accuracy(X_val, y_val)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'lr': lr, 'architecture': arch}
    
    return best_params, best_accuracy
```

---

## Contributing

Contributions are welcome and encouraged. To contribute:

### Setting Up Development Environment

1. Fork the repository
2. Clone your fork locally
3. Install development dependencies
4. Create a new branch for your feature

### Code Style Guidelines

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Include docstrings for all public methods
- Write unit tests for new features

### Submitting Changes

1. Ensure all tests pass
2. Update documentation as needed
3. Submit a pull request with a clear description of changes
4. Reference any related issues

### Areas for Contribution

- Additional activation functions
- New loss functions
- Batch processing support
- GPU acceleration
- Advanced optimizers (Adam, RMSprop)
- Regularization techniques
- More comprehensive unit tests
- Performance optimizations

---

## License

MIT License

Copyright (c) 2024 Satyendhran

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Acknowledgments

This project was developed as an educational implementation of neural networks, emphasizing clarity and extensibility over performance optimization.

## Contact and Support

For questions, issues, or feature requests:
- Open an issue in the project repository
- Refer to the documentation
- Check existing issues for solutions

## Version History

**Version 1.0.0**
- Initial release
- Core feedforward network implementation
- Basic activation functions
- MSE loss function
- Learning rate scheduling
- Early stopping support
- Training history tracking

---

For more detailed information about specific components, refer to the inline documentation in the source code.
