"""
learning_rate_schedulers.py
===========================

A collection of learning rate scheduling strategies for neural network training.

This module defines a base class `LearningRateScheduler` and multiple subclasses
that implement various learning rate scheduling algorithms, including:
- Constant
- Linear decay
- Exponential decay
- Step decay
- Cosine annealing
- Cosine annealing with warm restarts
- Cyclic
- One cycle
- Polynomial decay
- Warmup scheduler

All schedulers follow a unified interface compatible with optimizers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


# --------------------------------------------------------------------------- #
# Base Class
# --------------------------------------------------------------------------- #
class LearningRateScheduler(ABC):
    """
    Abstract base class for learning rate schedules.

    Notes
    -----
    Subclasses must implement the :meth:`get_lr` method.

    Methods
    -------
    get_lr(epoch : int) -> float
        Get learning rate for a given epoch.
    """

    @abstractmethod
    def get_lr(self, epoch: int) -> float:
        """
        Get learning rate for given epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.

        Returns
        -------
        float
            Computed learning rate for the epoch.
        """
        pass

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({params})"


# --------------------------------------------------------------------------- #
# Constant Learning Rate
# --------------------------------------------------------------------------- #
class ConstantLR(LearningRateScheduler):
    """
    Constant learning rate scheduler.

    Parameters
    ----------
    lr : float
        Fixed learning rate value.
    """

    def __init__(self, lr: float) -> None:
        self.lr = lr

    def get_lr(self, epoch: int) -> float:
        return self.lr


# --------------------------------------------------------------------------- #
# Linear Decay
# --------------------------------------------------------------------------- #
class LinearDecayLR(LearningRateScheduler):
    """
    Linearly decays learning rate from `start_lr` to `end_lr` over epochs.

    Parameters
    ----------
    start_lr : float
        Initial learning rate.
    end_lr : float
        Final learning rate.
    epochs : int
        Total number of epochs for decay.
    """

    def __init__(self, start_lr: float, end_lr: float, epochs: int) -> None:
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.epochs = epochs

    def get_lr(self, epoch: int) -> float:
        if epoch >= self.epochs:
            return self.end_lr
        ratio = epoch / self.epochs
        return self.start_lr - (self.start_lr - self.end_lr) * ratio


# --------------------------------------------------------------------------- #
# Exponential Decay
# --------------------------------------------------------------------------- #
class ExponentialDecayLR(LearningRateScheduler):
    """
    Exponentially decays learning rate:
    :math:`lr = initial_lr * (decay_rate ^ epoch)`

    Parameters
    ----------
    initial_lr : float
        Initial learning rate.
    decay_rate : float
        Decay rate per epoch (typically 0.9–0.99).
    """

    def __init__(self, initial_lr: float, decay_rate: float) -> None:
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate

    def get_lr(self, epoch: int) -> float:
        return self.initial_lr * (self.decay_rate**epoch)


# --------------------------------------------------------------------------- #
# Step Decay
# --------------------------------------------------------------------------- #
class StepDecayLR(LearningRateScheduler):
    """
    Step decay scheduler.

    Reduces learning rate by `drop_factor` every `step_size` epochs.

    Parameters
    ----------
    initial_lr : float
        Initial learning rate.
    drop_factor : float, default=0.5
        Factor to multiply the learning rate by at each step.
    step_size : int, default=1000
        Number of epochs between drops.
    """

    def __init__(
        self, initial_lr: float, drop_factor: float = 0.5, step_size: int = 1000
    ) -> None:
        self.initial_lr = initial_lr
        self.drop_factor = drop_factor
        self.step_size = step_size

    def get_lr(self, epoch: int) -> float:
        return self.initial_lr * (self.drop_factor ** (epoch // self.step_size))


# --------------------------------------------------------------------------- #
# Cosine Annealing
# --------------------------------------------------------------------------- #
class CosineAnnealingLR(LearningRateScheduler):
    """
    Cosine annealing scheduler.

    Learning rate follows a cosine curve between `max_lr` and `min_lr`.

    Parameters
    ----------
    max_lr : float
        Maximum learning rate.
    min_lr : float
        Minimum learning rate.
    T_max : int
        Number of epochs for one complete cycle.
    """

    def __init__(self, max_lr: float, min_lr: float, T_max: int) -> None:
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.T_max = T_max

    def get_lr(self, epoch: int) -> float:
        cos_inner = np.pi * (epoch % self.T_max) / self.T_max
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(cos_inner))


# --------------------------------------------------------------------------- #
# Cosine Annealing with Warm Restarts
# --------------------------------------------------------------------------- #
class CosineAnnealingWarmRestartsLR(LearningRateScheduler):
    """
    Cosine annealing with warm restarts (SGDR).

    Parameters
    ----------
    max_lr : float
        Maximum learning rate.
    min_lr : float
        Minimum learning rate.
    T_0 : int
        Initial restart period.
    T_mult : int, default=2
        Factor to multiply the period after each restart.
    """

    def __init__(self, max_lr: float, min_lr: float, T_0: int, T_mult: int = 2) -> None:
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.T_0 = T_0
        self.T_mult = T_mult

    def get_lr(self, epoch: int) -> float:
        T_cur, T_i = epoch, self.T_0
        while T_cur >= T_i:
            T_cur -= T_i
            T_i *= self.T_mult

        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1 + np.cos(np.pi * T_cur / T_i)
        )


# --------------------------------------------------------------------------- #
# Cyclic LR
# --------------------------------------------------------------------------- #
class CyclicLR(LearningRateScheduler):
    """
    Cyclic learning rate scheduler.

    Oscillates between `min_lr` and `max_lr` using a triangular or exponential mode.

    Parameters
    ----------
    min_lr : float
        Minimum learning rate.
    max_lr : float
        Maximum learning rate.
    step_size : int
        Half-period of a cycle (in epochs).
    mode : {'triangular', 'triangular2', 'exp_range'}, default='triangular'
        Cycle mode.
    gamma : float, default=0.999
        Decay factor for `exp_range` mode.
    """

    def __init__(
        self,
        min_lr: float,
        max_lr: float,
        step_size: int,
        mode: str = "triangular",
        gamma: float = 0.999,
    ) -> None:
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma

    def get_lr(self, epoch: int) -> float:
        cycle = np.floor(1 + epoch / (2 * self.step_size))
        x = abs(epoch / self.step_size - 2 * cycle + 1)

        if self.mode == "triangular":
            scale_factor = 1.0
        elif self.mode == "triangular2":
            scale_factor = 1.0 / (2.0 ** (cycle - 1))
        elif self.mode == "exp_range":
            scale_factor = self.gamma**epoch
        else:
            scale_factor = 1.0

        return (
            self.min_lr + (self.max_lr - self.min_lr) * max(0, (1 - x)) * scale_factor
        )


# --------------------------------------------------------------------------- #
# One Cycle LR
# --------------------------------------------------------------------------- #
class OneCycleLR(LearningRateScheduler):
    """
    One cycle learning rate policy.

    Increases learning rate to `max_lr` and then decreases it, following a single cycle.

    Parameters
    ----------
    max_lr : float
        Maximum learning rate.
    total_epochs : int
        Total number of epochs in the cycle.
    pct_start : float, default=0.3
        Percentage of cycle spent increasing learning rate.
    div_factor : float, default=25.0
        Determines initial lr = max_lr / div_factor.
    final_div_factor : float, default=10000.0
        Determines final lr = initial_lr / final_div_factor.
    """

    def __init__(
        self,
        max_lr: float,
        total_epochs: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 10000.0,
    ) -> None:
        self.max_lr = max_lr
        self.total_epochs = total_epochs
        self.pct_start = pct_start
        self.initial_lr = max_lr / div_factor
        self.final_lr = self.initial_lr / final_div_factor

    def get_lr(self, epoch: int) -> float:
        if epoch < self.pct_start * self.total_epochs:
            pct = epoch / (self.pct_start * self.total_epochs)
            return self.initial_lr + (self.max_lr - self.initial_lr) * pct
        pct = (epoch - self.pct_start * self.total_epochs) / (
            (1 - self.pct_start) * self.total_epochs
        )
        return self.max_lr - (self.max_lr - self.final_lr) * pct


# --------------------------------------------------------------------------- #
# Polynomial Decay
# --------------------------------------------------------------------------- #
class PolynomialDecayLR(LearningRateScheduler):
    """
    Polynomial decay scheduler.

    :math:`lr = (initial_lr - end_lr) * (1 - epoch/total_epochs)^power + end_lr`

    Parameters
    ----------
    initial_lr : float
        Initial learning rate.
    end_lr : float
        Final learning rate.
    total_epochs : int
        Total number of epochs.
    power : float, default=2.0
        Polynomial power.
    """

    def __init__(
        self, initial_lr: float, end_lr: float, total_epochs: int, power: float = 2.0
    ) -> None:
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.total_epochs = total_epochs
        self.power = power

    def get_lr(self, epoch: int) -> float:
        if epoch >= self.total_epochs:
            return self.end_lr
        decay = (1.0 - epoch / self.total_epochs) ** self.power
        return (self.initial_lr - self.end_lr) * decay + self.end_lr


# --------------------------------------------------------------------------- #
# Warmup Scheduler
# --------------------------------------------------------------------------- #
class WarmupLR(LearningRateScheduler):
    """
    Warmup scheduler.

    Gradually increases the learning rate for a few epochs, then delegates to
    another scheduler.

    Parameters
    ----------
    base_scheduler : LearningRateScheduler
        Scheduler to use after warmup.
    warmup_epochs : int
        Number of warmup epochs.
    warmup_start_lr : float, default=1e-6
        Starting learning rate for warmup.
    """

    def __init__(
        self,
        base_scheduler: LearningRateScheduler,
        warmup_epochs: int,
        warmup_start_lr: float = 1e-6,
    ) -> None:
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.target_lr = base_scheduler.get_lr(0)

    def get_lr(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.warmup_start_lr + (self.target_lr - self.warmup_start_lr) * (
                epoch / self.warmup_epochs
            )
        return self.base_scheduler.get_lr(epoch - self.warmup_epochs)
