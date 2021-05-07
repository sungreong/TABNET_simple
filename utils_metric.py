from dataclasses import dataclass
from typing import List
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    log_loss,
    balanced_accuracy_score,
    mean_squared_log_error,
)
import torch

def UnsupervisedLoss(y_pred, embedded_x, obf_vars, eps=1e-9):
    """
    Implements unsupervised loss function.
    This differs from orginal paper as it's scaled to be batch size independent
    and number of features reconstructed independent (by taking the mean)
    Parameters
    ----------
    y_pred : torch.Tensor or np.array
        Reconstructed prediction (with embeddings)
    embedded_x : torch.Tensor
        Original input embedded by network
    obf_vars : torch.Tensor
        Binary mask for obfuscated variables.
        1 means the variable was obfuscated so reconstruction is based on this.
    eps : float
        A small floating point to avoid ZeroDivisionError
        This can happen in degenerated case when a feature has only one value
    Returns
    -------
    loss : torch float
        Unsupervised loss, average value over batch samples.
    """
    errors = y_pred - embedded_x
    reconstruction_errors = torch.mul(errors, obf_vars) ** 2
    batch_stds = torch.std(embedded_x, dim=0) ** 2 + eps
    features_loss = torch.matmul(reconstruction_errors, 1 / batch_stds)
    # compute the number of obfuscated variables to reconstruct
    nb_reconstructed_variables = torch.sum(obf_vars, dim=1)
    # take the mean of the reconstructed variable errors
    features_loss = features_loss / (nb_reconstructed_variables + eps)
    # here we take the mean per batch, contrary to the paper
    loss = torch.mean(features_loss)
    return loss


@dataclass
class UnsupMetricContainer:
    """Container holding a list of metrics.
    Parameters
    ----------
    y_pred : torch.Tensor or np.array
        Reconstructed prediction (with embeddings)
    embedded_x : torch.Tensor
        Original input embedded by network
    obf_vars : torch.Tensor
        Binary mask for obfuscated variables.
        1 means the variables was obfuscated so reconstruction is based on this.
    """

    metric_names: List[str]
    prefix: str = ""

    def __post_init__(self):
        self.metrics = Metric.get_metrics_by_names(self.metric_names)
        self.names = [self.prefix + name for name in self.metric_names]

    def __call__(self, y_pred, embedded_x, obf_vars):
        """Compute all metrics and store into a dict.
        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_pred : np.ndarray
            Score matrix or vector
        Returns
        -------
        dict
            Dict of metrics ({metric_name: metric_value}).
        """
        logs = {}
        for metric in self.metrics:
            res = metric(y_pred, embedded_x, obf_vars)
            logs[self.prefix + metric._name] = res
        return logs


@dataclass
class MetricContainer:
    """Container holding a list of metrics.
    Parameters
    ----------
    metric_names : list of str
        List of metric names.
    prefix : str
        Prefix of metric names.
    """

    metric_names: List[str]
    prefix: str = ""

    def __post_init__(self):
        self.metrics = Metric.get_metrics_by_names(self.metric_names)
        self.names = [self.prefix + name for name in self.metric_names]

    def __call__(self, y_true, y_pred):
        """Compute all metrics and store into a dict.
        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_pred : np.ndarray
            Score matrix or vector
        Returns
        -------
        dict
            Dict of metrics ({metric_name: metric_value}).
        """
        logs = {}
        for metric in self.metrics:
            if isinstance(y_pred, list):
                res = np.mean(
                    [metric(y_true[:, i], y_pred[i]) for i in range(len(y_pred))]
                )
            else:
                res = metric(y_true, y_pred)
            logs[self.prefix + metric._name] = res
        return logs
