"""
Metrics Module for Surgical QA Evaluation

Implements evaluation metrics from paper1231:
- Spearman Rank Correlation Coefficient (SRCC)
- Mean Absolute Error (MAE)
- Pearson Correlation Coefficient (PCC)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
from scipy import stats


def compute_metrics(y_pred, y_gt, verbose=True):
    """
    Compute all evaluation metrics.

    Args:
        y_pred: Predicted scores (numpy array or tensor)
        y_gt: Ground truth scores (numpy array or tensor)
        verbose: Print metrics

    Returns:
        metrics: Dict containing all metrics
    """
    # Convert to numpy
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(y_gt):
        y_gt = y_gt.detach().cpu().numpy()

    # Flatten if needed
    y_pred = y_pred.flatten()
    y_gt = y_gt.flatten()

    # Compute metrics
    metrics = {}

    # Mean Absolute Error (MAE)
    metrics['mae'] = compute_mae(y_pred, y_gt)

    # Mean Squared Error (MSE)
    metrics['mse'] = compute_mse(y_pred, y_gt)

    # Root Mean Squared Error (RMSE)
    metrics['rmse'] = np.sqrt(metrics['mse'])

    # Spearman Rank Correlation Coefficient (SRCC)
    metrics['srcc'], srcc_pvalue = compute_srcc(y_pred, y_gt)
    metrics['srcc_pvalue'] = srcc_pvalue

    # Pearson Correlation Coefficient (PCC)
    metrics['pcc'], pcc_pvalue = compute_pcc(y_pred, y_gt)
    metrics['pcc_pvalue'] = pcc_pvalue

    # Additional metrics
    metrics['mean_pred'] = np.mean(y_pred)
    metrics['mean_gt'] = np.mean(y_gt)
    metrics['std_pred'] = np.std(y_pred)
    metrics['std_gt'] = np.std(y_gt)

    # Normalized MAE (as percentage of score range)
    score_range = y_gt.max() - y_gt.min()
    if score_range > 0:
        metrics['nmae'] = metrics['mae'] / score_range * 100
    else:
        metrics['nmae'] = 0.0

    # Print metrics
    if verbose:
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  NMAE: {metrics['nmae']:.2f}% (normalized)")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  SRCC: {metrics['srcc']:.4f} (p={metrics['srcc_pvalue']:.4f})")
        print(f"  PCC:  {metrics['pcc']:.4f} (p={metrics['pcc_pvalue']:.4f})")
        print("="*60)
        print(f"  Mean Pred: {metrics['mean_pred']:.4f}, Mean GT: {metrics['mean_gt']:.4f}")
        print(f"  Std Pred:  {metrics['std_pred']:.4f}, Std GT:  {metrics['std_gt']:.4f}")

    return metrics


def compute_mae(y_pred, y_gt):
    """
    Mean Absolute Error.

    From paper1231:
    MAE = (1/n) * sum(|y_i - y_gt_i|)

    Args:
        y_pred: Predicted scores
        y_gt: Ground truth scores

    Returns:
        mae: Mean absolute error
    """
    return np.mean(np.abs(y_pred - y_gt))


def compute_mse(y_pred, y_gt):
    """
    Mean Squared Error.

    Args:
        y_pred: Predicted scores
        y_gt: Ground truth scores

    Returns:
        mse: Mean squared error
    """
    return np.mean((y_pred - y_gt) ** 2)


def compute_srcc(y_pred, y_gt):
    """
    Spearman Rank Correlation Coefficient.

    From paper1231:
    SRCC = sum(p_i - p_avg)(q_i - q_avg) /
            sqrt(sum(p_i - p_avg)^2 * sum(q_i - q_avg)^2)

    Args:
        y_pred: Predicted scores
        y_gt: Ground truth scores

    Returns:
        srcc: Spearman correlation coefficient [-1, 1]
        pvalue: Two-tailed p-value
    """
    srcc, pvalue = stats.spearmanr(y_pred, y_gt)
    return srcc, pvalue


def compute_pcc(y_pred, y_gt):
    """
    Pearson Correlation Coefficient.

    Args:
        y_pred: Predicted scores
        y_gt: Ground truth scores

    Returns:
        pcc: Pearson correlation coefficient [-1, 1]
        pvalue: Two-tailed p-value
    """
    pcc, pvalue = stats.pearsonr(y_pred, y_gt)
    return pcc, pvalue


def compute_rank_correlation(y_pred, y_gt, method='spearman'):
    """
    Compute rank correlation.

    Args:
        y_pred: Predicted scores
        y_gt: Ground truth scores
        method: 'spearman' or 'kendall'

    Returns:
        correlation: Correlation coefficient
        pvalue: Two-tailed p-value
    """
    if method == 'spearman':
        return compute_srcc(y_pred, y_gt)
    elif method == 'kendall':
        tau, pvalue = stats.kendalltau(y_pred, y_gt)
        return tau, pvalue
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_top_k_accuracy(y_pred, y_gt, k=5, threshold=1.0):
    """
    Compute Top-K accuracy for ranking tasks.

    Args:
        y_pred: Predicted scores
        y_gt: Ground truth scores
        k: Top-K videos to consider
        threshold: Threshold for correct ranking

    Returns:
        accuracy: Top-K accuracy
    """
    n = len(y_pred)

    # Sort by predicted scores
    sorted_indices = np.argsort(y_pred)[::-1]  # Descending

    # Check if top-K has high-quality videos
    top_k_indices = sorted_indices[:k]
    top_k_gt = y_gt[top_k_indices]

    # Videos with high ground truth scores (e.g., >= 8.0)
    high_quality_mask = y_gt >= threshold
    correct = np.sum(high_quality_mask[top_k_indices])

    accuracy = correct / k
    return accuracy


class MetricsTracker:
    """
    Track metrics during training and evaluation.
    """
    def __init__(self, metrics_to_track=['mae', 'srcc', 'rmse']):
        """
        Args:
            metrics_to_track: List of metric names to track
        """
        self.metrics_to_track = metrics_to_track
        self.reset()

    def reset(self):
        """Reset all tracked metrics."""
        self.history = {metric: [] for metric in self.metrics_to_track}
        self.best_metrics = {metric: None for metric in self.metrics_to_track}
        self.best_metrics_for = {metric: {'val': None, 'epoch': None}
                               for metric in self.metrics_to_track}

    def update(self, metrics_dict, epoch=None):
        """
        Update tracked metrics with new values.

        Args:
            metrics_dict: Dict of metric name -> value
            epoch: Current epoch number
        """
        for metric_name in self.metrics_to_track:
            if metric_name in metrics_dict:
                value = metrics_dict[metric_name]
                self.history[metric_name].append(value)

                # Update best metrics
                if self.best_metrics[metric_name] is None:
                    self.best_metrics[metric_name] = value
                    self.best_metrics_for[metric_name] = {'val': value, 'epoch': epoch}
                else:
                    # For SRCC/PCC, higher is better
                    if metric_name in ['srcc', 'pcc']:
                        if value > self.best_metrics[metric_name]:
                            self.best_metrics[metric_name] = value
                            self.best_metrics_for[metric_name] = {'val': value, 'epoch': epoch}
                    # For errors, lower is better
                    else:
                        if value < self.best_metrics[metric_name]:
                            self.best_metrics[metric_name] = value
                            self.best_metrics_for[metric_name] = {'val': value, 'epoch': epoch}

    def get_history(self, metric_name):
        """Get history of a specific metric."""
        return self.history.get(metric_name, [])

    def get_best(self, metric_name):
        """Get best value and epoch for a specific metric."""
        return self.best_metrics_for.get(metric_name, {'val': None, 'epoch': None})

    def get_latest(self, metric_name):
        """Get latest value of a specific metric."""
        history = self.get_history(metric_name)
        return history[-1] if len(history) > 0 else None

    def print_summary(self):
        """Print summary of tracked metrics."""
        print("\n" + "="*60)
        print("METRICS SUMMARY")
        print("="*60)

        for metric_name in self.metrics_to_track:
            best = self.get_best(metric_name)
            latest = self.get_latest(metric_name)

            if best['val'] is not None:
                print(f"  {metric_name.upper()}:")
                print(f"    Best: {best['val']:.4f} (Epoch {best['epoch']})")
                if latest is not None:
                    print(f"    Latest: {latest:.4f}")
                print()

        print("="*60)


def format_metrics(metrics_dict, format_type='str'):
    """
    Format metrics for logging.

    Args:
        metrics_dict: Dict of metrics
        format_type: 'str', 'dict', or 'tensorboard'

    Returns:
        formatted: Formatted metrics
    """
    if format_type == 'str':
        lines = []
        for key, value in sorted(metrics_dict.items()):
            lines.append(f"{key}: {value:.4f}")
        return "\n".join(lines)
    elif format_type == 'dict':
        return metrics_dict
    elif format_type == 'tensorboard':
        # Format for TensorBoard logging
        tb_dict = {}
        for key, value in metrics_dict.items():
            tb_dict[f"metrics/{key}"] = value
        return tb_dict
    else:
        raise ValueError(f"Unknown format_type: {format_type}")


if __name__ == '__main__':
    # Test metrics
    np.random.seed(42)

    # Generate dummy predictions and ground truth
    n = 100
    y_gt = np.random.uniform(5.0, 10.0, n)  # True scores [5, 10]
    y_pred = y_gt + np.random.normal(0, 0.5, n)  # Add noise

    # Compute metrics
    metrics = compute_metrics(y_pred, y_gt)

    # Test metrics tracker
    tracker = MetricsTracker(metrics_to_track=['mae', 'srcc', 'rmse'])

    for epoch in range(5):
        # Simulate improving metrics
        epoch_metrics = {
            'mae': metrics['mae'] * (1 - epoch * 0.1),
            'srcc': metrics['srcc'] * (1 + epoch * 0.02),
            'rmse': metrics['rmse'] * (1 - epoch * 0.1)
        }
        tracker.update(epoch_metrics, epoch=epoch)

    tracker.print_summary()
