from typing import Any, Dict, List, Union
import torch
from torch import nn
import torch.nn.functional as F
import torchmetrics
from torchmetrics import Metric, MetricCollection
from torch.distributions.categorical import Categorical
from omegaconf import DictConfig, ListConfig

class EntropyCategorical(Metric):

    def __init__(self, compute_on_step=True, dist_sync_on_step=False):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)

        self.add_state('entropy', default=torch.tensor(0))

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert len(preds.shape) == 2
        # create categorical distribution over preds
        categorical = Categorical(logits=preds)
        # calculate mean entropy over batch
        self.entropy = categorical.entropy().mean()

    def compute(self):
        return self.entropy.float()


class MaxClassProbCategorical(Metric):

    def __init__(self, compute_on_step=True, dist_sync_on_step=False):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)

        self.add_state('max_class_prob', default=torch.tensor(0))

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert len(preds.shape) == 2
        # compute softmax output of predictions
        probs = F.softmax(preds, dim=1)
        # calculate mean of maximum class probability over batch
        max_probs, max_probs_ind = probs.max(dim=1)
        self.max_class_prob = max_probs.mean(dim=0)

    def compute(self):
        return self.max_class_prob.float()

class Loss(Metric):
    """Metric wrapper for loss functions.
    Used to compute loss on validation/test set.
    """
    higher_is_better = False # We minimize loss, lower is better
    full_state_update = False

    def __init__(self, loss_fn: nn.Module, compute_on_step=True, dist_sync_on_step=False):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)
        self.loss_fn = loss_fn
        self.add_state('loss', default=torch.tensor(0, dtype=torch.float), dist_reduce_fx='mean')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        loss = self.loss_fn(preds, target)
        self.loss = loss
    
    def compute(self):
        return self.loss.float()


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred.argmax(dim=-1) == target).to(dtype=torch.float32).mean()


class TAccuracy(nn.Module):
    higher_is_better = True

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return accuracy(pred, target)


class TError(nn.Module):
    higher_is_better = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1. - accuracy(pred, target)


_metric_registry = {
    'EntropyCategorical': EntropyCategorical,
    'MaxClassProbCategorical': MaxClassProbCategorical,
    'Accuracy': torchmetrics.Accuracy,
    'BinaryAccuracy': torchmetrics.classification.BinaryAccuracy,
    'MulticlassAccuracy': torchmetrics.classification.MulticlassAccuracy,
    'MeanSquaredError': torchmetrics.MeanSquaredError, 
    'TAccuracy': TAccuracy,
    'TError': TError,
}


def get_metric(metric_name: str, metric_args: Dict[str, Any] = {}) -> Metric:
    """Returns the metric (object) given its name. 
    For metrics see torchmetrics.

    Args:
        metric_name (str): The name. Case sensitive. Names according to torchmetrics.
        metric_args (Dict[str, Any]): Arguments to the constructor.

    Returns:
        Metric: the Metric module.
    """
    if metric_name in _metric_registry:
        return _metric_registry[metric_name](**metric_args)
    else:
        assert False, f"Unknown metric name \"{metric_name}\". Available metrics are: {str(_metric_registry.keys())}"


def get_metric_collection(metrics_cfg: List[Union[str, Dict[str, Any]]]) -> MetricCollection:
    metrics = {}
    for i, metric in enumerate(metrics_cfg):
        if isinstance(metric, (dict, DictConfig)):
            assert len(metric) == 1, f"Metric config must be a dict with one key, but is {metric}."
            metric_name = list(metric.keys())[0]
            metric_args = metric[metric_name]
            metrics[f'{metric_name}_{i}'] = get_metric(metric_name, metric_args)
        else:
            metric_name = metric
            metrics[f'{metric_name}_{i}'] = get_metric(metric_name)

    return MetricCollection(metrics)


def create_metrics(metrics_cfg: List[Union[str, Dict[str, Any]]]) -> MetricCollection:
    """Creates a MetricCollection object from a list of metric configs.
    For metrics see torchmetrics.

    Args:
        metrics_cfg (List[Union[str, Dict[str, Any]]]): A list of metric configs. 
            Each metric config can be a string (name of metric) or a dict with one key (name of metric) and the value is a dict of arguments to the metric constructor.

    Returns:
        MetricCollection: the MetricCollection module.
    """
    metrics = get_metric_collection(metrics_cfg)
    train_metrics = metrics.clone()
    val_metrics = metrics.clone()
    return train_metrics, val_metrics