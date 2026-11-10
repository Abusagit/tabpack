"""Prediction and ROC-AUC helpers for Homecredit training."""
import torch
from tabpackage.metrics import binary_roc_auc_pack


def prediction(logits):
    return logits.squeeze(-1).float().sigmoid()


def roc_auc(*, y_true, y_pred):
    scores = y_pred.float()
    single = scores.ndim == 1
    if single:
        scores = scores.unsqueeze(0)
    result = binary_roc_auc_pack(y_true=y_true.to(torch.int64), y_score=scores)
    return result.squeeze(0) if single else result
