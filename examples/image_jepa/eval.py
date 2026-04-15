"""
Evaluation utilities for self-supervised learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


class LinearProbe(nn.Module):
    """Linear probe classifier for evaluating representations."""

    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


def evaluate_linear_probe(model, linear_probe, val_loader, device, use_amp=True):
    """Evaluate linear probe on validation set.

    Returns:
        top1: Top-1 accuracy (%).
        top5: Top-5 accuracy (%).
        avg_loss: Mean cross-entropy over batches.
    """
    model.eval()
    linear_probe.eval()

    total_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                features, _ = model(data)

            outputs = linear_probe(features.float())
            loss = F.cross_entropy(outputs, target)

            total_loss += loss.item()
            total += target.size(0)

            _, predicted = outputs.max(1)
            correct_top1 += predicted.eq(target).sum().item()

            _, top5_pred = outputs.topk(5, dim=1)
            correct_top5 += top5_pred.eq(target.unsqueeze(1)).any(1).sum().item()

    top1 = 100.0 * correct_top1 / total
    top5 = 100.0 * correct_top5 / total
    avg_loss = total_loss / len(val_loader)

    return top1, top5, avg_loss
