import torch
import torch.nn as nn


class SegmentationLoss(nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predictions, targets):
        predictions = predictions.view(predictions.size(0), predictions.size(1), -1)
        targets = targets.view(targets.size(0), -1)
        loss = self.loss(predictions, targets)
        return loss
