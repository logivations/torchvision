import logging

import torch
import torch.nn as nn

from torch.autograd import Function


class ConfidenceLossFunction(Function):
    @staticmethod
    def forward(ctx, pred, target):
        """
                This method in the ConfidenceLossFunction uses the logic of the standard MSE loss, but calculates it separately for loaded and confidence outputs. Also, this function does not take the average error value at this stage, but stores the error value for each image separately.
        Then we scale the error from the loaded output to the ground truth value of confidence. Next, we take the average value of loaded loss and confidence loss and sum them.
                :param ctx:
                :param pred:
                :param target:
                :return:
        """
        # Save tensors
        loaded_pred, confidence_pred, orientation_pred = pred[:, 0].unsqueeze(1), pred[:, 1].unsqueeze(1), pred[:, 2].unsqueeze(1)

        loaded_target, confidence_target, orientation_target = target[:, 0].unsqueeze(1), target[
            :, 1
        ].unsqueeze(1), target[:, 2].unsqueeze(1)

        has_pallets_mask = (loaded_target > 0.0).float()
        ctx.save_for_backward(
            loaded_pred, confidence_pred, orientation_pred, loaded_target, confidence_target, orientation_target, has_pallets_mask
        )

        # Calculate loss
        loaded_loss = (loaded_pred - loaded_target) ** 2
        confidence_loss = (confidence_pred - confidence_target) ** 2
        orientation_loss = (orientation_pred - orientation_target) ** 2

        # Scale the loaded loss by the confidence
        scaled_loaded_loss = loaded_loss * confidence_target

        masked_orientation_loss = orientation_loss * has_pallets_mask

        total_loss = scaled_loaded_loss.mean() + confidence_loss.mean()

        if has_pallets_mask.sum() > 0:
            total_loss = total_loss + masked_orientation_loss.sum() / has_pallets_mask.sum()

        return total_loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        This method in the ConfidenceLossFunction class computes the gradients of the loss with respect to the predicted values during backpropagation. It first retrieves the predictions and target labels saved during the forward pass, then calculates the gradients for two components: the loaded predictions and the confidence predictions. These gradients are normalized by the batch size, concatenated to form a single gradient for predictions, scaled by grad_output, and returned for further parameter updates in the model.
        :param ctx:
        :param grad_output:
        :return:
        """
        (
            loaded_pred,
            confidence_pred,
            orientation_pred,
            loaded_target,
            confidence_target,
            orientation_target,
            has_pallets_mask
        ) = ctx.saved_tensors

        # Compute gradients for the loaded
        grad_loaded_pred = 2 * (loaded_pred - loaded_target) * confidence_target
        grad_loaded_pred = grad_loaded_pred / loaded_pred.size(0)  # Normalize by batch size

        # Compute gradients for the confidence
        grad_confidence_pred = 2 * (confidence_pred - confidence_target)
        grad_confidence_pred = grad_confidence_pred / confidence_pred.size(0)  # Normalize by batch size

        # Compute gradients for the orientation
        grad_orientation_pred = 2 * (orientation_pred - orientation_target)
        grad_orientation_pred = grad_orientation_pred / orientation_pred.size(0)  # Normalize by batch size

        num_with_pallets = has_pallets_mask.sum()
        if num_with_pallets > 0:
            grad_orientation_pred = grad_orientation_pred / num_with_pallets
        else:
            grad_orientation_pred = torch.zeros_like(orientation_pred)

        grad_loaded_pred = grad_loaded_pred.expand_as(loaded_pred)
        grad_confidence_pred = grad_confidence_pred.expand_as(confidence_pred)
        grad_orientation_pred = grad_orientation_pred.expand_as(orientation_pred)

        grad_pred = torch.cat([grad_loaded_pred, grad_confidence_pred, grad_orientation_pred], dim=1)
        grad_pred = grad_pred * grad_output

        return grad_pred, None


class ConfidenceLoss(nn.Module):
    def __init__(self):
        """"""
        super(ConfidenceLoss, self).__init__()

    def forward(self, pred, target):
        """Apply ConfidenceLoss function"""
        return ConfidenceLossFunction.apply(pred, target)
