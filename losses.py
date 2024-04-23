import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return bce + dice * 0.5

class GLDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """compute the weighted dice_loss
        Args:
            pred (tensor): prediction after softmax, shape(bath_size, channels, height, width)
            target (tensor): gt, shape(bath_size, channels, height, width)
        Returns:
            gldice_loss: loss value
        """    
        wei = torch.sum(target, axis=[0,2,3]) # (n_class,)
        wei = 1/(wei**2+sys.float_info.epsilon)
        intersection = torch.sum(wei*torch.sum(pred * target, axis=[0,2,3]))
        union = torch.sum(wei*torch.sum(pred + target, axis=[0,2,3]))
        gldice_loss = 1 - (2. * intersection) / (union + sys.float_info.epsilon)
        return gldice_loss
