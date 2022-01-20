import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class DiceCoefficient(nn.Module):
    def __init__(self):
        super(DiceCoefficient, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return dice

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)       
        
        #flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

#JACCARD
class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # inputs = torch.sigmoid(inputs)       
        
        #flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection - True Positive 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

class FocalLoss(nn.Module):

    def __init__(self, device, alpha=0.8, gamma=1.0, weights=None, activation = False):
        super(FocalLoss, self).__init__()
        self.device = device
        self.weight = weights
        self.gamma = torch.tensor(gamma, dtype=torch.float32).to(device)
        self.alpha = torch.tensor(alpha, dtype=torch.float32).to(device)
        self.eps = 1e-6
        self.activation = activation

    def forward(self, input, target):

        input = input.view(-1)
        target = target.view(-1)

        # target = F.one_hot(target, num_classes=self.num_classes).float()
        if self.activation:
            BCE_loss = F.binary_cross_entropy_with_logits(
                input, target, weight=self.weight, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(
                input, target, reduction='mean')
            
        bce_exp = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-bce_exp)**self.gamma * BCE_loss

        return F_loss

class WeightedSoftDiceLoss(nn.Module):
    def __init__(self, device, v1=0.1, v2=0.9):
        super(WeightedSoftDiceLoss, self).__init__()

        self.v1 = torch.tensor(v1, dtype=torch.float32).to(device)
        self.v2 = torch.tensor(v2, dtype=torch.float32).to(device)
        # self.pow = torch.tensor(2.0, dtype=torch.float32).to(device)

    def forward(self, predictions, targets, smooth=1e-6):
        
        #flatten label and prediction tensors
        p = predictions.view(-1)
        t = targets.view(-1)

        w = t*(self.v2-self.v1)+self.v1
        gp = w * 2. * p - 1
        gt = w * 2. * t - 1

        intersection = (gp * gt).sum()
        nominator = (gp**2.0).sum() + (gt**2.0).sum() + smooth
        wsdc = (2. * intersection + smooth) / nominator
        
        return 1 - wsdc

class L2ReconstructionLoss(nn.Module):
    def __init__(self):
        super(L2ReconstructionLoss, self).__init__()
    def forward(self, reconstruction, target):
        r = reconstruction.view(-1)
        t = target.view(-1)

        return F.mse_loss(r,t)

class L1ReconstructionLoss(nn.Module):
    def __init__(self):
        super(L1ReconstructionLoss, self).__init__()
    def forward(self, reconstruction, target):
        r = reconstruction.view(-1)
        t = target.view(-1)

        return F.l1_loss(r,t)


