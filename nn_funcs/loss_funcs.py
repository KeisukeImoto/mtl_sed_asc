import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class CEplusBCEWithLogitsLoss(nn.Module):
    def __init__(self,alpha=1.0,beta=1.0,reduction='none'):
        super(CEplusBCEWithLogitsLoss,self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(self,input,target):

        input1, input2 = input
        target1 = target[:,:,1:]
        target2 = target[:,0,0]
        target2 = target2.long()

        loss1 = F.binary_cross_entropy_with_logits(input1,target1,reduction=self.reduction)
        loss2 = F.cross_entropy(input2,target2,reduction=self.reduction)

        loss = self.alpha*loss1 + self.beta*loss2

        return loss