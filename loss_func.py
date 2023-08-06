import torch
import torch.nn as nn

class MeanLogarithmLoss(nn.Module):
    def __init__(self):
        super(MeanLogarithmLoss, self).__init__()

    def forward(self, output):
        # Calculate the element-wise mean of the logarithm of the absolute errors
        log_errors = -1*torch.log(torch.abs(output))
        
        # Calculate the mean of the log errors along the batch dimension
        loss = torch.mean(log_errors)
        
        return loss
