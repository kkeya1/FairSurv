# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 02:03:17 2020

@author: Kamrun Naher Keya
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

#%% Loss function is the neg log partial likelihood
class negLogLikelihood(nn.Module):
    # Source: deepSurv implementation with PyTorch (https://gitlab.com/zydou/deepsurv/-/tree/master/DeepSurv-Pytorch)
    def __init__(self):
        super(negLogLikelihood, self).__init__()

    def forward(self, prediction, targets):
        risk = prediction
        E = targets
        hazard_ratio = torch.exp(risk)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
        uncensored_likelihood = risk - log_risk
        censored_likelihood = uncensored_likelihood.transpose(0, 1) * E.float()
        num_observed_events = torch.sum(E.float())
        neg_likelihood = -torch.sum(censored_likelihood) / num_observed_events # average the loss 

        return neg_likelihood
#%% linear Cox PH model
class linearCoxPH_Regression(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearCoxPH_Regression, self).__init__()
        self.linear = nn.Linear(inputSize, outputSize,bias=False)

    def forward(self, x):
        out = self.linear(x) # linear layer to output Linear to output Log Hazard Ratio
        return out