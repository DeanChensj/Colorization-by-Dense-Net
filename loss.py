"""@package Multinomial Loss Function
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class MultinomialLoss(torch.autograd.Function):
    def __init__(self, use_cuda=False):
        self.use_cuda = use_cuda

    def forward(self, colorized_img, orig_img):
        return error

    def backward(self, grad_output):
        return grad_input
