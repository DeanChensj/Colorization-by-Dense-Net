import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from model import Net
from loss import MultinomialLoss


def load_data():
    return grey_img, color_img


parser = argparse.ArgumentParser()
parser.add_argument(('-b', '--batch'), type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument(('-e', '--epoch'), type=int, default=200, metavar='N',
                    help='the number of training epoches')
parser.add_argument(('-c', '--cuda'), type=bool, default=True, metavar='N',
                    help='whether to use cuda or not')
args = parser.parse_args()

batch_size = args.batch
num_epoches = args.epoch
use_cuda = args.cuda

dense_net = DenseNet(use_cuda)
mlp = Net(use_cuda)
mtn_loss = MultinomialLoss(use_cuda)
optimizer = None

if __name__ == "__main__":
    # Use numpy array here
    grey_img, color_img = load_data()
    num_imgs = grey_img.shape[0]
    num_batches = num_imgs // batch_size
    # Convert to pytorch Variable while training
    for i in range(num_epoches):
        # Random shuffle data every epoch
        perm = torch.randperm(num_imgs)
        grey_img, color_img = grey_img[perm], color_img[perm]
        for j in range(num_batches):
            grey_batch, color_batch = grey_img[j * batch_size:(
                j + 1) * batch_size], color_img[j * batch_size:(j + 1) * batch_size]
            grey_batch, color_batch = Variable(
                grey_batch), Variable(color_batch)
            feature_list = dense_net(grey_batch)
            # Reshape feature list if necessary
            # Use FCL to do regression
            optimizer.zero_grad()
            out_img = mlp(feature_list)
            err = mtn_loss(out_img, color_batch)
            err.backward()
            optimizer.step
            # Print debug info
            t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("[%s] Epoch %d, Batch %d, Loss = %lf" %
                  (t, i + 1, j + 1, err))
