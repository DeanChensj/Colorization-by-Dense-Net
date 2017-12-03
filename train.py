import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from model import Net
from skimage import color
from util import load_data
import time


parser = argparse.ArgumentParser()
# Optimization
parser.add_argument(('-b', '--batch'), type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument(('-e', '--epoch'), type=int, default=200, metavar='N',
                    help='the number of training epoches')
parser.add_argument(('-c', '--cuda'), type=bool, default=True, metavar='N',
                    help='whether to use cuda or not')
parser.add_argument(('-lr', '--learning-rate'), type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument(('-d', '--learning-rate-decay-factor'), type=float, default=0.5,
                    help='learning rate decay factor (default: 0.5)')
# Checkpoint
parser.add_argument(('-cp', '--checkpoint-every'), type=int, default=200,
                    help='checkpoint frequency')
parser.add_argument(('-rc', '--resume-from-checkpoint'), type=str, default='',
                    help='checkpoint file to resume from')

args = parser.parse_args()

batch_size = args.batch
num_epoches = args.epoch
use_cuda = args.cuda
lr = args.learning_rate
lr_decay = args.learning_rate_decay_factor
checkpoint_every = args.checkpoint_every
checkpoint_file = args.resume_from_checkpoint

if checkpoint_file != '':
    print('Loading checkpoint from %s' % checkpoint_file)
    model = torch.load(checkpoint_file)
else:
    print('Initializing model from scratch')
    model = Net(use_cuda)

criterion = nn.MSELoss()
optimizer = None

if __name__ == "__main__":
    # Use numpy array here
    grey_img, color_img = load_data()
    num_imgs = grey_img.shape[0]
    num_batches = num_imgs // batch_size
    loss_list = []

    # Convert to pytorch Variable while training
    for i in range(num_epoches):
        # Random shuffle data every epoch
        perm = torch.randperm(num_imgs)
        grey_img, color_img = grey_img[perm], color_img[perm]

        loss_sum = 0.0
        for j in range(num_batches):
            grey_batch, color_batch = grey_img[j * batch_size:(
                j + 1) * batch_size], color_img[j * batch_size:(j + 1) * batch_size]
            grey_batch, color_batch = Variable(
                grey_batch), Variable(color_batch)

            output_batch = model(grey_batch)
            optimizer.zero_grad()

            err = criterion(output_batch, color_batch)
            loss_sum += err.data[0]
            err.backward()
            optimizer.step()

            t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("[%s] Epoch %d, Batch %d, Loss = %lf" %
                  (t, i + 1, j + 1, err))

        loss_sum /= num_batches
        loss_list.append(loss_sum)
        if i % checkpoint_every == 0:
            torch.save(model, 'TrainedModel')
    print(loss_list)

