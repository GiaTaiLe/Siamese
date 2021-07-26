import numpy as np
import os
import pickle
import random
import argparse
import time
import shutil

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.transforms.functional as trnF
import torchvision.datasets as dset
from torchvision.utils import save_image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2 as cv
from model.resnet import resnet18

parser = argparse.ArgumentParser(description = "Train one-class model - ImageNet",
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--in_class', '-in', type=int, default=2, help='Class to have as the target/in distribution.')
#Optimization options
parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size.')
parser.add_argument('--epochs', '-e', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')

#Checkpoints
parser.add_argument('--save', '-s', type=str, default='snapshots/ood', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')

args = parser.parse_args()

state = {k:v for k,v in args._get_kwargs()}
print(state)

def train(net, train_loader, optimizer, criterion, epoch, args):
	batch_time = AverageMeter("Time", ":6.3f")
	data_time = AverageMeter("Data", ":6.3f")
	losses = AverageMeter("Loss", ":.4f")
	progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses],
		prefix="Epoch: [{}]".format(epoch))

	net.train()
	net.cuda()

	end = time.time()

	for i, images in enumerate(train_loader):
		data_time.update(time.time() - end)

		images[0] = images[0].cuda()
		images[1] = images[1].cuda()

		#compute output and loss
		p1, p2, z1, z2 = net(x1=images[0], x2=images[1])
		loss = -(criterion(p1,z2).mean() + criterion(p2,z1).mean())*0.5

		losses.update(loss.item(), images[0].size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		batch_time.update(time.time() - end)
		end = time.time()

		if i%100 == 0:
			progress.display(i)

def main():
	

class AverageMeter(object):
	def __init__(self, name, fmt=":f"):
		self.name = name
		self.fmt = fmt
		self.reset()

	def reset(self):
		self.val = 0.
		self.avg = 0.
		self.sum = 0.
		self.count = 0.

	def update(self, val, n=1):
		self.val = val
		self.sum+=val*n
		self.count+=n
		self.avg = self.sum/self.count

	def __str__(self):
		fmtstr = "{name}{val" + self.fmt +"}({avg" + self.fmt + "})"
		return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
	def __init__(self, num_batches, meters, prefix=""):
		self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
		self.meters = meters
		self.prefix = prefix

	def display(self, batch):
		entries = [self.prefix + self.batch_fmtstr.format(batch)]
		entries += [str(meter) for meter in self.meters]
		print("\t".join(entries))

	def _get_batch_fmtstr(self, num_batches):
		num_digits = len(str(num_batches//1))
		fmt = '{:' + str(num_digits) + 'd}'
		return '[' + fmt + '/' + fmt.format(num_batches) + ']'