import numpy as np
import os
import pickle
import random
import argparse
import time
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

def train(net, train_loader, optimizer):

	net.train()
	net.cuda()
	pass

def main():
	