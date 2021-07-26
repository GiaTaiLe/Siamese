import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as trn
import torchvision.datasets as dset

from resnet import resnet18

class SimSiam(nn.Module):
	def __init__(self, dim=2048, pred_dim=512):
		super(SimSiam, self).__init__()

		self.encoder = models.resnet50(num_classes=dim, zero_init_residual=True)

		prev_dim = self.encoder.fc.weight.shape[1]
		self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias = False),
										nn.BatchNorm1d(prev_dim),
										nn.Relu(inplace=True), #first layer
										nn.Linear(prev_dim, prev_dim, bias = False),
										nn.BatchNorm1d(prev_dim),
										nn.Relu(inplace=True), #second layer
										self.encoder.fc,
										nn.BatchNorm1d(dim, affine=False))
		self.encoder.fc[6].bias.requires_grad=False

		self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
										nn.BatchNorm1d(pred_dim),
										nn.Relu(inplace=True),
										nn.Linear(pred_dim, dim))

	def forward(self, x1, x2):
		"x1: first view of image, x2: second view of image"
		z1 = self.encoder(x1)
		z2 = self.encoder(x2)

		p1 = self.predictor(z1)
		p2 = self.predictor(z2)

		return p1, p2, z1.detach(), z2.detach()