import torch
import torchvision.transforms as trn
import numpy as np
import torchvision.datasets as dset

from folder import ImageFolderCustom

class prepared_dataset(torch.utils.data.Dataset):
	def __init__(self, dataset, in_class):
		self.dataset = dataset
		self.in_class = in_class
		self.num_pts = len(self.dataset)

	def __getitem__(self, index):
		x_origin, target = self.dataset[index]

		return torch.tensor(x_origin)

	def __len__(self):
		return self.num_pts

def data_load(in_class = None, train = True, in_or_out = "in"):
	classes = ['acorn', 'airliner', 'ambulance', 'american_alligator', 'banjo', 'barn', 'bikini', 'digital_clock',
			'dragonfly', 'dumbbell', 'forklift', 'goblet', 'grand_piano', 'hotdog', 'hourglass', 'manhole_cover',
			'mosque', 'nail', 'parking_meter', 'pillow', 'revolver', 'rotary_dial_telephone', 'schooner', 'snowmobile',
			'soccer_ball', 'stingray', 'strawberry', 'tank', 'toaster', 'volcano']
	normalize_transform = trn.Compose([trn.Resize(256), trn.CenterCrop(224),
                               trn.ToTensor(), trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
	if train:
		path = "/home/giatai/Documents/Python/data/ImageNet_30classes/one_class_train/" + classes[in_class]
		data_load = dset.ImageFolder(path, transform = normalize_transform)
	elif not train:
		path = "/home/giatai/Documents/Python/data/ImageNet_30classes/one_class_test/"
		if in_or_out == "out":
			data_load = ImageFolderCustom(path, transform = normalize_transform, remove_classes = classes[in_class])
		elif in_or_out == "in":
			path = path + classes[in_class]
			data_load = ImageFolderCustom(path, transform = normalize_transform)

	return prepared_dataset(data_load, in_class)
