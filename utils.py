import torch
from torch.autograd import Variable
import cv2
import torch.nn as nn
import numpy as np
import os

def save_results(input_img, gt_data, density_map, output_dir, fname='results.png'):
	input_img = input_img[0][0]
	gt_data = 255 * gt_data / np.max(gt_data)
	density_map = 255 * density_map / np.max(density_map)
	gt_data = gt_data[0][0]
	density_map = density_map[0][0]
	if density_map.shape[1] != input_img.shape[1]:
		density_map = cv2.resize(density_map, (input_img.shape[1], input_img.shape[0]))
		gt_data = cv2.resize(gt_data, (input_img.shape[1], input_img.shape[0]))
	result_img = np.hstack((input_img, gt_data, density_map))
	cv2.imwrite(os.path.join(output_dir, fname), result_img)


def save_density_map(density_map, output_dir, fname='results.png'):
	density_map = 255 * density_map / np.max(density_map)
	density_map = density_map[0][0]
	cv2.imwrite(os.path.join(output_dir, fname), density_map)


def display_results(input_img, gt_data, density_map):
	input_img = input_img[0][0]
	gt_data = 255 * gt_data / np.max(gt_data)
	density_map = 255 * density_map / np.max(density_map)
	gt_data = gt_data[0][0]
	density_map = density_map[0][0]
	if density_map.shape[1] != input_img.shape[1]:
		input_img = cv2.resize(input_img, (density_map.shape[1], density_map.shape[0]))
	result_img = np.hstack((input_img, gt_data, density_map))
	result_img = result_img.astype(np.uint8, copy=False)
	cv2.imshow('Result', result_img)
	cv2.waitKey(0)

def save_net(fname, net):
	import h5py
	h5f = h5py.File(fname, mode='w')
	for k, v in net.state_dict().items():
		h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
	import h5py
	h5f = h5py.File(fname, mode='r')
	for k, v in net.state_dict().items():
		param = torch.from_numpy(np.asarray(h5f[k]))
		v.copy_(param)

def np_to_variable(x, is_cuda=True, is_training=False, dtype=torch.FloatTensor):
	if is_training:
		v = Variable(torch.from_numpy(x).type(dtype))
	else:
		v = Variable(torch.from_numpy(x).type(dtype), requires_grad = False)
	if is_cuda:
		v = v.cuda()
	return v


def set_trainable(model, requires_grad):
	for param in model.parameters():
		param.requires_grad = requires_grad


def weights_normal_init(model, dev=0.01):
	if isinstance(model, list):
		for m in model:
			weights_normal_init(m, dev)
	else:
		for m in model.modules():
			if isinstance(m, nn.Conv2d):
				m.weight.data.normal_(0.0, dev)
				if m.bias is not None:
					m.bias.data.fill_(0.0)
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0.0, dev)