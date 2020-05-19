#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class AMCNN(nn.Module):
	def __init__(self, bn=False, num_classes=10):
		super(AMCNN, self).__init__()
		self.num_classes = num_classes
		self.base_layer_one = nn.Sequential(Conv2d(1, 8, 9, same_padding=True, NL='prelu', bn=bn),
		                                    Conv2d(8, 16, 7, same_padding=True, NL='prelu', bn=bn))

		self.base_layer_two = nn.Sequential(Conv2d(1, 8, 7, same_padding=True, NL='prelu', bn=bn),
		                                    Conv2d(8, 16, 5, same_padding=True, NL='prelu', bn=bn))

		self.base_layer_three = nn.Sequential(Conv2d(1, 8, 5, same_padding=True, NL='prelu', bn=bn),
		                                      Conv2d(8, 16, 3, same_padding=True, NL='prelu', bn=bn))

		self.base_layer_four = nn.Sequential(Conv2d(1, 8, 3, same_padding=True, NL='prelu', bn=bn),
		                                     Conv2d(8, 16, 3, same_padding=True, NL='prelu', bn=bn))

		self.hl_prior_1 = nn.Sequential(Conv2d(64, 16, 9, same_padding=True, NL='prelu', bn=bn),
		                                nn.MaxPool2d(2),
		                                Conv2d(16, 32, 7, same_padding=True, NL='prelu', bn=bn),
		                                nn.MaxPool2d(2),
		                                Conv2d(32, 16, 7, same_padding=True, NL='prelu', bn=bn),
		                                Conv2d(16, 8, 7, same_padding=True, NL='prelu', bn=bn))

		self.feamap0_out = nn.Sequential(nn.ConvTranspose2d(8, 20, 4, stride=2, padding=1, output_padding=0, bias=True),
		                                 nn.PReLU(),
		                                 nn.ConvTranspose2d(20, 8, 4, stride=2, padding=1, output_padding=0, bias=True),
		                                 nn.PReLU())

		self.hl_prior_2 = nn.Sequential(nn.AdaptiveMaxPool2d((32, 32)),
		                                Conv2d(8, 4, 1, same_padding=True, NL='prelu', bn=bn))

		self.hl_prior_fc1 = FC(4 * 1024, 512, NL='prelu')
		self.hl_prior_fc2 = FC(512, 256, NL='prelu')
		self.hl_prior_fc3 = FC(256, self.num_classes, NL='prelu')

		self.earlystage_one = nn.Sequential(
			Conv2d(64, 32, 7, same_padding=True, NL='prelu', bn=bn),
			Conv2d(32, 16, 7, same_padding=True, NL='prelu', bn=bn)
		)

		self.earlystage_two = nn.Sequential(
			Conv2d(64, 32, 5, same_padding=True, NL='prelu', bn=bn),
			Conv2d(32, 16, 5, same_padding=True, NL='prelu', bn=bn)
		)

		self.earlystage_three = nn.Sequential(
			Conv2d(64, 32, 3, same_padding=True, NL='prelu', bn=bn),
			Conv2d(32, 16, 3, same_padding=True, NL='prelu', bn=bn)
		)

		self.earlystage_out = nn.Sequential(
			Conv2d(3, 16, 1, same_padding=True, NL='relu', bn=bn)
		)

		self.largefeamap = nn.Sequential(
			Conv2d(48, 32, 5, same_padding=True, NL='prelu', bn=bn),
			nn.MaxPool2d(2),
			Conv2d(32, 24, 5, same_padding=True, NL='prelu', bn=bn),
			nn.MaxPool2d(2),
			Conv2d(24, 12, 3, same_padding=True, NL='prelu', bn=bn),
			Conv2d(12, 16, 3, same_padding=True, NL='prelu', bn=bn)
		)

		self.midfeamap = nn.Sequential(
			Conv2d(48, 32, 3, same_padding=True, NL='prelu', bn=bn),
			nn.MaxPool2d(2),
			Conv2d(32, 24, 3, same_padding=True, NL='prelu', bn=bn),
			nn.MaxPool2d(2),
			Conv2d(24, 12, 3, same_padding=True, NL='prelu', bn=bn),
			Conv2d(12, 16, 3, same_padding=True, NL='prelu', bn=bn)
		)

		self.feamap_out = nn.Sequential(
			nn.ConvTranspose2d(16, 12, 4, stride=2, padding=1, output_padding=0, bias=True),
			nn.PReLU(),
			nn.ConvTranspose2d(12, 16, 4, stride=2, padding=1, output_padding=0, bias=True),
			nn.PReLU()
		)

		self.feamap = nn.Sequential(
			Conv2d(40, 32, 3, same_padding=True, NL='prelu', bn=bn),
			Conv2d(32, 24, 3, same_padding=True, NL='prelu', bn=bn),
			Conv2d(24, 16, 1, same_padding=True, NL='prelu', bn=bn)
		)

		self.fout = nn.Sequential(
			nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1, output_padding=0, bias=True),
			nn.PReLU(),
			nn.ConvTranspose2d(8, 16, 4, stride=2, padding=1, output_padding=0, bias=True),
			nn.PReLU()
		)

		self.conv1x1 = nn.Sequential(
			Conv2d(16, 1, 1, same_padding=True, NL='relu', bn=bn)
		)

		self.attNet = nn.Sequential(
			attentionNet(16, 1, 5, same_padding=True)
		)

		self.fcat = nn.Sequential(
			Conv2d(19, 1, 1, same_padding=True, NL='relu', bn=bn)
		)

	def get_attmap(self, feamap):
		x_preattmap = self.attNet(feamap)
		x_fea1X1 = self.conv1x1(feamap)
		x_midattmap = torch.mul(x_preattmap, x_fea1X1)
		x_endattmap = torch.add(x_midattmap, x_fea1X1)
		return x_endattmap

	def forward(self, im_data):
		# Hierachical Desity Estimator Level 1
		x_bl_one = self.base_layer_one(im_data)
		x_bl_two = self.base_layer_two(im_data)
		x_bl_three = self.base_layer_three(im_data)
		x_bl_four = self.base_layer_four(im_data)
		x_base = torch.cat((x_bl_one, x_bl_two, x_bl_three, x_bl_four), 1)

		# AUCC
		aucc1 = self.hl_prior_1(x_base)
		aucc2 = self.hl_prior_2(aucc1)
		aucc2 = aucc2.view(aucc2.size()[0], -1)
		aucc = self.hl_prior_fc1(aucc2)
		aucc = F.dropout(aucc, training=self.training)
		aucc = self.hl_prior_fc2(aucc)
		aucc = F.dropout(aucc, training=self.training)
		x_cls = self.hl_prior_fc3(aucc)

		# Hierachical Desity Estimator Level 2
		x_early_one = self.earlystage_one(x_base)
		x_early_attmap_one = self.get_attmap(x_early_one)
		x_early_two = self.earlystage_two(x_base)
		x_early_attmap_two = self.get_attmap(x_early_two)
		x_early_three = self.earlystage_three(x_base)
		x_early_attmap_three = self.get_attmap(x_early_three)
		x_early_cat = torch.cat((x_early_one, x_early_two, x_early_three), 1)
		x_early = torch.cat((x_early_attmap_one, x_early_attmap_two, x_early_attmap_three), 1)
		x_early_attmap = self.earlystage_out(x_early)

		# Hierachical Desity Estimator Level 3
		x_LF = self.largefeamap(x_early_cat)
		x_LF_out = self.feamap_out(x_LF)
		x_LF_attmap = self.get_attmap(x_LF_out)
		x_MF = self.midfeamap(x_early_cat)
		x_MF_out = self.feamap_out(x_MF)
		x_MF_attmap = self.get_attmap(x_MF_out)
		x_LM_cat = torch.cat((x_LF, x_MF), 1)
		x_cat_attmap = torch.cat((x_LF_attmap, x_MF_attmap), 1)

		# Hierachical Desity Estimator Level 4
		x_cat = torch.cat((x_LM_cat, aucc1), 1)
		x_feamap = self.feamap(x_cat)
		x_feamap = self.fout(x_feamap)
		x_feamap_attmap = self.get_attmap(x_feamap)

		x_den = torch.cat((x_early_attmap, x_cat_attmap, x_feamap_attmap), 1)
		x_den = self.fcat(x_den)
		return x_den, x_cls

class attentionNet(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, strides=1, same_padding=False):
		super(attentionNet, self).__init__()
		padding = int((kernel_size - 1) / 2) if same_padding else 0
		self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size, strides, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size, strides, padding)
        )
		#self.sm = nn.Softmax()
	def forward(self,x):
		x_conv = self.conv(x)
		prob = self.spatial_softmax(x_conv)
		return prob

	def spatial_softmax(self, x):
		n_grids = x.size()[2] * x.size()[3]
		x_reshape = torch.reshape(x,(-1,n_grids))
		out = nn.Softmax()
		prob = out(x_reshape)
		prob = torch.reshape(prob, (1, 1, x.size()[2], x.size()[3]))
		return prob

class Conv2d(nn.Module):
	def __init__(self, in_dim, out_dim, kernel_size, stride=1, NL='prelu', same_padding=False, bn=False):
		super(Conv2d, self).__init__()
		padding = int((kernel_size - 1) / 2) if same_padding else 0
		self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding=padding)
		self.bn = nn.BatchNorm2d(out_dim, eps=0.001, momentum=0, affine=True) if bn else None
		if NL == 'relu' :
			self.relu = nn.ReLU(inplace=True)
		elif NL == 'prelu':
			self.relu = nn.PReLU()
		else:
			self.relu = None

	def forward(self, x):
		x = self.conv(x)
		if self.bn is not None:
			x = self.bn(x)
		if self.relu is not None:
			x = self.relu(x)
		return x


class FC(nn.Module):
	def __init__(self, in_dim, out_dim, NL='prelu'):
		super(FC, self).__init__()
		self.fc = nn.Linear(in_dim, out_dim)
		if NL == 'relu' :
			self.relu = nn.ReLU(inplace=True)
		elif NL == 'prelu':
			self.relu = nn.PReLU()
		else:
			self.relu = None

	def forward(self, x):
		x = self.fc(x)
		if self.relu is not None:
			x = self.relu(x)
		return x

