import os.path
import math
import numpy as np

import torch.nn as nn
import torch

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import cv2 # from opencv-python
from torch.utils import data
import random

import pdb


class Bottleneck(nn.Module):
	expansion = 4
	def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.stride = stride
		self.downsample = downsample

		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
		self.bn1 = nn.BatchNorm2d(planes, affine=True)
		for i in self.bn1.parameters():
			i.requires_grad = False

		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=dilation, bias=False, dilation=dilation)
		self.bn2 = nn.BatchNorm2d(planes, affine=True)
		for i in self.bn2.parameters():
			i.requires_grad = False

		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4, affine=True)
		for i in self.bn3.parameters():
			i.requires_grad = False

		self.relu = nn.ReLU(inplace=True)

	def forward(self, _input):
		conv1_out = self.relu(self.bn1(self.conv1(_input)))
		conv2_out = self.relu(self.bn2(self.conv2(conv1_out)))
		conv3_out = self.bn3(self.conv3(conv2_out))
		if self.downsample is not None:
			_input = self.downsample(_input)
		return self.relu(conv3_out + _input) # residual connection


class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out


class ResNet(nn.Module):
	def __init__(self, block, layers, num_classes):
		super(ResNet, self).__init__()
		self.inplanes = 64

		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64, affine=True)
		for i in self.bn1.parameters():
			i.requires_grad = False
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
		self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, 0.01)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion,affine=True))
		for i in downsample._modules['1'].parameters():
			i.requires_grad = False
		layers = []
		layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, dilation=dilation))

		return nn.Sequential(*layers)
	def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
		return block(dilation_series,padding_series,num_classes)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)

		return x


class CrossEntropy2d(nn.Module):

	def __init__(self, size_average=True, ignore_label=255):
		super(CrossEntropy2d, self).__init__()
		self.size_average = size_average
		self.ignore_label = ignore_label

	def forward(self, predict, target, weight=None):
		"""
			Args:
				predict:(n, c, h, w)
				target:(n, h, w)
				weight (Tensor, optional): a manual rescaling weight given to each class.
										   If given, has to be a Tensor of size "nclasses"
		"""
		n, c, h, w = predict.size()
		target_mask = (target >= 0) * (target != self.ignore_label)
		target = target[target_mask]
		if not target.data.dim():
			return Variable(torch.zeros(1))
		predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
		predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
		loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
		return loss


class VOCDataSet(data.Dataset):
	def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
		self.root = root
		self.list_path = list_path
		self.crop_h, self.crop_w = crop_size
		self.scale = scale
		self.ignore_label = ignore_label
		self.mean = mean
		self.is_mirror = mirror
		# self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
		self.img_ids = [i_id.strip() for i_id in open(list_path)]
		if not max_iters==None:
			self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
		self.files = []
		# for split in ["train", "trainval", "val"]:
		for name in self.img_ids:
			img_file = os.path.join(self.root, "JPEGImages/%s.jpg" % name)
			label_file = os.path.join(self.root, "SegmentationClass/%s.png" % name)
			if os.path.isfile(img_file) and os.path.isfile(label_file):
				self.files.append({
					"img": img_file,
					"label": label_file,
					"name": name
				})
		#pdb.set_trace()

	def __len__(self):
		return len(self.files)

	def generate_scale_label(self, image, label):
		f_scale = 0.5 + random.randint(0, 11) / 10.0
		image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
		label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
		return image, label

	def __getitem__(self, index):
		datafiles = self.files[index]
		image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
		label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
		size = image.shape
		name = datafiles["name"]
		if self.scale:
			image, label = self.generate_scale_label(image, label)
		image = np.asarray(image, np.float32)
		image -= self.mean
		img_h, img_w = label.shape
		pad_h = max(self.crop_h - img_h, 0)
		pad_w = max(self.crop_w - img_w, 0)
		if pad_h > 0 or pad_w > 0:
			img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
				pad_w, cv2.BORDER_CONSTANT, 
				value=(0.0, 0.0, 0.0))
			label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
				pad_w, cv2.BORDER_CONSTANT,
				value=(self.ignore_label,))
		else:
			img_pad, label_pad = image, label

		img_h, img_w = label_pad.shape
		h_off = random.randint(0, img_h - self.crop_h)
		w_off = random.randint(0, img_w - self.crop_w)
		# roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
		image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
		label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
		#image = image[:, :, ::-1]  # change to BGR
		image = image.transpose((2, 0, 1))
		if self.is_mirror:
			flip = np.random.choice(2) * 2 - 1
			image = image[:, :, ::flip]
			label = label[:, ::flip]

		return image.copy(), label.copy(), np.array(size), name