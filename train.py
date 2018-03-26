import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path
from models import *
import random

config = {
	'img_mean': np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32),
	'batch_size': 10,
	'data_dir': 'VOCdevkit/VOC2012',
	'data_list': 'VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt',
	'ignore_label': 255,
	'input_size': (321,321),
	'learning_rate': 2.5e-4,
	'momentum': 0.9,
	'num_classes': 21,
	'num_steps': 20000,
	'power': 0.9,
	'random_seed': 23,
	'save_interval': 1000,
	'model_dir': 'models/',
	'weight_decay': 0.0005
}

# GPU_IN_USE = torch.cuda.is_available()
GPU_IN_USE = False

def loss_calc(pred, label):
	"""
	This function returns cross entropy loss for semantic segmentation
	"""
	# out shape batch_size x channels x h x w -> batch_size x channels x h x w
	# label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
	label = Variable(label.long())
	criterion = torch.nn.CrossEntropyLoss(ignore_index=config['ignore_label'])
	if GPU_IN_USE:
		label = label.cuda()
		criterion = criterion.cuda()
	
	return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
	return base_lr*((1-float(iter)/max_iter)**(power))


def get_1x_lr_params_NOscale(model):
	"""
	This generator returns all the parameters of the net except for 
	the last classification layer. Note that for each batchnorm layer, 
	requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
	any batchnorm parameter
	"""
	b = []

	b.append(model.conv1)
	b.append(model.bn1)
	b.append(model.layer1)
	b.append(model.layer2)
	b.append(model.layer3)
	b.append(model.layer4)

	
	for i in range(len(b)):
		for j in b[i].modules():
			jj = 0
			for k in j.parameters():
				jj+=1
				if k.requires_grad:
					yield k

def get_10x_lr_params(model):
	"""
	This generator returns all the parameters for the last layer of the net,
	which does the classification of pixel into classes
	"""
	b = []
	b.append(model.layer5.parameters())

	for j in range(len(b)):
		for i in b[j]:
			yield i
			
			
def adjust_learning_rate(optimizer, i_iter):
	"""Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
	lr = lr_poly(config['learning_rate'], i_iter, config['num_steps'], config['power'])
	optimizer.param_groups[0]['lr'] = lr
	optimizer.param_groups[1]['lr'] = lr * 10


def main():
	"""Create the model and start the training."""
	
	input_size = config['input_size']

	# if GPU_IN_USE:
	# 	cudnn.enabled = True
	cudnn.enabled = False

	model = ResNet(Bottleneck,[3, 4, 23, 3], config['num_classes'])
	model.train()
	if GPU_IN_USE:
		model.cuda()
		cudnn.benchmark = True

	if not os.path.exists(config['model_dir']):
		os.makedirs(config['model_dir'])

	trainloader = data.DataLoader(VOCDataSet(config['data_dir'], config['data_list'],
											max_iters=config['num_steps']*config['batch_size'],
											crop_size=input_size,
											mean=config['img_mean']), 
					batch_size=config['batch_size'], shuffle=True, num_workers=5, pin_memory=True)

	optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': config['learning_rate'] }, 
				{'params': get_10x_lr_params(model), 'lr': 10*config['learning_rate']}], 
				lr=config['learning_rate'], momentum=config['momentum'],weight_decay=config['weight_decay'])
	optimizer.zero_grad()

	interp = nn.Upsample(size=input_size, mode='bilinear')

	for i_iter, batch in enumerate(trainloader):
		images, labels, _, _ = batch
		images = Variable(images)
		if GPU_IN_USE:
			images = images.cuda()
			labels = labels.cuda()

		optimizer.zero_grad()
		adjust_learning_rate(optimizer, i_iter)
		pred = interp(model(images))
		loss = loss_calc(pred, labels)
		loss.backward()
		optimizer.step()
		
		print('iter = {} of {} completed, loss = {}'.format(i_iter, config['num_steps'], loss.data.cpu().numpy()))

		if i_iter >= config['num_steps']-1:
			print('saving model ...')
			torch.save(model.state_dict(),os.path.join(config['model_dir'], 'VOC12_scenes_'+str(config['num_steps'])+'.pth'))
			break

		if i_iter % config['save_interval'] == 0 and i_iter!=0:
			print('saving model ...')
			torch.save(model.state_dict(),os.path.join(config['model_dir'], 'VOC12_scenes_'+str(i_iter)+'.pth'))     


if __name__ == '__main__':
	main()
