import numpy as np
import sys
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data
from models import *
import os
import torch.nn as nn

config = {
	'img_mean': np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32),
	'data_dir': 'VOCdevkit/VOC2012',
	'data_list': 'VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt',
	'num_classes': 21
}

# GPU_IN_USE = torch.cuda.is_available()
GPU_IN_USE = False

def get_iou(data_list, class_num, save_path=None):
	from multiprocessing import Pool 
	from deeplab.metric import ConfusionMatrix

	ConfM = ConfusionMatrix(class_num)
	f = ConfM.generateM
	pool = Pool() 
	m_list = pool.map(f, data_list)
	pool.close() 
	pool.join() 
	
	for m in m_list:
		ConfM.addM(m)

	aveJ, j_list, M = ConfM.jaccard()
	print('meanIOU: ' + str(aveJ) + '\n')
	if save_path:
		with open(save_path, 'w+') as f:
			f.write('meanIOU: ' + str(aveJ) + '\n')
			f.write(str(j_list)+'\n')
			f.write(str(M)+'\n')

def main():
	model_file = sys.argv[1]
	output_file = sys.argv[2]

	model = ResNet(Bottleneck,[3, 4, 23, 3], config['num_classes'])
	
	saved_state_dict = torch.load(model_file)
	model.load_state_dict(saved_state_dict)

	cudnn.enabled = False

	model.eval()
	if GPU_IN_USE:
		model.cuda()

	testloader = data.DataLoader(VOCDataSet(config['data_dir'], config['data_list'], crop_size=(505, 505), mean=config['img_mean'], scale=False, mirror=False), 
									batch_size=1, shuffle=False, pin_memory=True)

	interp = nn.Upsample(size=(505, 505), mode='bilinear')
	data_list = []

	for index, batch in enumerate(testloader):
		if index % 100 == 0:
			print('%d processd'%(index))
		image, label, size, name = batch
		size = size[0].numpy()
		_input = Variable(image, volatile=True)
		if GPU_IN_USE:
			_input = _input.cuda()
		output = model(_input)
		output = interp(output).cpu().data[0].numpy()

		output = output[:,:size[0],:size[1]]
		gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)
		
		output = output.transpose(1,2,0)
		output = np.asarray(np.argmax(output, axis=2), dtype=np.int)

		# show_all(gt, output)
		data_list.append([gt.flatten(), output.flatten()])

	get_iou(data_list, config['num_classes'], output_file)


if __name__ == '__main__':
	main()
