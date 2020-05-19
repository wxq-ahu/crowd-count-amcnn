import os
import torch
import numpy as np
import sys
from counter import CrowdCounter
from data_loader import ImageDataLoader
from utils import *
from torch.autograd import Variable
try:
	from termcolor import cprint
except ImportError:
	cprint = None

try:
	from pycrayon import CrayonClient
except ImportError:
	CrayonClient = None


def log_print(text, color=None, on_color=None, attrs=None):
	if cprint is not None:
		cprint(text, color=color, on_color=on_color, attrs=attrs)
	else:
		print(text)

def evaluate_model(trained_model, data_loader):
	net_ = CrowdCounter()
	load_net(trained_model, net_)
	net_.cuda()
	net_.eval()
	mae = 0.0
	mse = 0.0
	for blob in data_loader:
		im_data = blob['data']
		gt_data = blob['gt_density']
		#im_data = Variable(torch.from_numpy(im_data).type(torch.FloatTensor)).cuda()
		#gt_data = Variable(torch.from_numpy(gt_data).type(torch.FloatTensor)).cuda()
		density_map = net_(im_data)
		density_map = density_map.data.cpu().numpy()
		gt_count = np.sum(gt_data)
		et_count = np.sum(density_map)
		mae += abs(gt_count-et_count)
		mse += ((gt_count-et_count)*(gt_count-et_count))
	mae = mae/data_loader.get_num_samples()
	mse = np.sqrt(mse/data_loader.get_num_samples())
	return mae,mse

def train_model():
	method = 'AMCNN'  # method name - used for saving model file
	dataset_name = 'SHA'  # dataset name - used for saving model file
	output_dir = './saved_models_SHA/'  # model files are saved here

	# train and validation paths
	train_path = './data/formatted_trainval_A/shanghaitech_part_A_patches_9/train'
	train_gt_path = './data/formatted_trainval_A/shanghaitech_part_A_patches_9/train_den'
	val_path = './data/formatted_trainval_A/shanghaitech_part_A_patches_9/val'
	val_gt_path = './data/formatted_trainval_A/shanghaitech_part_A_patches_9/val_den'

	# training configuration
	start_step = 0
	end_step = 2000
	lr = 0.00001
	momentum = 0.9
	disp_interval = 500
	log_interval = 250

	rand_seed = 64678
	if rand_seed is not None:
		np.random.seed(rand_seed)
		torch.manual_seed(rand_seed)
		torch.cuda.manual_seed(rand_seed)

	# loadt training and validation data
	data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=False, pre_load=True)
	class_wts = data_loader.get_classifier_weights()
	data_loader_val = ImageDataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=False, pre_load=True)

	# load net and initialize it
	net = CrowdCounter(ce_weights=class_wts)
	weights_normal_init(net, dev=0.01)
	net.cuda()
	net.train()

	params = list(net.parameters())
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	# training
	train_loss = 0
	best_mae = sys.maxsize
	best_mse = sys.maxsize
	for epoch in range(start_step, end_step + 1):
		step = -1
		train_loss = 0
		for blob in data_loader:
			step = step + 1
			im_data = blob['data']
			gt_data = blob['gt_density']
			gt_class_label = blob['gt_class_label']

			# data augmentation on the fly
			if np.random.uniform() > 0.5:
				# randomly flip input image and density
				im_data = np.flip(im_data, 3).copy()
				gt_data = np.flip(gt_data, 3).copy()
			if np.random.uniform() > 0.5:
				# add random noise to the input image
				im_data = im_data + np.random.uniform(-10, 10, size=im_data.shape)

			density_map = net(im_data, gt_data, gt_class_label, class_wts)
			loss = net.loss
			train_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if step % disp_interval == 0:
				gt_count = np.sum(gt_data)
				density_map = density_map.data.cpu().numpy()
				et_count = np.sum(density_map)
				save_results(im_data, gt_data, density_map, output_dir)
				log_text = 'epoch: %4d, step %4d, gt_cnt: %4.1f, et_cnt: %4.1f, loss: %4.7f' % (epoch, step, gt_count, et_count, train_loss)
				log_print(log_text, color='green', attrs=['bold'])

		if (epoch % 2 == 0):
			save_name = os.path.join(output_dir, '{}_{}_{}.h5'.format(method, dataset_name, epoch))
			save_net(save_name, net)
			# calculate error on the validation dataset
			mae, mse = evaluate_model(save_name, data_loader_val)
			if mae <= best_mae and mse <= best_mse:
				best_mae = mae
				best_mse = mse
				best_model = '{}_{}_{}.h5'.format(method, dataset_name, epoch)
			log_text = 'EPOCH: %d, MAE: %.1f, MSE: %0.1f' % (epoch, mae, mse)
			log_print(log_text, color='green', attrs=['bold'])
			log_text = 'BEST MAE: %0.1f, BEST MSE: %0.1f, BEST MODEL: %s' % (best_mae, best_mse, best_model)
			log_print(log_text, color='green', attrs=['bold'])

if __name__ == '__main__':
	train_model()
