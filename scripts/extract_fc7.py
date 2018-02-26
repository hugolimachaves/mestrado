import numpy as np
import sys
import os
import argparse
import glob
import time
import pandas as pd
from extractor import Extractor

pycaffe_dir = '/opt/caffe/'
sys.path.insert(1,pycaffe_dir)
import caffe

encode = 'utf-8'
num_params = 8 #num params in input param file


def _get_Args():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("input", help = "Input folder with frames")
	parser.add_argument("-g","--gpu", action = 'store_true', help = "Switch for gpu computation.")
	parser.add_argument("-f", "--file", help= "Input params file", default="params_dnn")
	return parser.parse_args()


def initialize_dnn(gpu):

	if gpu:
		#GPU mode (0 - number of GPU)
		caffe.set_device(0) 
		caffe.set_mode_gpu()
	else:
		#CPU mode
		caffe.set_mode_cpu()


def get_params_dnn(param_file):
	
	params = pd.read_csv(param_file, sep=" ")
	return params


def preprocess_params(params):
	
	if(params['model_file'][0] == 'None' or os.path.isfile(params['model_file'][0] == False)):
		raise Exception("The model file must be informed")

	if(params['pretrained_model'][0] == 'None' or os.path.isfile(params['pretrained_model'][0] == False)):
		raise Exception("The pretrained model must be informed")

	if(params.count(axis=1)[0] != num_params): #Count collums in dataframe and compares with the num_params
		raise Exception("There is some missing parameter")


def extract_features(params, input_):

	if(os.path.exists(input_) == False):
		raise Exception("Input not found")

	#Params
	model_def = params['model_file'][0]
	pretrained_model = params['pretrained_model'][0]
	image_dims = None if params['img_dims'][0] == 'None' else params['img_dims'][0]
	mean = None if params['mean'][0] == 'None' else params['mean'][0]
	input_scale = None if params['input_scale'][0] == 'None' else params['input_scale'][0]
	raw_scale = None if params['raw_scale'][0] == 'None' else params['raw_scale'][0]
	channel_swap = None if params['channel_swap'][0] == 'None' else params['channel_swap'][0]
	layer = None if params['layer'][0] == 'None' else params['layer'][0]
	
	#Instatiate Extractor class
	extractor = Extractor(model_def, pretrained_model, image_dims=image_dims, mean=mean, input_scale=input_scale, raw_scale=raw_scale, channel_swap=channel_swap, layer=layer)
	output = extractor.extract(input_)
	return output,layer


def write_features(name, features, layer):

	out_file = name + "." + layer
	df = pd.DataFrame(features)
	df.to_csv(out_file, sep=',', line_terminator='\n', encoding=encode, header=None, index=False)

def _main(args):
	
	initialize_dnn(args.gpu)
	params = get_params_dnn(args.file)
	preprocess_params(params)
	features,layer = extract_features(params, args.input)
	name = args.input.split(os.path.sep)[-2]
	write_features(name, features, layer)

if __name__ == '__main__':
	# parse arguments
	args = _get_Args()
	_main(args)

