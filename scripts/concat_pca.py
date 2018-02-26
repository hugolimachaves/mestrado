'''
gather all the files, of all classes, of a specified level.
Then  a PCA basis is found for this files
INPUT: CNN Flow, press -h for more info on the console
OUTPUT: PCA autovector e mean at the cnnf folder
ARGUMENTS:
*path to the cnn flows
*name of the folder of the desired elvel
*class 1 folder
*class 2 folder
*class 3 folder
*class 4 folder
*class 5 folder
*class 6 folder
'''

import cv2 as cv
import numpy as np
import pca
import argparse
import os

def _get_Args():
	parser = argparse.ArgumentParser()
	parser.add_argument("path", help="path containing all the descriptors files")
	parser.add_argument("--features", "-f",  help="numero de features apos o PCA", default = 100, type = int)
	return parser.parse_args()

def read_file(name):
	desc = open(name, "r")
	descriptor = desc.readlines()
	desc.close()
	return descriptor

def concat_base():
	file_path = args.path
	caminho = []
	cnnFlow = []
	allCnnFlow = np.array([],'float32')
	not_first_time = False
	print('Current class\'s path :')
	print(file_path)
	for i in os.listdir(file_path): # laço para o numero de arquivos que tem na pasta ( todas as pastas tem o mesmo numero de arquivos)
		complete_path = os.path.join(file_path,i)
		cnnFlow = read_file(complete_path)
		cnnFlow2 = pca.file2PCA(cnnFlow)
		if not_first_time:
			allCnnFlow = np.append(allCnnFlow,cnnFlow2,axis=0)
		else:
			allCnnFlow = cnnFlow2
		not_first_time = True
	return allCnnFlow

def _main(args):
	allCnnFlow = concat_base()
	cnnFlow_conformed = pca.conform2PCA(allCnnFlow)
	n_features_after_PCA = args.features
	print('Wait! can take a little while to get it done')
	eigenVectors, mean = pca.baseground_PCA(cnnFlow_conformed,n_features_after_PCA)
	print('Done!') 
	print(eigenVectors.shape)
	pca.write_pca_baseground('mean_', '' , mean, args.path)
	pca.write_pca_baseground('eigenVectors_' , '' , eigenVectors, args.path)


if __name__ == '__main__':
	args = _get_Args()
	_main(args)
