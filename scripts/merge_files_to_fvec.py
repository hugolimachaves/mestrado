import cv2 as cv
import numpy as np
import pca
import argparse
import os
from DescrFVEC import saveDescriptor
import pandas as pd


encode = "utf-8"

train_set = {11, 12, 13, 14, 15, 16, 17, 18}
validation_set = {1, 4, 19, 20, 21, 23, 24, 25}	
test_set = {2, 3, 5, 6, 7, 8, 9, 10, 22}


def _get_Args():
	parser = argparse.ArgumentParser()
	parser.add_argument("dir", help="Directory containing all the descriptors folders splited in training, validation and test. Inside those directories are expected to exist the folders with each level.")
	parser.add_argument("name", help="Name of fvec file to be created", default="cnnflow")
	parser.add_argument("level", type=int, help="Number of cnnflow pyramid levels")
	parser.add_argument("--outdir","-o", help="Directory to save merged descriptors in fvec file", default = '')
	parser.add_argument("--individual_levels","-i", type=int, help="If present the fvec will be build with only the level passed in level argument", default = '')
	return parser.parse_args()

def read_desc_file(name):
	
	desc = pd.read_csv(name, header=None)
	descriptor = desc.values
	return descriptor
	
def extract_action_and_type(label):

	type = -1
	# Extraio a posicao do numero da pessoa no label transformo em int e vejo em qual conjunto esta
	if(int(label[6:8]) in train_set):
		type = 0
	elif(int(label[6:8]) in validation_set):
		type = 1
	elif(int(label[6:8]) in test_set):
		type = 2
		
	action = -1
	
	if('boxing' in label):
		action = 0
	elif('handclapping' in label):
		action = 1
	elif('handwaving' in label):
		action = 2
	elif('jogging' in label):
		action = 3
	elif('running' in label):
		action = 4
	elif('walking' in label):
		action = 5
		
	return str(action), str(type)

def save_label(label, filename):
	
	with open(filename, 'a', encoding = encode) as file:
		
		action, type = extract_action_and_type(label)
		
		file.write(label + " " + type + " " + action + "\n")




def concat_desc(indir, name, outdir, level, individual):

	
	if not os.path.isdir(outdir) and outdir != '':
		os.makedirs(outdir)
	
	#name of file with all fvecs
	fvec_filename = os.path.join(outdir, name + '.fvec')
	#name of file with all labels sorted according with fvec
	labels_filename = os.path.join(outdir, 'label_' + name +'.txt')
	

	for set in ['training', 'validation', 'test']: # iteracao nos conjuntos de treinamento, validacao e teste

		files = []
		for i in range(level):
			# listando os arquivos de cada nivel
			files.append(os.listdir(os.path.join(indir, set, 'l'+str(i))))
		
		for i in range(len(files[0])): # itero sobre todos arquivos da pasta
		
			descriptors = np.array([])
			
			if not individual:
				for j in range(level):
					# leitura de cada um dos niveis do arquivo especificado
					descriptor = read_desc_file(os.path.join(indir, set, 'l'+str(j), files[j][i]) )
					# tranformando em um matriz linha
					descriptor = np.reshape(descriptor, (descriptor.shape[0]))
					# concatenando
					descriptors = np.concatenate((descriptors, descriptor))
					
				# convertendo para float32
				descriptors = descriptors.astype('float32')
				# removo a extensao
				file_name = os.path.splitext(files[j][i])[0]
				# escrevo no fvec 
				saveDescriptor(fvec_filename, descriptors)
				save_label(file_name, labels_filename)
				
			else:
				# leitura do unico nivel usado
				descriptor = read_desc_file(os.path.join(indir, set, 'l'+str(individual), files[individual][i]) )
				# tranformando em um matriz linha
				descriptor = np.reshape(descriptor, (descriptor.shape[0]))
				# concatenando
				descriptors = np.concatenate((descriptors, descriptor))
				
				# convertendo para float32
				descriptors = descriptors.astype('float32')
				# removo a extensao
				file_name = os.path.splitext(files[individual][i])[0]
				# escrevo no fvec 
				saveDescriptor(fvec_filename, descriptors)
				save_label(file_name, labels_filename)
	

		
def _main(args):
	concat_desc(args.dir, args.name, args.outdir, args.level, args.individual_levels)

if __name__ == '__main__':
	args = _get_Args()
	_main(args)
