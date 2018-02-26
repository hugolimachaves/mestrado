import cv2 as cv
import numpy as np
import pca
import argparse
import os
from DescrFVEC import saveDescriptor


encode = "utf-8"

train_set = {11, 12, 13, 14, 15, 16, 17, 18}
validation_set = {1, 4, 19, 20, 21, 23, 24, 25}	
test_set = {2, 3, 5, 6, 7, 8, 9, 10, 22}


def _get_Args():
	parser = argparse.ArgumentParser()
	parser.add_argument("path", help="path containing all the descriptors folders")
	parser.add_argument("--outdir","-o", help="path containing merged descriptors", default = "all")
	parser.add_argument("--level_0", "-l0",help="level 0 folder's name", default = 'nivel0')
	parser.add_argument("--level_1", "-l1",help="level 1 folder's name", default = 'nivel1')
	parser.add_argument("--level_2", "-l2",help="level 2 folder's name", default = 'nivel2')
	parser.add_argument("--level_3", "-l3",help="level 3 folder's name", default = 'nivel3')
	return parser.parse_args()

def read_desc_file(name):
	
	desc = open(name, "r")
	descriptor = desc.readlines()
	desc.close()
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




def concat_desc(path, outdir, level_0, level_1, level_2, level_3):

	
	if not os.path.isdir(outdir) and outdir != '':
		os.makedirs(outdir)
	
	#name of file with all fvecs
	fvec_filename = os.path.join(outdir, 'kth_cnnflow.fvec')
	#name of file with all labels sorted according with fvec
	labels_filename = os.path.join(outdir, 'label_kth_cnnflow.txt')
	level = []
	caminho = []
	level.append(level_0)
	level.append(level_1)
	level.append(level_2)
	level.append(level_3)
	
	

	for j in range(4): # um laço para cada nivel para obter o caminhos para as pastas de cada nivel
		# juntando as pastas com o separador dado pelo so
		caminho.append(os.path.join(str(path), str(level[j]))) # caminho para cada nivel

	for i in range (len(  os.listdir(  os.path.join(str(path), str(level[0])) ))): # laço para o numero de arquivos que tem na pasta ( todas as pastas tem o mesmo numeor de arquivos)

		new_descriptor = [] #descritor com todos os nives
		for j in caminho: # percorrer os quatro niveis

			#descriptor: descritor de cada nivel
			descriptor = read_desc_file( os.path.join(j, os.listdir(j)[i]) ) # leitura de cada um dos niveis do arquivo especificado
			for k in range(len(descriptor)):

				new_descriptor.append(float(descriptor[k][0]))	

		new_descriptor_np = np.array( new_descriptor,'float32' )
		files_name = os.listdir(j)[i]
		#tirar sufixo
		files_name = files_name.split('.')[0]
		#files_name = files_name + '.fvec'
		saveDescriptor(fvec_filename, new_descriptor_np)
		save_label(files_name, labels_filename)

		'''file = open(outdir + files_name + '.fvec',"wb") # gravar no diretorio de saida com os nomes dos arquivos

		for i in range(new_descriptor_np.shape[1]): 


			to_write = bytes([new_descriptor[i]])
			file.write(to_write)

		file.close()'''

		del new_descriptor
	

		
def _main(args):
	concat_desc(args.path, args.outdir, args.level_0, args.level_1, args.level_2, args.level_3)

if __name__ == '__main__':
	args = _get_Args()
	_main(args)
