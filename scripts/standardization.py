import argparse
import numpy as np
import os
import time
from multiprocessing import Pool
import pandas as pd
from sklearn import preprocessing
import sys


encode = "utf-8"
threads_file = 'threads.txt'
sets = ['training', 'validation', 'test']

def _get_Args():
	parser = argparse.ArgumentParser()
	parser.add_argument("indir", help="Directory with all dataset files splitted in traning, validation and test folders")
	parser.add_argument("outdir", help="Output directory to write the modified features")
	parser.add_argument("ext", help="Extension of file to normalize")
	parser.add_argument("type", help="Normalization type", choices=['max_min', 'standard'])
	parser.add_argument("-l", "--level", type = int, help="Pyramid level of cnnflows", default=4)
	return parser.parse_args()
	
def get_threads():

	try:
		with open(threads_file, "r", encoding=encode) as input:
			# Reading number of threads
			threads = int(input.read())
			return threads
	except Exception as e:
		print("A problem ocurred trying to read the number of threads: ", e)

def read_fc7(name):
	
	fc7 = pd.read_csv(name, header=None)
	return fc7.values
	
def write_fc7(name, fc7):
	df = pd.DataFrame(fc7)
	df.to_csv(name, sep=',', line_terminator='\n', encoding=encode, header=None, index=False)
	
def read_cnnf_as_np(name):

	try:
		with open(name, "r", encoding=encode) as file:
			# Quebro por fragmentos, depois por linha, e depois por valor
			cnnf_features = [[[float(item) if item != '' else 0.0 for item in line.split(" ") ] for line in snippet.split("\n") ] for snippet in file.read().split("\n\n")]
			# Ultimo elemento sempre fica vazio por causa do ultimo \n
			#del pca_features[-1]
		
		return np.array(cnnf_features)
		
	except Exception as e:
		print("A problem occurred trying to read pca file: ", e)
		print("With parameters: \n", vars(args))
		return 0
	
def write_cnnf(name, cnnf):

	
	try:	
		# With automatically closes output
		with open(name, "w", encoding=encode) as output:
			# Joining cnn flows elements with space and then joining cnn flows with \n and finally joining snippets with \n\n
			output.write("\n\n".join(["\n".join([" ".join(list(map(str, j))) for j in i]) for i in cnnf]))
			
		return 0
		
	except Exception as e:
		print("Some error occurred while writing cnnflow pyramid into file: ", e)
		return 1
	
'''
Lista e le os arquivos dentro do diretorio passado em @indir com nomes terminados com a extensao .fc7
Eh esperado que dentro de @indir exsitam as pastas de treino, validacao e teste
Como saida sao retornados dois dicionarios (cada chave do dicionario se refere a uma pasta e em cada valor existem listas):
um com os dados de cada arquivo listado e 
o outro com os nomes prontos para serem usados na gravacao dos arquivos (depois de processados) no diretorio de saida @outdir
'''
	
def read_fc7_files(indir, outdir, set):

	in_fc7_files = []
	out_fc7_files = []
		
	# reading fc7 filenames
	dir = os.path.join(indir, set)
	fc7_files = [file for file in os.listdir(dir) if os.path.splitext(file)[1] == '.fc7']
	# reading fc7 files
	in_fc7_files = np.array([read_fc7(os.path.join(dir, file)) for file in fc7_files])
	
	# creating output dir
	dir = os.path.join(outdir, set)
	if not os.path.isdir(dir):
		os.makedirs(dir, exist_ok = True)
		
	# joining with output dir
	out_fc7_files = [os.path.join(dir, file) for file in fc7_files]
		
	return in_fc7_files, out_fc7_files
	
'''
Lista arquivos dentro do diretorio passado em @indir com nomes terminados com a extensao .cnnf
Eh esperado que dentro de @indir exsitam as pastas de treino, validacao e teste, 
e que dentro de cada uma destas existam outras @level pasta dividindo os arquivos em niveis (nivel0, nivel1, nivel2, nivel3)
Como saida sao retornados dois dicionarios (cada chave do dicionario se refere a uma pasta e em cada valor existem listas para cada @level)::
um com os dados de cada arquivo listado e 
o outro com os nomes prontos para serem usados na gravacao dos arquivos (depois de processados) no diretorio de saida
'''
	
def read_cnnflow_files(indir, outdir, level, set):

	in_cnnflow_files = []
	out_cnnflow_files = []

	for i in range(level):
		# reading cnnflow filenames
		dir = os.path.join(indir, set, "nivel"+str(i))
		
		cnnflow_files = [file for file in os.listdir(dir) if os.path.splitext(file)[1] == '.cnnf']
		
		# joinning with input dir
		in_cnnflow_files.append(np.array([read_cnnf_as_np(os.path.join(dir, file)) for file in cnnflow_files]))
		
		# creating output dir
		dir = os.path.join(outdir, set, "nivel"+str(i))
		if not os.path.isdir(dir):
			os.makedirs(dir, exist_ok = True)
			
		# joining with output dir
		out_cnnflow_files.append([os.path.join(dir, file) for file in cnnflow_files])
		
	return in_cnnflow_files, out_cnnflow_files
	
	
	
def write_fc7_files(files, names, set):

	print(time.ctime(), " Writing files ")

	print(time.ctime(), " Writing set ", set)
	for file, name in zip(files, names):
		write_fc7(name, file)

			
			
def write_cnnflow_files(files, names, level, set):

	print(time.ctime(), " Writing files ")

	print(time.ctime(), " Writing set ", set)
	for i in range(level):
		print(time.ctime(), " Writing level ", str(i))
		for file, name in zip(files[i], names[i]):
			write_cnnf(name, file)
				
				
	
	
def normalize(type, indir, outdir, ext, level):

	if not os.path.isdir(outdir):
		os.makedirs(outdir, exist_ok = True)
		
	# Criacao de um arquivo de log, depois tenho que alterar isso para fazer um log de verdade de toda execucao
	
	with open(os.path.join(args.outdir, 'log.txt'), "w", encoding=encode) as file:
		file.write('Arquivos dessa pasta foram criados com a execucao do standardization com os seguintes parametros: \n'+str(vars(args)))
	
	
	if type == 'max_min':
		
		# Criando objeto que cuidara da tranformacao
		min_max_scaler = preprocessing.MinMaxScaler()
		
		if ext == '.fc7':
			for set in sets:
				print(time.ctime(), " Reading files of set ", set)
				# Lendo os arquivos de entrada e criando o nome dos arquivos de saida
				in_files, out_filenames = read_fc7_files(indir, outdir, set)
				
				if set == 'training':
					# Criando tranformacao a partir dos dados de treino
					min_max_scaler.fit(np.concatenate(in_files, axis=0))
				
				print(time.ctime(), " Trasnforming files of set", set)
				
				# Aplicando transformacao nos outros conjuntos de dados
				print(time.ctime(), " Tansforming set ", set)
				for i in range(len(in_files)):
					in_files[i] = min_max_scaler.transform(in_files[i])
					
					
				write_fc7_files(in_files, out_filenames, set)
		
		
		elif ext == '.cnnf':
			
			for set in sets:
				print(time.ctime(), " Reading files of set", set)
				# Lendo os arquivos de entrada e criando o nome dos arquivos de saida
				in_files, out_filenames = read_cnnflow_files(indir, outdir, level, set)
				
				# para cada level
				for k in range(level):
					print(time.ctime(), " Tansforming level ", str(k))

					if set == 'training':
						try:
							# filtrando linhas que podem estar vazias e causar problemas nesse ponto
							in_files[k] = np.array([ np.array([item for item in file if np.sum(abs(np.array(item))) != 0]) for file in in_files[k]])
							# junto os dados de todos arquivos de treino
							temp = np.concatenate(in_files[k], axis=0)
							# junto os dados de todos snippets
							temp = temp.reshape((temp.shape[0]*temp.shape[1], temp.shape[2]))
							# Criando tranformacao a partir dos dados de treino
							min_max_scaler.fit(temp)
						except ValueError:
							print("Shape do problema:", in_files[k].shape)
							for item in in_files[k]:
								print("Shape de cada cara:",item.shape)
							sys.exit()
					
					print(time.ctime(), " Trasnforming files of set", set)
					# Aplicando transformacao nos outros conjuntos de dados
					# para cada conjunto
					# filtrando linhas que podem estar vazias e causar problemas nesse ponto
					in_files[k] = np.array([ np.array([item for item in file if np.sum(abs(np.array(item))) != 0]) for file in in_files[k]])
					# para cada arquivo
					for i in range(len(in_files[k])):
						# para cada snippet
						for j in range(len(in_files[k][i])):
							in_files[k][i][j] = min_max_scaler.transform(in_files[k][i][j])
					
			
				write_cnnflow_files(in_files, out_filenames, level, set)
			
		
		
	elif type == 'standard':
		
		# Criando objeto que cuidara da tranformacao
		std_scaler = preprocessing.StandardScaler()
		
		if ext == '.fc7':
			for set in sets:
				print(time.ctime(), " Reading files of set ", set)
				# Lendo os arquivos de entrada e criando o nome dos arquivos de saida
				in_files, out_filenames = read_fc7_files(indir, outdir, set)
				
				if set == 'training':
					# Criando tranformacao a partir dos dados de treino
					std_scaler.fit(np.concatenate(in_files, axis=0))
				
				print(time.ctime(), " Trasnforming files of set", set)
				
				# Aplicando transformacao nos outros conjuntos de dados
				for i in range(len(in_files)):
					in_files[i] = std_scaler.transform(in_files[i])
					
					
				write_fc7_files(in_files, out_filenames, set)
		
		
		elif ext == '.cnnf':
			# para cada level
			
			for set in sets:
				print(time.ctime(), " Reading files of set", set)
				# Lendo os arquivos de entrada e criando o nome dos arquivos de saida
				in_files, out_filenames = read_cnnflow_files(indir, outdir, level, set)
				
				for k in range(level):
					print(time.ctime(), " Tansforming level ", str(k))

					if set == 'training':
						try:
							# filtrando linhas que podem estar vazias e causar problemas nesse ponto
							in_files[k] = np.array([ np.array([item for item in file if np.sum(abs(np.array(item))) != 0]) for file in in_files[k]])
							# junto os dados de todos arquivos de treino
							temp = np.concatenate(in_files[k], axis=0)
							# junto os dados de todos snippets
							temp = temp.reshape((temp.shape[0]*temp.shape[1], temp.shape[2]))
							# Criando tranformacao a partir dos dados de treino
							std_scaler.fit(temp)
						except ValueError:
							print("Shape do problema:", in_files[k].shape)
							for item in in_files[k]:
								print("Shape de cada cara:",item.shape)
							sys.exit()
					
					print(time.ctime(), " Trasnforming files of set", set)
					
					# Aplicando transformacao nos outros conjuntos de dados
					# para cada conjunto
					# filtrando linhas que podem estar vazias e causar problemas nesse ponto
					in_files[k] = np.array([ np.array([item for item in file if np.sum(abs(np.array(item))) != 0]) for file in in_files[k]])
					# para cada arquivo
					for i in range(len(in_files[k])):
						# para cada snippet
						for j in range(len(in_files[k][i])):
							in_files[k][i][j] = std_scaler.transform(in_files[k][i][j])
					
			
				write_cnnflow_files(in_files, out_filenames, level, set)
			
	
def _main(args):
	normalize(args.type, args.indir, args.outdir, args.ext, args.level)
	
if __name__ == '__main__':
	# parse arguments
	args = _get_Args()
	_main(args)