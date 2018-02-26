import projecaov2
import multiprocessing as mp
import argparse
import os
import pca
import numpy

arquivoAutovetores = '_eigenVectors_.pcab'
arquivoMedia = '_mean_.pcab'
programa = 'projecaov2.py'

def _get_Args():
	parser = argparse.ArgumentParser()
	parser.add_argument('arquivo', help= 'arquivo e caminho para o txt como todos os outros caminhos')
	parser.add_argument('nFeatures', help = 'Numero de features apos o pca')
	return parser.parse_args()

def lerCaminhos2(arquivo):
	file = open(arquivo)
	caminhos2 = file.read().split('\n')
	file.close()
	return caminhos2

def separarCaminhos(caminhos2):
	caminhoBases = []
	caminhoLeituraArquivos = []
	caminhoEscritaArquivos = []
	contador = 0
	for i in caminhos2:
		if 'bases' in i:
			contador = 1
			continue
		if 'leitura' in i:
			contador = 2
			continue
		if 'escrita' in i:
			contador = 3
			continue
		if contador == 1:
			caminhoBases.append(i)
		if contador == 2:
			caminhoLeituraArquivos.append(i)
		if contador == 3:
			caminhoEscritaArquivos.append(i)
	return caminhoBases, caminhoLeituraArquivos, caminhoEscritaArquivos

def paralelizar(comando):
	nThreads = len(comando)
	pool = mp.Pool(processes=nThreads)
	pool.map(os.system, comando)
	pool.close()
	pool.join()

def _main(args):
	pathBase, pathCaminho, pathEscrita = separarCaminhos(lerCaminhos2(args.arquivo)) # lista com com caminhos para a base do pca, onde arquivo para ler e o arquivos de projecao
	
	#argumentos.append(pathBase, pathCaminho, pathEscrita, arquivoMedia, arquivoAutovetores)
	comandos = []
	print("caminhos para a base: ",pathBase)
	print("numeros de bases :", len(pathBase))
	for i in range(len(pathBase)):
		argumentos = []
		argumentos.append('python')
		argumentos.append(programa)
		argumentos.append(pathBase[i])
		argumentos.append(pathCaminho[i])
		argumentos.append(pathEscrita[i])
		argumentos.append(arquivoMedia)
		argumentos.append(arquivoAutovetores)
		argumentos.append(args.nFeatures)
		comandos.append(str.join(' ',argumentos))
	paralelizar(comandos)

if __name__ == '__main__':
	args = _get_Args()
	_main(args)
