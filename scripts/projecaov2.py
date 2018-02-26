import multiprocessing as mp
import argparse
import os
import pca
import numpy

def lerBases(caminhoBases,arquivoAutovetores,arquivoMedia,nFeatures):
	path2ev = os.path.join(caminhoBases,arquivoAutovetores)
	file_eg = open(path2ev , "r" )
	lines = file_eg.readlines()
	file_eg.close()
	eigenVectors = pca.file2PCA(lines,nFeatures)
	path2mean = os.path.join(caminhoBases,arquivoMedia)
	file_mean = open(path2mean ,"r")
	lines = file_mean.readlines()
	file_mean.close()
	mean = pca.file2PCA(lines,nFeatures)
	return mean, eigenVectors

def reduzirVetores(caminhoBases, caminhoLeituraArquivos, caminhoEscritaArquivos, arquivoMedia, arquivoAutovetores, nFeatures):
	files = os.listdir(caminhoLeituraArquivos)
	media, autoVetores = lerBases(caminhoBases,arquivoAutovetores,arquivoMedia,nFeatures)
	for i in files:
		if '.cnnf' in i:
			cnn_flow = pca.read_file(os.path.join(caminhoLeituraArquivos,i)) #reading the content fo the file
			cnn_flow_padronizado = pca.file2PCA(cnn_flow) #conforming it to apply PCA
			projection = pca.projecao_PCA(cnn_flow_padronizado, media, autoVetores)
			pca.write_pca_reduction(i,projection,caminhoEscritaArquivos) #terminar de fazer isso aqui

def _get_Args():
	parser = argparse.ArgumentParser()
	parser.add_argument('caminhoBases', help= 'caminho para o arquivo de bases ortonormais e o vetor de media')
	parser.add_argument('caminhoLeitura', help= 'caminhos para os arquivos que serao reduzidos')
	parser.add_argument('caminhoEscrita', help='caminho onde os arquivos reduzidos serao escritos')
	parser.add_argument('arquivoMedia', help='nome do arquivo que contem o vetor de medias')
	parser.add_argument('arquivoAutovetores', help='nome do arquivo que contem a base de vetores ortonormais')
	parser.add_argument('nFeatures', help = 'Numero de features após o pca', type = int)
	return parser.parse_args()

def _main(args):
	reduzirVetores(args.caminhoBases, args.caminhoLeitura, args.caminhoEscrita, args.arquivoMedia, args.arquivoAutovetores,args.nFeatures)

if __name__ == '__main__':
	args = _get_Args()
	_main(args)