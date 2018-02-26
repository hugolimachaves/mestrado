import argparse
import cv2
import numpy as np
import os
from kmeans import read_pca, create_codebook, write_codebook
from histogram import create_histogram, write_histogram, read_codebook
import time
from multiprocessing import Pool
import re

encode = "utf-8"
threads_file = 'threads.txt'

def _get_Args():
	parser = argparse.ArgumentParser()
	group = parser.add_mutually_exclusive_group()
	parser.add_argument("dir", help="Directory of dataset reduced by pca divided in training, validation and test directories. Inside those directories are expected to exist the folders with each level.")
	parser.add_argument("level", type=int, help="Number of cnnflow pyramid levels")
	parser.add_argument("size", type=int, help="Size of codebook")
	parser.add_argument("-o", "--outdir", help="Output directory of histograms", nargs='?', const='', default='')
	group.add_argument("-s", "--savecodes", help="The codebooks will be saved in disk with the informed name", nargs='?', const='codebook', default='codebook')
	group.add_argument("-l", "--loadcodes", help="The codebooks will be loaded from disk from the informed directory")
	parser.add_argument("-i", "--iterations", type=int, help="Number of iterations of kmeans algorithm", default=10)
	parser.add_argument("-c", "--concatenate", help="Concatenation option with codebooks with different sizes", action = 'store_true')
	parser.add_argument("-c2", "--concatenate2", help="Concatenation option with 17 histograms", action = 'store_true')
	return parser.parse_args()

def get_threads():

	try:
		with open(threads_file, "r", encoding=encode) as input:
			# Reading number of threads
			threads = int(input.read())
			return threads
	except Exception as e:
		print("A problem ocurred trying to read the number of threads: ", e)
		
	
def create_all_histograms(pca_videos, codebooks, concatenate = 1, concatenate2 = 1):

	video_histograms = []
	
	'''with Pool(get_threads()) as p:
		histograms = list(p.map(create_histogram, pca_videos, [codebooks]*len(pca_videos)))
		video_histograms.append(histograms)
	'''
	for pca in pca_videos:
		histograms = []
		if concatenate2 > 1:
			for i in range(concatenate2):
				if i == 0:
					histograms = create_histogram(pca[i::concatenate2], codebooks, concatenate)
				else:
					histograms = np.concatenate((histograms, create_histogram(pca[i::concatenate2], codebooks, concatenate)), axis = 0)
		else:
			histograms = create_histogram(pca, codebooks, concatenate)
		 
		video_histograms.append(histograms)
	
	return video_histograms
		
def write_all_histograms(dir, video_histograms, outdir):
	
	# Extraio os nomes dos arquivos de entrada sem a extensao pra usar como nome dos arquivos de saida
	name_and_hist = []
	names = [os.path.basename(os.path.splitext(file)[0]) for file in os.listdir(dir) if os.path.splitext(file)[1] == '.pca']
	'''with Pool(get_threads()) as p:
		p.map(write_histogram, names, [outdir]*len(names), video_histograms)
		name_and_hist.append({'name': name, 'histogram': histogram})'''
	for name, histogram in zip(names, video_histograms):
		write_histogram(name, outdir, histogram)
		name_and_hist.append({'name': name, 'histogram': histogram})
	
	return name_and_hist
		
		
def BoF_for_each_level(level, indir, codebook_size, kmeans_iterations, concatenate = False):

	codebooks = []
	pca_training = []

	print(time.ctime(), ' Reading training pca videos...')
	concat = 1
	
	for i in range(level):
	
		print(time.ctime(), " Level ",str(i))
		# Diretorio com as cnnflows reduzidas por pca de cada nivel
		pca_dir = os.path.join(indir, 'training', 'l'+str(i))
		pca_videos = read_pca(pca_dir)
		print(time.ctime(), ' OK')
	
		
		print(time.ctime(), ' Creating codebook with size', codebook_size,'...')
		if concatenate:
			if i < 3:
				concat = 2 ** i
			else:
				concat = 10
		
		codebook = create_codebook(pca_videos, codebook_size, kmeans_iterations, concat)
		print(time.ctime(), ' OK size: ',codebook_size)
		
		pca_training.append(pca_videos)
		codebooks.append(codebook)
		
	return codebooks, pca_training
	
def read_pca_validation_and_test(level, indir):

	pca = {'validation':[], 'test':[]}
	
	for set in ['validation', 'test']:
		print(time.ctime(), ' Reading ',set,' pca videos...')
		for i in range(level):
			
			print(time.ctime(), " Level ",str(i))
			# Diretorio com as cnnflows reduzidas por pca de cada nivel e conjunto
			pca_dir = os.path.join(indir, set, 'l'+str(i))
			pca_videos = read_pca(pca_dir)
			print(time.ctime(), ' OK')
			
			pca[set].append(pca_videos)
			
	return pca['validation'], pca['test']
	
def save_codebook_for_each_level(codebooks, codebook_name, outdir, iterations):
	
	outdir = os.path.join(outdir, 'codebooks')
	for i in range(len(codebooks)):
		# Nome do codebook seguido: por l e o numero do nivel, s e o tamanho do codebook e i e o numero de iteracoes
		cb_name = codebook_name + '_l' + str(i) +'_s' + str(len(codebooks[i])) + '_i' + str(iterations)
		print(time.ctime(), ' Saving ',cb_name,'...')
		write_codebook(cb_name, outdir, codebooks[i])
		print(time.ctime(), ' OK')
		
def create_and_write_histogram_for_each_set(codebooks, pca_sets, indir, outdir, concatenate = False, concatenate2 = False):

	all_histograms = {'training':[], 'validation':[], 'test':[]}
	concat = 1
	concat2 = 1
	
	for set in ['training', 'validation', 'test']:
		
		i = 0
		for level, codebook in zip(pca_sets[set], codebooks):
		
			if concatenate:
				if i < 3:
					concat = 2 ** i
				else:
					concat = 10
					
			if concatenate2:
				if i < 3:
					concat2 = 2 ** i
				else:
					concat2 = 10
		
			print(time.ctime(), ' Creating histograms of all videos in set ',set,' level ',str(i),'...')
			video_histograms = create_all_histograms(level, codebook.tolist(), concat, concat2)
			print(time.ctime(), ' OK')
			
			print(time.ctime(), ' Writing all histrograms created...')
			
			out_dir = os.path.join(outdir, set, 'l'+str(i))
			in_dir = os.path.join(indir, set, 'l'+str(i))
			name_and_hist = write_all_histograms(in_dir, video_histograms, out_dir)
			print(time.ctime(), ' OK')
			
			all_histograms[set].append(name_and_hist)
			i += 1
		
	return all_histograms
	
def load_codebooks(level, dir):
	
	print(time.ctime(), ' Loading codebooks...')
	
	dic_files = [file for file in os.listdir(dir) if os.path.splitext(file)[1] == '.dic']
	
	# Ordenando pelo level
	r = re.compile(r'_l(\d)_')
	dic_files = sorted(dic_files, key=lambda x:r.search(x).group(0))
	
	codebooks = [read_codebook(os.path.join(dir, file)) for file in dic_files]
	
	print(time.ctime(), ' OK')
	
	return np.array(codebooks)
	
def read_pca_training(level, indir):

	pca = []
	
	print(time.ctime(), ' Reading training pca videos...')
	
	for i in range(level):
		
		print(time.ctime(), " Level ",str(i))
		# Diretorio com as cnnflows reduzidas por pca de cada nivel do conjunto training
		pca_dir = os.path.join(indir, 'training', 'l'+str(i))
		pca_videos = read_pca(pca_dir)
		print(time.ctime(), ' OK')
		
		pca.append(pca_videos)
			
	return pca
	
	
def _main(args):

	print(time.ctime(), " Running Bag of features with following arguments: ", str(vars(args)))
	
	pca_sets = {'training':[],'validation':[],'test':[]}
	
	codebooks = []
	
	if not args.loadcodes:
		# Faco o bag para cada nivel do conjunto de treino e salvo os codigos e os arquivos pca para criar os histogramas depois
		codebooks, pca_sets['training'] = BoF_for_each_level(args.level, args.dir, args.size, args.iterations, args.concatenate)
		
		# Gravo os codebooks em disco
		save_codebook_for_each_level(codebooks, args.savecodes, args.outdir, args.iterations)
	else:
		pca_sets['training'] = read_pca_training(args.level, args.dir)
		codebooks = load_codebooks(args.level, args.loadcodes)
	
	
	
	# Leio os arquivos do pca dos outros conjuntos para criar os histogramas tambem
	pca_sets['validation'], pca_sets['test'] = read_pca_validation_and_test(args.level, args.dir)
	
	# Crio e gravo os histogramas de todos os conjuntos
	create_and_write_histogram_for_each_set(codebooks, pca_sets, args.dir, args.outdir, args.concatenate, args.concatenate2)	
	
	
if __name__ == '__main__':
	# parse arguments
	args = _get_Args()
	_main(args)
