import argparse
import cv2
import numpy as np
import os
from scipy.cluster.vq import kmeans2, whiten, kmeans
from sklearn import metrics, cluster
import time
import pandas as pd

encode = "utf-8"
threads_file = 'threads.txt'
np.random.seed(1)

def _get_Args():
	parser = argparse.ArgumentParser()
	parser.add_argument("dir", help="Directory of dataset reduced by pca")
	parser.add_argument("size", type=int, help="Size of codebook")
	parser.add_argument("-n", "--name", help="Codebook file name", nargs='?', const='codebook', default='codebook')
	parser.add_argument("-o", "--outdir", help="Output directory", nargs='?', const='', default='')
	parser.add_argument("-i", "--iterations", type=int, help="Number of iterations of kmeans algorithm")
	return parser.parse_args()

def read_pca(dir):

	try:
		'''
		Para cada arquivo no diretorio dir, faco o split do nome do arquivo por '.' (ponto) e, se o ultimo elemento apos o split for a 
		extensao que desejo, o nome o arquivo sera concatenado com o diretorio e vai compor a lista de nomes dos arquivos que serao abertos
		'''
		names = [os.path.join(dir, file) for file in os.listdir(dir) if os.path.splitext(file)[1] == '.pca']
		videos = []
		
		for name in names:
			with open(name, "r", encoding=encode) as file:
				# Quebro por fragmentos, depois por linha, e depois por valor
				pca_features = [[float(item) if item != '' else 0.0 for item in line.split(" ") ] for line in file.read().split("\n")]
				# Ultimo elemento sempre fica vazio por causa do ultimo \n
				#del pca_features[-1]
				videos.append(pca_features)
		
		return videos
		
	except Exception as e:
		print("A problem occurred trying to read pca file: ", e)
		print("With parameters: \n", vars(args))
		return 0
		
def get_threads():

	try:
		with open(threads_file, "r", encoding=encode) as input:
			# Reading number of threads
			threads = int(input.read())
			return threads
	except Exception as e:
		print("A problem ocurred trying to read the number of threads: ", e)
		
def get_levels(n):
	
	if n == 17:
		return 4
		
	elif n == 7:
		return 3
		
	elif n == 3:
		return 2
		
	elif n == 1:
		return 1
	else:
		return 0
		
def get_interval_by_level(n):

	if n == 0:
		return 0, 1
		
	elif n == 1:
		return 1, 3
		
	elif n == 2:
		return 3, 7
		
	elif n == 3:
		return 7, 17
	else:
		return 0,0
		
	
def create_codebook(pca_videos, size, iter, concatenate = 1):
	
	
	codebook = []
	video_concat = []
	
	if concatenate > 1:
		for k, video in enumerate(pca_videos):
			temp_video = np.empty((len(video) // concatenate, len(video[0]) * concatenate))
			
			for j in range(0, len(video), concatenate):
				temp = np.array([video[j]])
				
				for i in range(1, concatenate):
					temp = np.concatenate((temp, [video[j + i]]), axis = 1)
					
				temp_video[ j // concatenate] = temp
				
			video_concat.append(temp_video)
	else:
		video_concat = pca_videos

	num_videos  = len(video_concat)
	#print("Num videos:", num_videos)
	
	# Inicializo o empilhamento de descritores
	
	descriptors = np.array(video_concat[0])
	#descriptors = np.reshape(descriptors, (descriptors.shape[0], 100))
	#print(descriptors.shape)
	# Para cada video
	for j in range(1, num_videos):
		# Transformo cnnflow reduzida em um np.array
		descriptor = np.array(video_concat[j])
		#print(descriptor.shape)
		# Empilho os descritores
		descriptors = np.vstack((descriptors, descriptor))
	
	
	
	# scipy kmeans que da problema e as vezes retorno codebooks com menos centroids do que o parametro passado
	# Rodo o kmeans
	# Em ordem os parâmetros passados são: dados, numero de clusters, numero de iteracoes
	#descriptors = whiten(descriptors)
	#print(time.ctime(), " Size before kmeans execution: ", size)
	#print(time.ctime(), " Descriptors shape: ", descriptors.shape)
	#voc, variance = kmeans(descriptors, size, iter, thresh=0.0000001, check_finite=True)
	#print(time.ctime(), " Size after kmeans execution: ", size)
	#print(time.ctime(), " VOC size: ", len(voc))
	
	# abordagem scipy.kmeans2 rodando 10 vezes e pegando o com melhor coeficiente de silhueta
	'''best_s = -2
	best_centroids = []
	for i in range(iter):
		centroids, labels = kmeans2(descriptors, size, 300, minit='random', missing='warn')
		score = metrics.silhouette_score(descriptors, labels, metric='euclidean')
		print("Silhouette coefficient of codebook: ", score)
		if score > best_s:
			best_s = score
			best_centroids = centroids
		
	print("Best silhouette found: ", best_s)'''
	
	# abordagem simples do scipy.kmeans2
	centroids, labels = kmeans2(descriptors, size, iter, minit='random', missing='warn')

	
	# abordagem do sklearn.kmeans que ocupa mais memeoria, porem pega o melhor dentro de um numero de inicializacoes
	#model = cluster.KMeans(n_clusters=size, init='random', n_init = iter, n_jobs=get_threads()).fit(descriptors)
	#labels = model.labels_
	#centroids = model.cluster_centers_
	#codebook = model.cluster_centers_
	#del descriptors
	codebook = centroids
	return codebook
	
def write_codebook(name, outdir, codebook):

	
	try:	
		# If path doesn't exists, make it
		if not os.path.isdir(outdir) and outdir != '':
			os.makedirs(outdir)

		out_file = os.path.join(outdir, name) + ".dic"

		# With automatically closes output
		with open(out_file, "w", encoding=encode) as output:
			# Casting np.array to list then cast elements to str, join them with space and finally join rows with \n
			output.write("\n".join([" ".join(list(map(str, line))) for line in codebook.tolist()]))
			
		return 0
		
	except Exception as e:
		print("Some error occurred while writing codebook into file: ", e)
		return 1
	
def _main(args):
	
	pca_videos = read_pca(args.dir)
	if not args.iterations:
		args.iterations = 10
	codebook = create_codebook(pca_videos, args.size, args.iterations)
	if not args.outdir:
		args.outdir = ''
	if not args.name:
		args.name = "codebook"
	
	write_codebook(args.name, args.outdir, codebook)
	
	
if __name__ == '__main__':
	# parse arguments
	begin_time = time.ctime()
	args = _get_Args()
	_main(args)
	end_time = time.ctime()
	
