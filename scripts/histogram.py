import argparse
import cv2
import numpy as np
import os
from scipy.cluster.vq import kmeans2, vq
import pandas as pd

encode = "utf-8"

def _get_Args():
	parser = argparse.ArgumentParser()
	parser.add_argument("name", help="Video file name reduced by pca")
	parser.add_argument("codename", help="Codebook file name")
	parser.add_argument("-o", "--outdir", help="Output directory")
	return parser.parse_args()

def read_pca_as_np(name):

	try:
		with open(name, "r", encoding=encode) as file:
			# Quebro por fragmentos, depois por linha, e depois por valor
			pca_features = [[float(item) if item != '' else 0.0 for item in line.split(" ") ] for line in file.read().split("\n")]
		
		return np.array(pca_features)
		
	except Exception as e:
		print("A problem occurred trying to read pca file: ", e)
		print("With parameters: \n", vars(args))
		return 0
		
def read_codebook(name):

	try:
		codebook = []
		
		with open(name, "r", encoding=encode) as file:
			# Quebro por pontos e depois por valor
			codebook = [[float(value) if value != '' else 0.0 for value in point.split(" ")] for point in file.read().split("\n")]
	
		return codebook
		
	except Exception as e:
		print("A problem occurred trying to read codebook file: ", e)
		print("With parameters: \n", vars(args))
		return 0

# retorna o nivel de acordo com o numero de cnnflows em um fragmento
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
		
# retorna o intervalo correspondente de cnnflows para um determinado nivel
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
		
	
def create_histogram(pca, codebook, concatenate = 1):
	
	#num_snippet  = len(pca)
	video_concat = []
	
	if concatenate > 1:
			video_concat = np.empty((len(pca) // concatenate, len(pca[0]) * concatenate))
			
			for j in range(0, len(pca), concatenate):
				temp = np.array([pca[j]])
				
				for i in range(1, concatenate):
					temp = np.concatenate((temp, [pca[j + i]]), axis = 1)
					
				video_concat = temp
				
	else:
		video_concat = pca
	
	# Crio um np.array com dimensoes numero de niveis e quantidade de pontos por cluster
	histogram = np.zeros((len(codebook)), "int16")
			
	words, distance = vq(video_concat, codebook)
	for w in words:
		histogram[w] += 1

	return histogram
	
def write_histogram(name, outdir, histogram):
	
	hist_dataframe = pd.DataFrame(histogram)
	
	try:	
		# If path doesn't exists, make it
		if not os.path.isdir(outdir):
			os.makedirs(outdir)

		out_file = os.path.join(outdir, name) + ".desc"
		
		hist_dataframe.to_csv(out_file, header=None, index=False)
		
		# With automatically closes output
		#with open(out_file, "w", encoding=encode) as output:
			# Joining 
		#	output.write(" ".join(list(map(str, histogram.tolist()))))
		
		return 0
		
	except Exception as e:
		print("Some error occurred while writing histogram into file: ", e)
		return 1
	
def _main(args):
	
	pca = read_pca_as_np(args.name)
	codebook = read_codebook(args.codename)
	histogram = create_histogram(pca, codebook)
	if not args.outdir:
		args.outdir = ''
	name = os.path.basename(os.path.splitext(args.name)[0])
	write_histogram(name, args.outdir, histogram)
	
	
if __name__ == '__main__':
	# parse arguments
	args = _get_Args()
	_main(args)
