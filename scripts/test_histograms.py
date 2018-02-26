import numpy as np
import argparse
import os
import pandas as pd
from merge_files_to_fvec import read_desc_file
from histogram import write_histogram, read_codebook

encode = "utf-8"

train_set = {11, 12, 13, 14, 15, 16, 17, 18}
validation_set = {1, 4, 19, 20, 21, 23, 24, 25}	
test_set = {2, 3, 5, 6, 7, 8, 9, 10, 22}


def _get_Args():
	parser = argparse.ArgumentParser()
	parser.add_argument("dir", help="Directory with training histograms")
	parser.add_argument("codedir", help="Directory with codebooks")
	parser.add_argument("name", help="Name of statistics file", default="histogram_statistics")
	parser.add_argument("level", type=int, help="Number of cnnflow pyramid levels")
	parser.add_argument("--outdir","-o", help="Directory to save merged descriptors in fvec file", default = '')
	return parser.parse_args()
	
def read_all_desc(indir, level):

	descriptors = []
	
	for i in range(level):
		folder = 'l'+str(i)
		level_dir = os.path.join(indir, folder)
		names = [os.path.join(level_dir, file) for file in os.listdir(level_dir) if os.path.splitext(file)[1] == '.desc']
		temp = read_desc_file(names[0])
		temp = np.reshape(temp, (temp.shape[1], temp.shape[0]))
		desc_level = np.array(temp)
		
		for i in range(1,len(names)):
			desc = read_desc_file(names[i])
			desc = np.reshape(desc, (desc.shape[1], desc.shape[0]))
			desc_level = np.vstack((desc_level, desc))
		
		descriptors.append(desc_level)
		
	return descriptors
	
def read_all_codebook(indir, level):

	names = [os.path.join(indir, file) for file in os.listdir(indir) if os.path.splitext(file)[1] == '.dic']
	
	codebooks = [None, None, None, None]
	
	for file in names:
		codebook = read_codebook(file)
		for i in range(level):
			if 'l'+str(i) in os.path.basename(file):
				codebooks[i] = codebook
	
	return codebooks
		
def sum_frequencies(descriptors, level):

	sum = []
	for i in range(level):
		sum.append(np.sum(descriptors[i], 0))
		
	return sum
	
def write_statistics(results, level, outdir, name, codebooks):
	
	for i in range(level):
		temp = np.sort(results[i])
		indices = np.argsort(results[i])
		
		code = codebooks[i]
		code = np.array([x for _,x in sorted(zip(indices,code))])
		
		print("Shape of temp:", temp.shape)
		print("Shape of indices:", indices.shape)
		temp = np.reshape(temp,(temp.shape[0], 1))
		indices = np.reshape(indices, (indices.shape[0], 1))
		#code = np.reshape(code, (code.shape[0], 1))
		print("Shape of temp reshaped:", temp.shape)
		print("Shape of indices reshaped:", indices.shape)
		result = np.hstack((temp, indices, code))
		print("Shape after stack: ", result.shape)
		#result = np.reshape(result, (result.shape[1], result.shape[0]))
		#print("Shape after reshape: ", result.shape)
		write_histogram(name+str(i), outdir, result)
		
		
		
def _main(args):
	
	# retorna descritores de cada level empilhados
	descriptors = read_all_desc(args.dir, args.level)
	
	codebooks = read_all_codebook(args.codedir, args.level)
	
	sum = sum_frequencies(descriptors, args.level)
	
	write_statistics(sum, args.level, args.outdir, args.name, codebooks)
	

	
if __name__ == '__main__':
	args = _get_Args()
	_main(args)