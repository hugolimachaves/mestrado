from lshash import lshash
import argparse
import pandas as pd
import numpy as np
import os

encode = "utf-8"

def _get_Args():
	parser = argparse.ArgumentParser()
	parser.add_argument("name", help="Input file name")
	parser.add_argument("bits", type=int, help="Number of bits of codification")
	parser.add_argument("-o", "--outdir", help="Output directory")
	return parser.parse_args()
	
def read_fc7_file(name):

	
	fc7 = pd.read_csv(name, header=None)
	return fc7.values
	
def codify_frames(frames, num_bits):
	
	num_features = frames.shape[1]
	# Initializing hash
	lsh = lshash.LSHash(num_bits, num_features)
	# Getting plane of first and unique hash table
	plane = lsh.uniform_planes[0]
	bin_frames = []
	
	frames = frames - np.mean(frames, 0)
	
	for i in range(len(plane)):
		plane[i][-1] *= 10
			
	
	for i in frames:
		# Extracting features as float list
		features = i.tolist()
		bin_frames.append(lsh._hash(plane, features))
	
	return bin_frames
	
def write_binary_frames(name, outdir, bin_frames):

	name = os.path.split(name)[1]
	# Removing file extension
	name = ".".join(name.split('.')[:-1])
	bin_dataframe = pd.DataFrame(bin_frames)
	
	try:
		# If path doesn't exists, make it
		if not os.path.isdir(outdir) and outdir != '':
			os.makedirs(outdir)
				
		out_file = os.path.join(outdir, name) + ".lsh"
		
		bin_dataframe.to_csv(out_file, header=None, index=False)
			
		return 0
		
	except Exception as e:
		print("Some error occurred while writing binary frames into file: ", e)
		return 1
	
def _main(args):
	
	frames = read_fc7_file(args.name)
	binary_frames = codify_frames(frames, args.bits)
	if not args.outdir:
		args.outdir = ''
	write_binary_frames(args.name, args.outdir, binary_frames)
	
if __name__ == '__main__':
	# parse arguments
	args = _get_Args()
	_main(args)


	
