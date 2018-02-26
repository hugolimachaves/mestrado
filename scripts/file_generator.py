import random
import argparse
import os

encode = "utf-8"

def _get_Args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--fc7", help="Creates a fc7 file", action='store_true')
	parser.add_argument("-l", "--lsh", help="Creates a binary file with n bits", type=int)
	parser.add_argument("-k", "--keyframes", help="Creates a key frames file", action='store_true')
	parser.add_argument("-c", "--cnnflows", help="Creates a CNN flows file with n levels in the pyramid", type=int, choices=[1, 2, 3, 4])
	parser.add_argument("-p", "--pca", help="Creates a CNN flows reduced by PCA file with n levels in the pyramid", type=int, choices=[1, 2, 3, 4])
	parser.add_argument("-b", "--codebooks", help="Creates a codebooks file", action='store_true')
	parser.add_argument("-d", "--descriptors", help="Creates a video descriptors file with n levels in the pyramid", type=int, choices=[1, 2, 3, 4])
	parser.add_argument("-n", "--name", help="Output file name")
	parser.add_argument("-o", "--outdir", help="Output directory")
	parser.add_argument("-s", "--seed", help="Seed of the random function", type=int)
	parser.add_argument("-q", "--quant", help="Number of files that will be generate", type=int)
	return parser.parse_args()
	
	
def create_fc7_file(name, outdir):
	minimum = 1
	maximum = 1000
	frames = 100
	
	output_file = os.path.join(outdir, name) + ".fc7"
	output = open(output_file, "w", encoding=encode)
	
	for i in range(frames):
		for j in range(4096):
			feature = random.uniform(minimum, maximum)
			output.write(str(feature)+" ")
		output.write("\n")
		
	output.close()
	
def create_lsh_file(name, outdir, num_bits=16):
	minimum = 1
	maximum = 40
	frames = 100
	count = 0
	
	output_file = os.path.join(outdir, name) + ".lsh"
	output = open(output_file, "w", encoding=encode)
	
	# Pick a random integer between 0 and 2 power num_bits
	value = random.randint(0, 2**num_bits)	
	# Converts variable value to binary with num_bits size and zero paded at left. More info: https://stackoverflow.com/questions/10411085/converting-integer-to-binary-in-python
	bin_value = list(('{0:0'+str(num_bits)+'b}').format(value))
	
	
	while count < frames:
	
		number = random.randint(minimum, min(maximum, frames - count))
		str_value = "".join(bin_value)
		
		for i in range(number):
			output.write(str_value+"\n")
			
		count = count + number
		
		# Flip one bit at index position
		index = random.randint(0, num_bits - 1)	
		bin_value[index] = str(abs (int( bin_value[index]) - 1 ))
	
		
	output.close()

	
def create_keyframes_file(name, outdir):
	# min and max size values of snippets
	minimum = 20
	maximum = 30
	frames = 100
	count = 0
	number = 0
	
	output_file = os.path.join(outdir, name) + ".bkf"
	output = open(output_file, "w", encoding=encode)
	
	number = random.randint(minimum, maximum)

	while (frames - count) >= 2*minimum:
		# Logic to let at least minimun frames to the last snippet
		temp = min(maximum, (frames - count - minimum))
		
		count = count + number
		output.write(str(count)+"\n")
		
		number = random.randint(minimum, temp)
		
	output.write(str(frames)+"\n")
	output.close()

def create_cnnflows_file(name, outdir, level=4):
	minimum = -999
	maximum = 999
	flows = 1
	snipets = 4;
	
	if level < 4:
		flows = 2**level - 1
	else:
		flows = 2**3 - 1 + 10
		
	output_file = os.path.join(outdir, name) + ".cnnf"
	output = open(output_file, "w", encoding=encode)
	
	for i in range(snipets):
		for j in range(flows):
			for k in range(4096):
				cnnf = random.uniform(minimum, maximum)
				output.write(str(cnnf)+" ")
			output.write("\n")
		output.write("\n")
		
	output.close()
	

def create_pca_file(name, outdir, level=4):
	minimum = 1
	maximum = 999
	flows = 1
	snipets = 4;
	
	if level < 4:
		flows = 2**level - 1
	else:
		flows = 2**3 - 1 + 10
		
	output_file = os.path.join(outdir, name) + ".pca"
	output = open(output_file, "w", encoding=encode)
	
	for i in range(snipets):
		for j in range(flows):
			for k in range(100):
				pca = random.uniform(minimum, maximum)
				output.write(str(pca)+" ")
			output.write("\n")
		output.write("\n")
		
	output.close()
	

def create_codebooks_file(name, outdir):
	minimum = 1
	maximum = 1000
	size = 4000
	
	output_file = os.path.join(outdir, name) + ".dic"
	output = open(output_file, "w", encoding=encode)
	
	for i in range(size):
		for j in range(100):
			coord = random.uniform(minimum, maximum)
			output.write(str(coord)+" ")
		output.write("\n")
		
	output.close()
	

def create_descriptors_file(name, outdir, level = 4):
	minimum = 0
	maximum = 5
	size = 4000
	
	output_file = os.path.join(outdir, name) + ".desc"
	output = open(output_file, "w", encoding=encode)
	
	for i in range(level):
		for j in range(size):
			freq = random.randint(minimum, maximum)
			output.write(str(freq)+" ")
		output.write("\n")
		
	output.close()
	return
	
	
def _main(args):

	no_args_flag = True
	
	# If ommited, args.seed gets None value and the seed uses randomness sources provided by the OS
	random.seed(args.seed)
	
	if not args.outdir:
		args.outdir = ''
	else:
		if not os.path.isdir(args.outdir):
			os.makedirs(args.outdir)
	
	if not args.name:
		name = 'test'
		args.name = name
	else:
		name = args.name
		
	if not args.quant:
		args.quant = 1
		
	for i in range(args.quant):
		
		if args.fc7:
			create_fc7_file(name, args.outdir)
			no_args_flag = False
			
		if args.lsh:
			create_lsh_file(name, args.outdir, args.lsh)
			no_args_flag = False
			
		if args.keyframes:
			create_keyframes_file(name, args.outdir)
			no_args_flag = False
		
		if args.cnnflows:
			create_cnnflows_file(name, args.outdir, args.cnnflows)
			no_args_flag = False
		
		if args.pca:
			create_pca_file(name, args.outdir, args.pca)
			no_args_flag = False
		
		if args.codebooks:
			create_codebooks_file(name, args.outdir)
			no_args_flag = False
		
		if args.descriptors:
			create_descriptors_file(name, args.outdir, args.descriptors)
			no_args_flag = False
		
		if no_args_flag:
			create_fc7_file(name, args.outdir)
			create_lsh_file(name , args.outdir)
			create_keyframes_file(name, args.outdir)
			create_cnnflows_file(name, args.outdir)
			create_pca_file(name, args.outdir)
			create_codebooks_file(name, args.outdir)
			create_descriptors_file(name, args.outdir)
		
		name = args.name + str(i)

		
if __name__ == '__main__':
	# parse arguments
	args = _get_Args()
	_main(args)
	
	


		




