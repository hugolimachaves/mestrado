import argparse
import os
import shutil

encode = "utf-8"

train_set = {11, 12, 13, 14, 15, 16, 17, 18}
validation_set = {1, 4, 19, 20, 21, 23, 24, 25}	
test_set = {2, 3, 5, 6, 7, 8, 9, 10, 22}

def _get_Args():
	parser = argparse.ArgumentParser()
	parser.add_argument("indir", help="Input directory to split")
	parser.add_argument("outdir", help="Output directory")
	parser.add_argument("ext", help="Extension of files")
	return parser.parse_args()
	
def extract_type(label):

	type = ''
	# Extraio a posicao do numero da pessoa no label transformo em int e vejo em qual conjunto esta
	if(int(label[6:8]) in train_set):
		type = 'training'
	elif(int(label[6:8]) in validation_set):
		type = 'validation'
	elif(int(label[6:8]) in test_set):
		type = 'test'
		
	return type
	
def split_sets(dir, outdir, ext):
	
	if os.path.isdir(dir):		# se o diretorio existir
		for root, dirs, files in os.walk(dir):		#caminho pelo diretorio
			for file in files:
				if os.path.splitext(file)[1] == ext:		#verifico se eh da extensao desejada
				
					type= extract_type(file)			#extraio o tipo
				
					if "nivel0" in root:					#verifico em qual nivel esta
						src = os.path.join(root, file)		#pego o caminho de origem e o de destino
						dst = os.path.join(outdir, type, "nivel0")
						if not os.path.isdir(dst):				#crio o caminho de destino caso nao exista
							os.makedirs(dst)
						shutil.copy(src, dst)				#copio o arquivo
						
					elif "nivel1" in root:
						src = os.path.join(root, file)
						dst = os.path.join(outdir, type, "nivel1")
						if not os.path.isdir(dst):
							os.makedirs(dst)
						shutil.copy(src, dst)
						
					elif "nivel2" in root:
						src = os.path.join(root, file)
						dst = os.path.join(outdir, type, "nivel2")
						if not os.path.isdir(dst):
							os.makedirs(dst)
						shutil.copy(src, dst)
						
					elif "nivel3" in root:
						src = os.path.join(root, file)
						dst = os.path.join(outdir, type, "nivel3")
						if not os.path.isdir(dst):
							os.makedirs(dst)
						shutil.copy(src, dst)
					else:
						src = os.path.join(root, file)		#pego o caminho de origem e o de destino
						dst = os.path.join(outdir, type)
						if not os.path.isdir(dst):				#crio o caminho de destino caso nao exista
							os.makedirs(dst)
						shutil.copy(src, dst)				#copio o arquivo
		
	else:
		print("Diretorio invalido!!")
	
def _main(args):

	split_sets(args.indir, args.outdir, args.ext)
	
if __name__ == '__main__':
	# parse arguments
	args = _get_Args()
	_main(args)
	