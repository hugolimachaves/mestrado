import argparse
import math
import pandas as pd
import os
from binary_features import read_fc7_file
import numpy as np
from algeb import normalizar
from multiprocessing import Pool
import time


encode = "utf-8"
threads_file = 'threads.txt'
sets = ['training', 'validation', 'test']

def _get_Args():
	parser = argparse.ArgumentParser()
	group = parser.add_mutually_exclusive_group()
	group.add_argument("-k", "--key_dir", help="Input directory with key frames files divided in training, validation and test sets")
	parser.add_argument("dir", help="Input directory with fc7 files divided in training, validation and test sets")
	parser.add_argument("level", type=int, help="Number of pyramid levels", choices=[1, 2, 3, 4])
	parser.add_argument("-o", "--outdir", help="Output directory")
	group.add_argument("-r", "--regular", help="Regular size of snippets", type=int)
	parser.add_argument("-s", "--stride", help="Stride for snippets with regular sizes", type=int)
	parser.add_argument("-n", "--normalize", help="L2 normalization of input data", action='store_true')
	parser.add_argument("-i", "--individual", help="Generation of only one level", action='store_true')
	#parser.add_argument("-s", "--seed", help="Seed of the random function", type=int)
	return parser.parse_args()
	
def get_threads():

	try:
		with open(threads_file, "r", encoding=encode) as input:
			# Reading number of threads
			threads = int(input.read())
			return threads
	except Exception as e:
		print("A problem ocurred trying to read the number of threads: ", e)
		

'''
@first_frame posicao do primeiro frame desse snippet dentro de @frames
@last_frame posicao do primeiro frame desse snippet dentro de @frames
@frames contem fc7's de cada frame 
@level quantidade de niveis da piramide
'''
def generate_cnn_flows_of_snippet(first_frame, last_frame, frames, level, individual):
	
	# criando a lista de cnnflows vazia para cada nível
	cnn_flows = [[] for i in range(level)]
	
	if not individual or (individual and level == 1):
		# first_frame e last_frame são 1-indexed, entao subtraio uma unidade
		first_frame_features = frames[first_frame - 1]
		last_frame_features = frames[last_frame - 1]
		
		# geracao das cnnflows a partir da subtracao das fc7's do primeiro e ultimo frame do snippet
		# revendo o codigo agora, seria melhor usar np.array e fazer a subtracao direto
		cnn_flow = [float(b) - float(a) for a, b in zip(first_frame_features, last_frame_features)]
		# essa eh a cnnflow do nivel 0, que sempre existira
		cnn_flows[0].append(cnn_flow)
	
	if (not individual and level > 1) or (individual and level == 2):
		# quebro o snippet em 2
		sub_snippet_size = math.floor((last_frame - first_frame + 1)/2)
		# eh necessario que cada pedaco do snippet tenha no minimo 2 frames para realizar a subtracao
		if sub_snippet_size < 2:
			return cnn_flows
		
		# pego o resto da divisao acima para distribuir igualmente entre os primeiros sub-snippets
		rest = (last_frame - first_frame + 1) - (2*sub_snippet_size)
		# x agora representa o primeiro frame de um sub-snippet e y o ultimo frame
		x = first_frame
		for i in range(2):
			y = x + sub_snippet_size - 1
			
			# distribuicao regular do resto da divisao entre os sub-snippets
			if(i < rest):
				y = y + 1
			
			
			first_frame_features = frames[x - 1]
			last_frame_features = frames[y - 1]
		
			cnn_flow = [float(b) - float(a) for a, b in zip(first_frame_features, last_frame_features)]
			cnn_flows[1].append(cnn_flow)
			
			x = y + 1
		
	if (not individual and level > 2) or (individual and level == 3):
		sub_snippet_size = math.floor((last_frame - first_frame + 1)/4)
		if sub_snippet_size < 2:
			return cnn_flows
		rest = (last_frame - first_frame + 1) - (4*sub_snippet_size)
		x = first_frame
		for i in range(4):
			y = x + sub_snippet_size - 1
			
			if(i < rest):
				y = y + 1
			
			
			first_frame_features = frames[x - 1]
			last_frame_features = frames[y - 1]
		
			cnn_flow = [float(b) - float(a) for a, b in zip(first_frame_features, last_frame_features)]
			cnn_flows[2].append(cnn_flow)
			
			x = y + 1
			
	if (not individual and level > 3) or (individual and level == 4):
		# ultimo nivel tem sempre 10 snippets...
		sub_snippet_size = math.floor((last_frame - first_frame + 1)/10)
		if sub_snippet_size < 2:
			return cnn_flows
		rest = (last_frame - first_frame + 1) - (10*sub_snippet_size)
		
		x = first_frame
		for i in range(10):
			y = x + sub_snippet_size - 1
			
			if(i < rest):
				y = y + 1
			
			first_frame_features = frames[x - 1]
			last_frame_features = frames[y - 1]
		
			cnn_flow = [float(b) - float(a) for a, b in zip(first_frame_features, last_frame_features)]
			cnn_flows[3].append(cnn_flow)
			
			x = y + 1
	
	
	return cnn_flows
	
def read_keyframes(name):

	key_frames = pd.read_csv(name, header=None)
	
	return key_frames.values

	
def create_pyramid(key_frames, frames, level, stride, regular, individual):

	# Se for os keyframes foram lidos de arquivo preciso fazer essa conversao
	if type(key_frames[0]) == type([]):
		key_frames = [i for item in key_frames for i in item]
	
	first_frame = 1
	# criando lista de listas vazia
	cnn_flow_snippets = [[] for i in range(level)]
	# criando cnnflow de cada snippet de um arquivo
	for i in range(len(key_frames)):
		
		last_frame = key_frames[i]
		# gera cnn flows de um snippet formado pelo intervalo fechado entre @first_frame e @last_frame
		cnn_flows = generate_cnn_flows_of_snippet(first_frame, last_frame, frames, level, individual)
		if not individual:
			for k in range(level):
				# guardando as cnnflows separadamente para futuramente salvar em pastas separadas
				cnn_flow_snippets[k].append(cnn_flows[k])
		else:
			cnn_flow_snippets[level - 1].append(cnn_flows[level - 1])
		# Atualizacao do @first_frame
		if not stride:
			first_frame = last_frame + 1
		elif regular:
			first_frame = first_frame + stride

	return cnn_flow_snippets
	
def write_cnn_flow(name, outdir, cnn_flow_snippets, level, individual):

	# desconsiderando o path
	name = os.path.basename(name)
	# removendo extensao
	name = os.path.splitext(name)[0]
	
	try:	
		if not individual:
			for k in range(level):
				# pasta para este nivel
				out_dir = os.path.join(outdir, 'nivel'+str(k))
				# If path doesn't exists, make it
				if not os.path.isdir(out_dir):
					# na execucao em paralela pode ser que eu tente criar um diretorio existente, por isso exist_ok = True
					os.makedirs(out_dir, exist_ok = True)
				# agregando o diretorio passado + nome + n_nivel + extensao de cnnflow
				out_file = os.path.join(out_dir, name) + str(k) + ".cnnf"
			
				# With automatically closes output
				with open(out_file, "w", encoding=encode) as output:
					# Joining cnn flows elements with space and then joining cnn flows with \n and finally joining snippets with \n\n
					output.write("\n\n".join(["\n".join([" ".join(list(map(str, j))) for j in i]) for i in cnn_flow_snippets[k]]))
		else:
			# estiver criando apenas um nivel individualmente
			# pasta para este nivel
			out_dir = os.path.join(outdir, 'nivel'+str(level - 1))
			# If path doesn't exists, make it
			if not os.path.isdir(out_dir):
				# na execucao em paralela pode ser que eu tente criar um diretorio existente, por isso exist_ok = True
				os.makedirs(out_dir, exist_ok = True)
			# agregando o diretorio passado + nome + n_nivel + extensao de cnnflow
			out_file = os.path.join(out_dir, name) + str(level - 1) + ".cnnf"
		
			# With automatically closes output
			with open(out_file, "w", encoding=encode) as output:
				# Joining cnn flows elements with space and then joining cnn flows with \n and finally joining snippets with \n\n
				# indice zero pois eh o caso de estar criando apenas um nivel individualmente
				output.write("\n\n".join(["\n".join([" ".join(list(map(str, j))) for j in i]) for i in cnn_flow_snippets[level - 1]]))
			
		return 0
		
	except Exception as e:
		print("Some error occurred while writing cnnflow pyramid into file: ", e)
		return 1



def cnnflow_pyramid(fc7_name, indir, key_dir, key_name, level, normalize, regular, stride, outdir, individual):
	'''
	Criacao das cnnflows de um arquivo fc7 passado @fc7_name presente no diretorio @indir
	Realiza a divisao dos snippets de acordo com o arquio @key_name presente no diretorio @key_dir (opcional)
	int @level indica a quantidade de niveis que a piramide de cada snippet tera
	bool @normalize indica se as entradas serao normalizados com a norma l2 ou nao
	int @regular indica o tamanho regular que os snippet terao (opcional)
	int @stride representa os salto de um snippet para o outro (opcional)
	dir @outdir indica o diretorio de saida no qual os arquivos criados serao gravados]
	@regular e @stride sao mutuamente inclusivos
	@key_name e @key_dir sao mutuamente inclusivos
	(@regular, @stride) e (@key_name, @key_dir) sao mutualmente exclusivos
	'''		

	frames = read_fc7_file(os.path.join(indir, fc7_name))
		
	if normalize:
		frames = [(normalizar(frame)).tolist() for frame in frames]
		frames = np.array(frames)
	
	if regular:
		size = frames.shape[0]
		# se for snippet de tamanho regular eu gero os supostos keyframes de acordo com a stride, quando passada
		if not stride:
			key_frames = np.arange(0, size, regular).tolist()
			# nao preciso do primeiro key frame pois para mim sempre sera o primeiro frame do video
			del key_frames[0]
		else:
			key_frames = np.arange(regular, size, stride).tolist()
		
		key_frames.append(size)
	else:
		key_frames = read_keyframes(os.path.join(key_dir, key_name)).tolist()
		
	# criacao da bendita piramide para um arquivo
	cnn_flow_snippets = create_pyramid(key_frames, frames.tolist(), level, stride, regular, individual)
	
	if not outdir:
		outdir = ''
		
	write_cnn_flow(fc7_name, outdir, cnn_flow_snippets, level, individual)
	
def _main(args):
	
	args.outdir = os.path.join(args.outdir, 'cnnflows')
	
	if not os.path.isdir(args.outdir):
		os.makedirs(args.outdir)
	
	# Criacao de um arquivo de log, depois tenho que alterar isso para fazer um log de verdade de toda execucao
	
	with open(os.path.join(args.outdir, 'log.txt'), "w", encoding=encode) as file:
		file.write('Arquivos dessa pasta foram criados com a execucao da piramide com os seguintes parametros: \n'+str(vars(args)))
	
	
	# para cada conjunto
	for set in sets:
	
		print(time.ctime(), " Creating cnnflow of ",set," set")
	
		dir = os.path.join(args.dir, set)
		# Listando arquivos fc7
		fc7_files = [file for file in os.listdir(dir) if os.path.splitext(file)[1] == '.fc7']
		
		# Pegando nome do diretorio de saida para futura escrita de arquivos
		out_dir = os.path.join(args.outdir, set)
		
		# Lista de contera os numeros dos frames que sao key frames
		key_dir = []
		# Lista de contera os arquivos que contem os key frames de cada arquivo fc7
		key_files = []
		
		# Se foi passado um diretorio com os key frames
		if args.key_dir:
			key_dir = os.path.join(args.key_dir, set)
			# Listo apenas os arquivos bkf (binary key frame) do diretorio passado
			key_files = [file for file in os.listdir(key_dir) if os.path.splitext(file)[1] == '.bkf']
		else:
			# Preencho com None so para ter uma lista a ser passada na execucao em paralelo
			key_files = [None for i in fc7_files]
			
		# replicacao dos demais parametros para realizar execucao em paralelo
		indir_list = [dir for i in fc7_files]
		key_dir_list = [key_dir for i in fc7_files]
		level_list = [args.level for i in fc7_files]
		normalize_list = [args.normalize for i in fc7_files]
		regular_list = [args.regular for i in fc7_files]
		stride_list = [args.stride for i in fc7_files]
		out_dir_list = [out_dir for i in fc7_files]
		individual_list = [args.individual for i in fc7_files]
		
		# execucao em paralelo com um numero de thread disponiveis
		with Pool(get_threads()) as p:
			p.starmap(cnnflow_pyramid, zip(fc7_files, indir_list, key_dir_list, key_files, level_list, normalize_list, regular_list, stride_list, out_dir_list, individual_list), chunksize=10)
	
	
if __name__ == '__main__':
	# parse arguments
	args = _get_Args()
	_main(args)
