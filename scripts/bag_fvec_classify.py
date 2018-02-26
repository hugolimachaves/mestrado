import argparse
import numpy as np
import os
import time
import subprocess as sp
import bag_of_features as BoF
import merge_files_to_fvec as fvec
from DescrFVEC import saveDescriptor

encode = "utf-8"

def _get_Args():
	parser = argparse.ArgumentParser()
	parser.add_argument("indir", help="Directory with all dataset reduced by pca")
	parser.add_argument("outdir", help="Directory where any intermediate files will be created")
	parser.add_argument("level", type=int, help="Number of cnnflow pyramid levels")
	parser.add_argument("size", type=int, help="Size of codebook")
	parser.add_argument("name", help="Name of fvec final file")
	parser.add_argument("-i", "--iterations", type=int, help="Number of iterations of kmeans algorithm", default=10)
	parser.add_argument("-c", "--concatenate", help="Concatenation option", action = 'store_true')
	parser.add_argument("-c2", "--concatenate2", help="Concatenation option with 17 histograms", action = 'store_true')
	parser.add_argument("--individual_levels","-il", type=int, help="If present the fvec will be build with only the level passed in level argument", default = '')
	return parser.parse_args()
	
def concat_histograms(all_histograms, name, indir, outdir, individual):
	
	if not os.path.isdir(outdir) and outdir != '':
		os.makedirs(outdir)
	
	#name of file with all fvecs
	fvec_filename = os.path.join(outdir, name + '.fvec')
	#name of file with all labels sorted according with fvec
	labels_filename = os.path.join(outdir, 'label_' + name +'.txt')
	
	print(time.ctime(), ' Concatenando histogramas no arquivo ',fvec_filename,'...')
	print(time.ctime(), ' Salvando labels no arquivo ',labels_filename,'...')
	
	for set in ['training', 'validation', 'test']: # iteracao nos conjuntos de treinamento, validacao e teste
		
		print(time.ctime(), ' Concatenando conjunto de ',set,'...')
		
		if not individual:
			for i in range(len(all_histograms[set][0])): # itero sobre todos histogramas do nivel
			
				descriptors = np.array([])
				for j in range(len(all_histograms[set])): # itero em cada nivel
					# concatenando
					descriptors = np.concatenate((descriptors, all_histograms[set][j][i]['histogram']))
					
				# convertendo para float32
				descriptors = descriptors.astype('float32')
				# removo a extensao
				file_name = all_histograms[set][j][i]['name']
				# escrevo no fvec 
				saveDescriptor(fvec_filename, descriptors)
				fvec.save_label(file_name, labels_filename)
		else:
			# concatenando
			descriptors = np.concatenate((descriptors, all_histograms[set][individual][i]['histogram']))
			
			# convertendo para float32
			descriptors = descriptors.astype('float32')
			# removo a extensao
			file_name = all_histograms[set][individual][i]['name']
			# escrevo no fvec 
			saveDescriptor(fvec_filename, descriptors)
			save_label(file_name, labels_filename)
		print(time.ctime(), ' OK')
	
def _main(args):	
	
	if not os.path.isdir(args.outdir):
		os.makedirs(args.outdir)
	
	# Criacao de um arquivo de log, depois tenho que alterar isso para fazer um log de verdade de toda execucao
	
	with open(os.path.join(args.outdir, 'log.txt'), "w", encoding=encode) as file:
		file.write('Arquivos dessa pasta foram criados com a execucao do bag of words com os seguintes parametros: \n'+str(vars(args)))
	
	
	
	################################### Bag of Features ##################################
	
	pca_sets = {'training':[],'validation':[],'test':[]}
	
	# Faco o bag para cada nivel do conjunto de treino e salvo os codigos e os arquivos pca para criar os histogramas depois
	codebooks, pca_sets['training'] = BoF.BoF_for_each_level(args.level, args.indir, args.size, args.iterations, args.concatenate)
	
	# Gravo os codebooks em disco
	BoF.save_codebook_for_each_level(codebooks, 'codebook', os.path.join(args.outdir, 'histograms'), args.iterations)
	
	# Leio os arquivos do pca dos outros conjuntos para criar os histogramas tambem
	pca_sets['validation'], pca_sets['test'] = BoF.read_pca_validation_and_test(args.level, args.indir)
	
	# Crio  e gravo os histogramas de todos os conjuntos
	all_histograms = BoF.create_and_write_histogram_for_each_set(codebooks, pca_sets, args.indir, os.path.join(args.outdir, 'histograms'), args.concatenate, args.concatenate2)
	
	
	
	################# Merge of histograms to create fvec and label files ##################
	
	fvec_outdir = os.path.join(args.outdir, 'fvec')
	
	concat_histograms(all_histograms, args.name, os.path.join(args.outdir, 'histograms'), os.path.join(args.outdir, 'fvec'), args.individual_levels)

	
	
	################################ Classfication run #####################################
	
	with open('threads.txt', "r", encoding=encode) as file:
		n_threads = file.read()
	
	print(time.ctime(), ' Rodando classificador...')
	
	sp.run(['python2', 'classify_split.py', os.path.join(fvec_outdir, args.name+'.fvec'), os.path.join(fvec_outdir, 'label_'+args.name+'.txt'), os.path.join(fvec_outdir, 'output_report.txt'),'-clf','svm','-n_threads',n_threads])
	
	print(time.ctime(), ' Execucao finalizada!!')
	
if __name__ == '__main__':
	# parse arguments
	args = _get_Args()
	_main(args)
