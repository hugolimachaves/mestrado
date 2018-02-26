'''
File intented to parallelize PCA projection as long as you provide the eigenvectors that compound the orthonormal basis and the mean vector.
This file doesn't allow one to control the number of threads by prompt args
Warning: # of threads limited by the # of folders where the data is read from.
'''

import multiprocessing as mp
import argparse
import os

def _get_Args():
	parser = argparse.ArgumentParser()
	#parser.add_argument("caminho_script_mestre", help = "Caminho do script o qual vai chamar o script que faz o trabalho em si" )
	#parser.add_argument("script_mestre", help = "Nome do script o qual vai chamar o script que faz o trabalho em si")
	parser.add_argument("caminho_script", help ="Caminho absoluto para o script")
	#parser.add_argument("script", help="Nome do script a ser executado")
	parser.add_argument("finalFeatures", help="Numero final de features (dimensoes) após o PCA")
	parser.add_argument("caminho_comum_in", help="Tronco comum do caminho dos arquivos de entrada")
	parser.add_argument("treinoIn", help="nome da pasta de treino do arquivo de entrada")
	parser.add_argument("validacaoIn", help="nome da pasta de validacao do arquivo de entrada")
	parser.add_argument("testeIn", help="nome da pasta de teste do arquivo de entrada")
	parser.add_argument("caminho_comum_out", help="Tronco comum do caminho dos arquivos de saida")
	parser.add_argument("treinoOut", help="nome da pasta de treino do arquivo de saida")
	parser.add_argument("validacaoOut", help="nome da pasta de validacao do arquivo de saida")
	parser.add_argument("testeOut", help="nome da pasta de teste do arquivo de saida")
	parser.add_argument("inl0", help="nome da pasta de teste do arquivo de entrada no nivel 0")
	parser.add_argument("inl1", help="nome da pasta de teste do arquivo de entrada no nivel 1")
	parser.add_argument("inl2", help="nome da pasta de teste do arquivo de entrada no nivel 2")
	parser.add_argument("inl3", help="nome da pasta de teste do arquivo de entrada no nivel 3")
	parser.add_argument("outl0", help="nome da pasta de teste do arquivo de saída no nivel 0")
	parser.add_argument("outl1", help="nome da pasta de teste do arquivo de saída no nivel 1")
	parser.add_argument("outl2", help="nome da pasta de teste do arquivo de saída no nivel 2")
	parser.add_argument("outl3", help="nome da pasta de teste do arquivo de saída no nivel 3")
	return parser.parse_args()
	
def _main(args):
	#caminho_mestre = args.caminho_script_mestre
	#script_mestre = arg.script_mestre
	caminho_script = args.caminho_script
	#script_executado = args.script
	nDimensoesFinais = args.finalFeatures
	caminho_comum_in = args.caminho_comum_in
	caminho_comum_out = args.caminho_comum_out
	groupIn = [] # nome da pasta( treino, teste e validacao)
	groupIn.append(args.treinoIn)
	groupIn.append(args.validacaoIn)
	groupIn.append(args.testeIn)
	groupOut = []
	groupOut.append(args.treinoOut)
	groupOut.append(args.validacaoOut)
	groupOut.append(args.testeOut)
	nivelIn = []
	nivelIn.append(args.inl0)
	nivelIn.append(args.inl1)
	nivelIn.append(args.inl2)
	nivelIn.append(args.inl3)
	nivelOut = []
	nivelOut.append(args.outl0)
	nivelOut.append(args.outl1)
	nivelOut.append(args.outl2)
	nivelOut.append(args.outl3)
	argumentos = []
	for i in range(3): # percorrer três grupos
		for j in range(1): # percorrer 4 níveis
			#comando = 'python ' + script_mestre + ' ' + script_executado + ' ' + nDimensoesFinais + ' ' + caminho_comum_in
			comando = 'python ' + caminho_script + ' ' + nDimensoesFinais + ' ' + caminho_comum_in
			comando = comando + ' ' + groupIn[0] + ' ' + groupIn[1] + ' ' + groupIn[2]
			comando = comando + ' ' + caminho_comum_out + ' ' + groupOut[0] + ' ' + groupOut[1] + ' ' + groupOut[2]
			comando = comando + ' ' + nivelIn[0] + ' ' + nivelIn[1]  + ' ' + nivelIn[2] + ' ' + nivelIn[3]
			comando = comando + ' ' + nivelOut[0] + ' ' + nivelIn[1] + ' ' + nivelIn[2] + ' ' + nivelIn[3] 
			argumentos.append(comando)
	nThreads = len(argumentos)
	pool = mp.Pool(processes=nThreads)
	pool.map(os.system, argumentos)
	pool.close()
	pool.join()
 
	
if __name__ == "__main__":
	print('passou primeiro')
	args = _get_Args()
	_main(args)

	
