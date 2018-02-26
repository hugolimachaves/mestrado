'''
Rotina p/ verificar se há valores anômalos (+- Inf e NaN) nos arquivos
*Fornecer o caminho para os arquivos que serão analisados (não inclui o nome o próprio arquivo)
*Fornecer a extensão dos arquivos que serão analisados
+ Se houver alguma anomalia, esta será printada na tela e um arquivo txt será salvo no caminho analisado
INPUT:
1)Caminho (global) onde estao os arquivos a serem analisados
2) Extesão dos arquivos
'''

import os
import algeb as alg
import argparse

#teste
#caminho_fornecido = 'C:\\Users\\HugoAparecido\\AppData\\Local\\Programs\\Python\\Python36\\implementacao_v5\\cnn_action_pyramid-master\\training\\'
#extensao = '.pcab'

def _get_Args():
	parser = argparse.ArgumentParser()
	parser.add_argument("path", help="Caminho global o da pasta onde estao os arquivos a serem analisados")
	parser.add_argument("ext", help="tipo da extensao a ser analisada, ex.: .txt")
	return parser.parse_args()

def get_arquivos_validos(caminho_fornecido,extensao):
	arquivos_candidatos = os.listdir(caminho_fornecido)
	arquivos_validos = []
	for i in arquivos_candidatos:
		if i.endswith(extensao):
			arquivos_validos.append(i)
	return arquivos_validos

def get_path_2_arquivos_validos(caminho_fornecido,arquivos_validos):
	caminho_arq_validos = []
	for i in range(len(arquivos_validos)):
		caminho_arq_validos.append(os.path.join(caminho_fornecido,arquivos_validos[i]))
	return caminho_arq_validos

def detect_anomalia(caminho_arq_validos,caminho_fornecido):
	lista_de_anomalias = []
	for i in caminho_arq_validos:
		file = open(i,'r')
		linhas = file.readlines()
		file.close()
		cont_linha = 0
		for j in linhas:
			linha_split = j.split(' ')
			linha_float = list(map(float,linha_split))
			if alg.detectNaN(linha_float):
				log = '\n***NOT A NUMBER*** detectado na linha ' + str(cont_linha) + ' em:'
				lista_de_anomalias.append(log)
				lista_de_anomalias.append(i)
				print(log)
				print(i)
			if alg.detectInf(linha_float):
				log = '\n***INFINITO*** detectado na linha ' + str(cont_linha) + ' em:'
				lista_de_anomalias.append(log)
				lista_de_anomalias.append(i)
				print(log)
				print(i)
			cont_linha += 1
	if lista_de_anomalias:
		file = open( str(caminho_fornecido) + "log_anomalias.txt","w")
		for i in range(len(lista_de_anomalias)):
			file.write(lista_de_anomalias[i])
		file.close()

def _main(args):
	caminho_fornecido = args.path
	extensao = args.ext
	arquivos_validos = get_arquivos_validos(caminho_fornecido,extensao)
	caminho_de_arquivos_validos  = get_path_2_arquivos_validos(caminho_fornecido , arquivos_validos)
	detect_anomalia(caminho_de_arquivos_validos,caminho_fornecido)
	
if '__main__' == __name__:
	args = _get_Args()
	_main(args)
