import os
import argparse
import multiprocessing as mp

def _get_Args():
	parser = argparse.ArgumentParser()
	#parser.add_argument("caminho_script", help ="Caminho absoluto para o script")
	parser.add_argument("nThreads", help = "numero de threads que deseja" , type = int)
	parser.add_argument("script", help="Nome do script a ser executado")
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

def get_comando(script, nDimensoesFinais, caminho_comum_in, caminho_comum_out, refIn, refOut, nivelIn, nivelOut ):
	comando = []
	for i in range(3): # percorrer três grupos
		for j in range(1): # percorrer 4 níveis
			caminho_das_bases = os.path.join(caminho_comum_in,refIn[0])
			caminho_das_bases = os.path.join(caminho_das_bases,nivelIn[j])
			caminho_dos_arquivos = os.path.join(caminho_comum_in,refIn[i])
			caminho_dos_arquivos = os.path.join(caminho_dos_arquivos,nivelIn[j])
			caminho_out = os.path.join(caminho_comum_out,refOut[i])
			caminho_out = os.path.join(caminho_out,nivelOut[j])
			apendice = caminho_dos_arquivos + ' ' + nDimensoesFinais + ' ' +  caminho_das_bases + ' ' + caminho_out
			comando.append('python ' + script + ' ' + apendice)
	#indicador = ('Executing command: ' + comando).center(300,'#')
	return comando

def _main(args):
	#parâmetros
	nThreads = args.nThreads
	script = args.script
	nDimensoesFinais = args.finalFeatures
	caminho_comum_in = args.caminho_comum_in
	caminho_comum_out = args.caminho_comum_out
	refIn = [] # nome da pasta( treino, teste e validacao)
	refIn.append(args.treinoIn)
	refIn.append(args.validacaoIn)
	refIn.append(args.testeIn)
	refOut = []
	refOut.append(args.treinoOut)
	refOut.append(args.validacaoOut)
	refOut.append(args.testeOut)
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
	#comandos
	comando = get_comando(script, nDimensoesFinais, caminho_comum_in, caminho_comum_out, refIn, refOut, nivelIn, nivelOut)
	paralelizar(comando,nThreads)

def paralelizar(comando,nThreads):
	pool = mp.Pool(processes=nThreads)
	pool.map(os.system, comando)
	#pool.close()
	#pool.join()

if __name__ == '__main__':
	args = _get_Args()
	_main(args)
