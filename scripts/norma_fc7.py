import numpy as np
import pandas as pd
import os


'''
###########################
norma_l2
input: vetor
output: vetor normalizado

###########################
norma_max_min
input: vetor, valor maximo de referencia, valor mínimo de referencia
output: vetor normalizado

###########################
norma_estatistica_global
input: vetor, media (apenas um escalar para todo o vetor), desvio (apenas um escalar para todo o vetor), numero de desvios para o range de normalização
output: vetor normalizado

###########################
norma_estatistica_local
input: vetor, vetor com a media para cada posição do vetor, vetor com o desvio para cada posição do vetor 
(obviamente, para esses 2 ultimos paramentros, precisa-se de um banco vetores para que se tenha sentido estatístico) , 
número de desvios para o range de normalizacao ( será o mesmo valor para todos os componentes ) 
output: vetor normalizado
'''

def norma_max_min(vetor, max, min):

	range = max - min
	vetor_normalizado = (vetor - min)/range
	return vetor_normalizado
	
def norma_estatistica_global(vetor, media, vetor_desvio, nDesvios):
	range = 2*(nDesvios*(vetor_desvio))
	vetor_normalizado = vetor - media
	vetor_normalizado = vetor_normalizado/range
	return vetor_normalizado

def norma_estatistica_local(vetor, vetor_media, vetor_desvio, nDesvios):
	range = 2*(nDesvios*(vetor_desvio))
	vetor_normalizado = vetor - vetor_media
	vetor_normalizado = vetor_normalizado/range
	return vetor_normalizado

def norma_l2(vetor):
	vetor = np.array(vetor)
	norma=np.linalg.norm(vetor, 2)
	vetor_normalizado =  vetor/norma
	#print(norma)
	return vetor_normalizado


# ########################################################################
#O que está abaixo é exclusivamente para fins de teste do código.
# ########################################################################
'''
rootdir = ["C:\\Users\\HugoAparecido\\Desktop"]

df_all = pd.DataFrame()
for rootfolder in rootdir:
	files = [file for file in os.listdir(rootfolder) if os.path.splitext(file)[1] == '.fc7']
	for element in files:
		folder = os.path.join(rootfolder, element)
		df = pd.read_csv(folder, header=None)
		df = df[1:]
		df = df[:-1]
		df_all = df_all.append(df, ignore_index=True)

#mean = np.mean(df_all.as_matrix().astype(np.float))
#stdev = np.std(df_all.as_matrix().astype(np.float))
#print(mean)
#print(stdev)

matriz = (df_all.as_matrix().astype(np.float))
vetor = matriz[0][:]
vetor2 = matriz[1][:]
vetor3 = matriz[2][:]
vetor_media = np.mean(matriz, axis=0)
vetor_desvio = np.std(matriz, axis=0)
max = np.max(matriz)
min = np.min(matriz)
media = -2.37
desvio_padrao =  2.63
nDesvios = 3
'''

'''
print("vetor media")
print(vetor_media)
print("vetor desvios")
'''
#print(vetor_desvio)


'''
print("teste de funções".center(30,'#'))
print("vetor de teste")
print(vetor)
print(norma_l2(vetor))

print('teste da função: max_min'.center(30,'*'))
print(norma_max_min(vetor,max,min))

print('teste da função: estatistica global'.center(30,'*'))
print(norma_estatistica_global(vetor, media, desvio_padrao, nDesvios))
'''
#print('teste da função: estatistica local'.center(30,'*'))
#print(norma_estatistica_local(vetor, vetor_media, vetor_desvio, nDesvios))
#print(norma_estatistica_local(vetor2, vetor_media, vetor_desvio, nDesvios))
#rint(norma_estatistica_local(vetor3, vetor_media, vetor_desvio, nDesvios))





