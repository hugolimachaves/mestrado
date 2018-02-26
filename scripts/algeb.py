import numpy as np
import math

'''
normalizar: recebe uma lista como parâmetro de entrada e retorna essa mesma lista normalizada pela l2. 
Entrada: list
Saída: np.array
'''
def normalizar(lst):
	lista = list(map(float, lst))
	#print(lista)
	lista = np.array(lista)
	#lista = lista.reshape(1,lista.shape[0])
	modulo = np.power(np.sum(np.power(lista,2)),0.5)
	lista = lista/modulo
	return lista
	
def detectNaN(lst):
	for i in lst:
		k = float(i)
		if math.isnan(k):
			return True
	return  False
	
def detectInf(lst):
	for i in lst:
		k = float(i)
		if (k == float('inf')) or (k == -float('inf'))  :
			return True
	return  False
		
