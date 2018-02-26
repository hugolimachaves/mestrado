import os 
file = open('/home/schrodinger/sdb/CNNFeatures/scripts/parametros/checklist.txt','r')
linhas = file.readlines()
linhas =  [ linha.split(" ") for linha in linhas]
file.close()

for i in range(len(linhas)):
	linhas[i][1] = linhas[i][1].split('\n')[0]
	comando = 'python ' + 'verify.py ' + linhas[i][0] + ' ' + linhas[i][1]
	print(comando)
	os.system(comando)
