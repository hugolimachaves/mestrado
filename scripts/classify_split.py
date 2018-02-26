import numpy as np
import argparse
import sys
import os
import time
sys.path.insert(0, 'common/')
from DescrFVEC import getDescriptorsSplit
from classifiers import classifierGS
from classifiers import reportGS
from sklearn.metrics.pairwise import chi2_kernel

random_state = 13 #seed for random methods
power_norm = False #Enable power normalization
power_exp = 0.2 #Power normalization exponent
L2 = False #Enable L2 norm after a power normalization
std = False #Enable data standardization (mean removal and variance scaling)
clf_model = "rf" #'rf' = random forest or 'svm' = svm
n_threads = -1 #Number of threads

#Get arguments
def getArgs():
	parser = argparse.ArgumentParser(description='FVEC classifier.')
	parser.add_argument("input_fvec", action='store', type=str,
                        help="Descriptors file (.fvec).", default='list')
	parser.add_argument("input_set_labels", action='store', type=str,
                        help="Labels and splits file.", default='list')
	parser.add_argument("output_report", action='store', type=str,
                        help="Report file.", default='list')
	parser.add_argument('-power_norm', type=float, help="Power normalization.")
	parser.add_argument('-l2', action='store_true', help="Enable L2 normalization.")
	parser.add_argument('-std', action='store_true', help="Enable data standardization.")
	parser.add_argument('-clf', type=str, help="Classifier. rf = random forest, svm = SVM.")
	parser.add_argument('-n_threads', type=int, help="Number of threads.")
	parser.add_argument('--seed', '-s', type=int, help= "Randomic seed.")
	#parser.add_argument('-kernel', help="Kernel name")

	global power_norm, power_exp, L2, std, clf_model, n_threads
	if (parser.parse_args().power_norm):
		power_norm = True
		power_exp = parser.parse_args().power_norm
	L2 = parser.parse_args().l2
	std = parser.parse_args().std
	if(parser.parse_args().clf):
		clf_model = parser.parse_args().clf
	if (parser.parse_args().n_threads and parser.parse_args().n_threads >= 1):
		n_threads = parser.parse_args().n_threads

	with open(parser.parse_args().output_report, "w") as f:
		f.write('INFO:\n')
		f.write('\tFile: %s.\n' %parser.parse_args().input_fvec)		
		f.write('\tLabels: %s.\n' %parser.parse_args().input_set_labels)
		f.write('\tProtocol: GridSearch and fixed train-validation-test split.\n')		
		f.write('\tClassifier: %s \n' % clf_model)
		if power_norm:
			f.write('\tPower normalization. Power exp = %.2f\n' %power_exp)
		if L2:
			f.write('\tNormalization L2.\n')
		if std:
			f.write('\tData standardization\n')
		f.write('\tNumber of threads: %d\n' % n_threads)
		f.write('\n-----------------------------------------------------------\n')

	return parser.parse_args()

def classify(descriptors, labels, sample_indexes, report_file):
	descr_np = np.array(descriptors)
	labels_np = np.array(labels)
	descriptors, labels = [],[]

	with open(report_file, "a") as f:
		f.write('\n\tDescriptor size: %d\n' %len(descr_np[0]))	
		f.write('\tNumber of descriptors: %d\n' %len(descr_np))	
		f.write('\n-----------------------------------------------------------\n')
	
	X_train, y_train = descr_np[sample_indexes[0]], labels_np[sample_indexes[0]]
	X_valid, y_valid = descr_np[sample_indexes[1]], labels_np[sample_indexes[1]]
	X_test, y_test = descr_np[sample_indexes[2]], labels_np[sample_indexes[2]]
	descr_np, labels_np = [],[]
	

	X_2fit, y_2fit = np.concatenate((X_train,X_valid)), np.concatenate((y_train,y_valid))
	################################### Mudancas para usar chi2 como kernel
	
	#X_2fit = chi2_kernel(X_2fit, gamma=.5)
	#X_test = chi2_kernel(X_test, gamma=.5)
	
	###################################
	train_indexes2 = range(0,len(X_train))
	valid_indexes2 = range(len(X_train),len(X_2fit))
	custom_split = zip([train_indexes2],[valid_indexes2])
	X_train, y_train, X_valid, y_valid = [],[],[],[]
	
	print("Passing random state:", str(random_state))
	clf = classifierGS(clf_model, custom_split, random_state, n_threads)	
	clf.fit(X_2fit,y_2fit)

	print("\n\n***** Len X_test: ",str(len(X_test)))
	
	predictions = clf.predict(X_test) 
	
	reportGS(clf, y_test, predictions, report_file)
def _log(args):

	log_file = os.path.join(os.path.basename(args.output_report), "log.txt")

	with open(log_file, "a") as file:
		file.write("\n %s classify_split runned with following arguments: %s" %(time.ctime(), str(vars(args))) )
	
	
if __name__ == '__main__':
	args = getArgs()
	if args.seed:
		random_state = args.seed
	(descriptors, sample_indexes, labels) = getDescriptorsSplit(args.input_fvec, args.input_set_labels, L2, power_norm, power_exp, std)
	classify(descriptors, labels, sample_indexes, args.output_report)
