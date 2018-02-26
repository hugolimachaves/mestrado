import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



scoring='accuracy' #metric used to decide the best set of parameters in gridSearch. 'f1' = f_score, 'accuracy'

def classifierGS(clf_model, custom_iterator_or_n_folds, random_state, n_threads):
	if(clf_model == "rf"):
		pipeline = Pipeline([	
			('clf', RandomForestClassifier(criterion='entropy',random_state=random_state,oob_score=False))
		])
		parameters = {
			'clf__n_estimators': (10, 50, 100, 200, 300),
			'clf__max_depth': (50, 150, 250),
			'clf__min_samples_split': (1, 2, 3),
			'clf__min_samples_leaf': (1, 2, 3)
		}
	elif (clf_model == "svm"):
		pipeline = Pipeline([	
			('clf', SVC(kernel='rbf', random_state=random_state))
		])
		parameters = {
			'clf__gamma': (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 12.5, 3.125, 1.3889, 0.7812, 0.5),
			'clf__C': (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5)
		}
	elif (clf_model == "svm_linear"):
		print ("runing LinearSVC")
		pipeline = Pipeline([	
			('clf', LinearSVC(random_state=random_state))
		])
		parameters = {
			'clf__C': (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5)
		}
	else:
		pipeline = Pipeline([	
			('clf', knn())
		])
		parameters = {
			'clf__n_neighbors': (1,2,3,4,5),
		}	

	#Train
	grid_search = GridSearchCV(pipeline, parameters, n_jobs=n_threads, verbose=1, scoring=scoring, cv=custom_iterator_or_n_folds)

	return grid_search


def reportGS(clf, y_test, predictions, report_file):
	with open(report_file, "a") as f:
		#Train report
		f.write('\nGRID SEARCH:\n')
		f.write('\tBest score: %0.2f \n\n' % clf.best_score_)
		f.write('\tBest parameters set:\n')
		best_parameters = clf.best_estimator_.get_params()
		for param_name in sorted(best_parameters.keys()):
			f.write('\t\t%s: %r \n' % (param_name, best_parameters[param_name]))
		f.write('\n-----------------------------------------------------------\n')

		#Test report
		f.write('\nTEST:\n\t')
		f.write(classification_report(y_test, predictions))
	
		f.write('\nAccuracy: %0.4f\n' % accuracy_score(y_test, predictions))

		cm = confusion_matrix(y_test, predictions)
		f.write('\nConfusion matrix:\n')
		f.write(np.array2string(cm, separator=' '))
		plt.matshow(cm)
		plt.title('Confusion matrix')
		plt.colorbar()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.savefig(report_file+'.png')



def report(clf, y_test, predictions, report_file):
	with open(report_file, "a") as f:
		#Test report
		f.write('\nTEST:\n\t')
		f.write(classification_report(y_test, predictions))
	
		f.write('\nAccuracy: %0.4f\n' % accuracy_score(y_test, predictions))

		cm = confusion_matrix(y_test, predictions)
		f.write('\nConfusion matrix:\n')
		f.write(np.array2string(cm, separator=' '))
		plt.matshow(cm)
		plt.title('Confusion matrix')
		plt.colorbar()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.savefig(report_file+'.png')
