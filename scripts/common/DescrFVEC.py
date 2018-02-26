import numpy as np
import struct
from sklearn import preprocessing
from sklearn.preprocessing import normalize

CintSize = 4 #bytes for a integer in C
CfloatSize = 4 #bytes for a float in C

def getData(fvec_file, L2=False, power_norm=False, power_exp=0.2, std=False):
	f = open(fvec_file, 'rb')
	chunks = f.read()
	descr_size = struct.unpack('i',chunks[0:CintSize])[0] #unpack the first int (descriptor size n)
	num_videos = int(len(chunks)/(CintSize + descr_size*CfloatSize)) #get the number of videos
	fmt = ('i'+str(descr_size)+'f')*num_videos #format to unpack the entire file
	descr_list = struct.unpack(fmt,chunks) 
	descr_list = np.array(descr_list,dtype=float).reshape((num_videos, 1+descr_size))[:,1:]

	if(power_norm): descr_list = np.power(descr_list, power_exp)
	if(L2):	normalize(descr_list, copy=False, axis=1)
	if(std): descr_list = preprocessing.scale(descr_list)

	return descr_list


#Get the descriptors and labels from files and transform the data according to the selected methods
#fvec_file is the path to a binary file with m n-dimensional descriptors: <n><descr_1[n]>...<n><descr_m[n]> 
#labels_file is the path to a text file with m tuples: <video_name> <label>
def getDescriptors(fvec_file, labels_file, L2=False, power_norm=False, power_exp=0.2, std=False):
	descr_list = getData(fvec_file, L2, power_norm, power_exp, std)

	labels = (np.loadtxt(labels_file, comments='\\', dtype=str)[:,1]).astype(int)
	
	if labels.size != descr_list.shape[0]:
		print ("Error: Number of labels must be equal to number of videos.")
		return (None,None)
	
	return (descr_list,labels)


#Get the descriptors, sets and labels from files and transform the data according to the selected methods
#fvec_file is the path to a binary file with m n-dimensional descriptors: <n><descr_1[n]>...<n><descr_m[n]> 
#set_label_file is the path to a text file with m tuples: <video_name> <set> <label>
def getDescriptorsSplit(fvec_file, set_label_file, L2=False, power_norm=False, power_exp=0.2, std=False):
	descr_list = getData(fvec_file, L2, power_norm, power_exp, std)

	set_label = np.loadtxt(set_label_file, comments='\\', dtype=str)

	sets = set_label[:,1].astype(int)
	train_set = np.where(sets==0)[0]
	valid_set = np.where(sets==1)[0]
	test_set = np.where(sets==2)[0]

	labels = set_label[:,2].astype(int)

	if labels.size != descr_list.shape[0]:
		print ("Error: Number of labels must be equal to number of videos.")
		return (None,None, None)
	
	return (descr_list, (train_set, valid_set, test_set), labels)

#append descriptor to file
def saveDescriptor(filename, descriptor):
	descr_size = descriptor.size
	fmt = 'i'+str(descr_size)+'f'
	f = open(filename, 'ab')
	chunk = struct.pack(fmt,descr_size,*descriptor)
	f.write(chunk)

