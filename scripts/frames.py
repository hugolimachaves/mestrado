import numpy as np
from numpy import linalg as la
import cv2
import math
import argparse
import struct
from math import exp
import os.path
import sys
from scipy import signal, linalg
#import cPickle
import _pickle as cPickle
from os import listdir
from os.path import isfile, join

def _get_Args():
        parser = argparse.ArgumentParser()
        parser.add_argument("video", action='store', type=str, help="Video path", default='list')
        return parser.parse_args()

#lista de frames de um video
def videoCapture(l):
	
	frame_list = np.array([])
	vid = cv2.VideoCapture(l)
	if vid:
		width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
		frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
		fps = float(vid.get(cv2.CAP_PROP_FPS))
	
		print ("width %d" % width)
		print ("height %d" % height)
		print ("frame_count %d" % frame_count)
		print ("fps %d" % fps)

		for i in range (frame_count):
			ret, frame = vid.read()
			if ret == False:
				break;
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			frame = cv2.GaussianBlur(frame,(5,5),0)
			#salvar frame como imagem nesse ponto


if __name__ == '__main__':
	# parse arguments
	args = get_Args()
	videoCapture(args.video)
