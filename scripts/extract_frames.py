import numpy as np
import cv2
import argparse
import os

encode = "utf-8"

def _get_Args():

        parser = argparse.ArgumentParser()
        parser.add_argument("video", action='store', type=str, help="Video path", default='list')
        return parser.parse_args()

def read_video(name):

	frames = []
	vid = cv2.VideoCapture(name)

	if vid:
		frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
		kernel = (5,5)

		for i in range(frame_count):
			ret,frame = vid.read() #ret=false if something goes wrong
			if ret is False:
				break;
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert frame to grayscale
			frame = cv2.GaussianBlur(frame,kernel,0) #Gaussian blur filter to reduce noise
			frames.append(frame)
	return frames

def write_video_frames(name, frames):
	
	#Extract name
	name = os.path.split(name)[-1]

	#Make a directory in the video folder with the frames	
	dir_ = 'frames' + os.path.sep + name
	try:
		os.makedirs(dir_)
		#save frames
		for i in range(len(frames)):
			name_frame = dir_ + os.path.sep + str(i) + '.png'
			cv2.imwrite(name_frame, frames[i])
	except:
		raise IOError("Directory can't be created or already exists: " + dir_)

def _main(args):
	
	frames = read_video(args.video)
	write_video_frames(args.video,frames)

if __name__ == '__main__':
	# parse arguments
	args = _get_Args()
	_main(args)
	
