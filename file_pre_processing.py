import av
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as pilimage
from skimage import io
import os 
import cv2
##########################################################################################
##########################################################################################
# 	Instructions:
#		The goal of this file is to convert the image or movie into matrices.
#		This is all done in this file to keep this step outside the main code. 
#		This may require some changes to accomodate different file types
##########################################################################################
##########################################################################################
def file_pre_processing(file_name):
	folder_name = 'ALL_MOVIES_MATRICES'
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)
	##########################################################################################
	file_names_list = [file_name]

	for kk in range(0,len(file_names_list)):
	
		t1 = file_names_list[kk]
	
		folder_output = folder_name + '/' + t1 + '_matrices'
		if not os.path.exists(folder_output):
			os.makedirs(folder_output)
		
		folder_input  = 'ALL_MOVIES_RAW/' + t1 + '/' + t1 
		container = av.open(folder_input + '.avi')
		for frame in container.decode(video=0):
			frame_img = frame.to_image()
			frame_npy = np.array(frame_img)
			max_li = [] 
			num_frames = frame_npy.shape[2]
			if num_frames > 0:
				arr = 0.2989 * frame_npy[:,:,0] + 0.5870 * frame_npy[:,:,1] + 0.1140 * frame_npy[:,:,2]
			else:
				arr = frame_npy
	
			np.save(folder_output + '/frame-%04d' % frame.index, arr)
	
		# add a png of the image just to see -- not necessary 
		plt.figure()
		plt.imshow(arr)
		plt.axis('off')
		plt.title(t1)
		plt.savefig(folder_output + '/sample_image.png')
