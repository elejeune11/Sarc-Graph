import av
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as pilimage
from skimage import io
import os 
import cv2
import glob
import imageio
import moviepy.editor as mp
##########################################################################################
##########################################################################################
# 	Instructions:
#		The goal of this file is to convert the image or movie into matrices.
#		This is all done in this file to keep this step outside the main code. 
#		This may require some changes to accomodate different file types
##########################################################################################
##########################################################################################
def file_pre_processing(file_name,extension='avi'):
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
		container = av.open(folder_input + '.' + extension)
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

##########################################################################################
def file_pre_processing_tif(file_name):
	folder_name = 'ALL_MOVIES_MATRICES'
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)	
	
	folder_output = folder_name + '/' + file_name + '_matrices'
	if not os.path.exists(folder_output):
		os.makedirs(folder_output)
	
	im = io.imread('ALL_MOVIES_RAW/' + file_name + '/' + file_name + '.tif')
	
	for kk in range(0,im.shape[0]):
		frame_npy = 0.2989 * im[kk,:,:,0] + 0.5870 * im[kk,:,:,1] +  0.1140 * im[kk,:,:,2]
		np.save(folder_output + '/frame-%04d' % (kk), frame_npy)
	
	plt.figure()
	plt.imshow(frame_npy)
	plt.axis('off')
	plt.title(file_name)
	plt.savefig(folder_output + '/sample_image.png')
	
	
##########################################################################################
def file_pre_processing_tif2(file_name):
	folder_name = 'ALL_MOVIES_MATRICES'
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)	
	
	folder_output = folder_name + '/' + file_name + '_matrices'
	if not os.path.exists(folder_output):
		os.makedirs(folder_output)
	
	im = io.imread('ALL_MOVIES_RAW/' + file_name + '/' + file_name + '.tif')
	
	for kk in range(0,im.shape[0]):
		frame_npy = im[kk,:,:]
		np.save(folder_output + '/frame-%04d' % (kk), frame_npy)
	
	plt.figure()
	plt.imshow(frame_npy)
	plt.axis('off')
	plt.title(file_name)
	plt.savefig(folder_output + '/sample_image.png')

##########################################################################################
def file_pre_processing_Kehan(file_name, file_source, is_avi):
	folder_name = 'ALL_MOVIES_MATRICES'
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)	
	
	folder_output = folder_name + '/' + file_name + '_matrices'
	if not os.path.exists(folder_output):
		os.makedirs(folder_output)
	
	# read image
	if is_avi:
		container = av.open('Kehan_Movies/' + file_source + '.avi')
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
	else:
		im = io.imread('Kehan_Movies/' + file_source + '.tif')
		for kk in range(0,im.shape[0]):
			frame_npy = im[kk,:,:]
			np.save(folder_output + '/frame-%04d' % (kk), frame_npy)
	
	plt.figure()
	plt.imshow(frame_npy)
	plt.axis('off')
	plt.title(file_name)
	plt.savefig(folder_output + '/sample_image.png')

##########################################################################################
def file_pre_processing_Kehan_timelapse(file_name, file_source, channel):
	folder_name = 'ALL_MOVIES_MATRICES'
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)	
	
	folder_output = folder_name + '/' + file_name + '_channel_%i'%(channel) + '_matrices'
	if not os.path.exists(folder_output):
		os.makedirs(folder_output)
	
	im = io.imread( file_source + '.tif')
	
	for kk in range(0,im.shape[0]):
		frame_npy = im[kk,channel,:,:]
		np.save(folder_output + '/frame-%04d' % (kk), frame_npy)
	
	plt.figure()
	plt.imshow(frame_npy)
	plt.axis('off')
	plt.title(file_name)
	plt.savefig(folder_output + '/sample_image.png')
	
##########################################################################################
def file_pre_processing_template(file_name):
	folder_name = 'ALL_MOVIES_MATRICES'
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)
	
	folder_output = folder_name + '/' + file_name + '_matrices'
	if not os.path.exists(folder_output):
		os.makedirs(folder_output)
	
	# im = read the movie in
	
	# for kk in range(0,num_movie_frames):
	# 	frame_npy = one image frame
	# 	np.save(folder_output + '/frame-%04d' % (kk), frame_npy)
	
	plt.figure()
	plt.imshow(frame_npy)
	plt.axis('off')
	plt.title(file_name)
	plt.savefig(folder_output + '/sample_image.png')
	
##########################################################################################
def get_frame_matrix(folder_name, frame):
	"""Get the npy matrix for a frame of the movie."""
	if frame < 10: file_root = '_matrices/frame-000%i'%(frame)
	elif frame < 100: file_root = '_matrices/frame-00%i'%(frame)
	else: file_root = '_matrices/frame-0%i'%(frame)
	root = 'ALL_MOVIES_MATRICES/' + folder_name + file_root  + '.npy'
	raw_img = np.load(root)
	return raw_img

##########################################################################################
def make_movie_from_npy(file_name,include_eps=False): 
	folder_name = 'ALL_MOVIES_MATRICES'
	folder_output = folder_name + '/' + file_name + '_matrices'
	folder_output_movie = folder_name + '/' + file_name + '_matrices/movie'
	if not os.path.exists(folder_output_movie):
		os.makedirs(folder_output_movie)
	
	img_list = [] 
	num_frames = len(glob.glob('ALL_MOVIES_MATRICES/' + file_name + '_matrices/*.npy'))
	for kk in range(0,num_frames):
		raw_img = get_frame_matrix(file_name, kk)
		plt.figure()
		plt.imshow(raw_img, cmap=plt.cm.gray)
		ax = plt.gca()
		ax.set_xticks([]); ax.set_yticks([])
		plt.savefig(folder_output_movie + '/' + 'frame_%04d.png'%(kk),bbox_inches = 'tight', pad_inches = 0)
		if include_eps:
			plt.savefig(folder_output_movie + '/' + 'frame_%i.eps'%(kk),bbox_inches = 'tight', pad_inches = 0)
		plt.close()
		img_list.append(imageio.imread(folder_output_movie + '/' + 'frame_%04d.png'%(kk)))
	
	imageio.mimsave(folder_output_movie + '/contract_anim.gif', img_list)	
	clip = mp.VideoFileClip(folder_output_movie + '/contract_anim.gif')
	clip.write_videofile( folder_output_movie + '/' + file_name + '.mp4')
	
	return
	

