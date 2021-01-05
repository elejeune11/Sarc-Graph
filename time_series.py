import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os 
import pickle
from matplotlib import cm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, WhiteKernel, ConstantKernel as C, ExpSineSquared
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
import sys
import cv2
import glob
##########################################################################################

##########################################################################################
# Input info and set up 
##########################################################################################
def timeseries_all(folder_name, keep_thresh=0.75,save_for_vis=False):
	"""Perform the full timeseries analysis."""
	num_frames = len(glob.glob('ALL_MOVIES_MATRICES/' + folder_name + '_matrices/*.npy'))
	external_folder_name = 'ALL_MOVIES_PROCESSED'
	if not os.path.exists(external_folder_name):
		os.makedirs(external_folder_name)

	out_time = external_folder_name + '/' + folder_name + '/timeseries'

	if not os.path.exists(external_folder_name + '/' + folder_name): os.makedirs(external_folder_name + '/' + folder_name)
	if not os.path.exists(out_time): os.makedirs(out_time)

	##########################################################################################
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	# Set up timeseries  
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	##########################################################################################

	# --> import sarcomere data from tracking 
	sarc_data_fname = external_folder_name + '/' + folder_name + '/tracking_results/tracking_results_sarcomeres.txt'
	save_data = np.loadtxt(sarc_data_fname)
	all_frames = save_data[:,0]
	all_particles = save_data[:,2]
	all_leng = save_data[:,7]
	all_wid = save_data[:,8]
	all_ang = save_data[:,9]

	# --> figure out the unique sarcomere IDs (from particle ID)
	# --> set up matrices -- unique sarcomeres x number of frames 
	unique_particles = np.unique(all_particles).astype('int')
	organized_data = np.zeros((unique_particles.shape[0],num_frames))
	organized_data_wid = np.zeros((unique_particles.shape[0],num_frames))
	organized_data_ang = np.zeros((unique_particles.shape[0],num_frames))

	for kk in range(0,save_data.shape[0]):
		part = int(all_particles[kk])
		frame = int(all_frames[kk])
		part_idx = np.argmin(np.abs(unique_particles - part))
		organized_data[part_idx,frame] = all_leng[kk]
		organized_data_wid[part_idx,frame] = all_wid[kk]
		organized_data_ang[part_idx,frame] = all_ang[kk]

	organized_x = np.zeros((unique_particles.shape[0],num_frames))
	organized_y = np.zeros((unique_particles.shape[0],num_frames))

	for kk in range(0,save_data.shape[0]):
		part = int(all_particles[kk])
		frame = int(all_frames[kk])
		part_idx = np.argmin(np.abs(unique_particles - part))
		organized_x[part_idx,frame] = save_data[kk,3]
		organized_y[part_idx,frame] = save_data[kk,4]

	# --> get all of the timeseries -- keep the missing 
	ALL_frames_recorded = []; ALL_leng_recorded = []; ALL_record_size = [] 
	ALL_x_pos = []; ALL_y_pos = []; ALL_wid_recorded = []; ALL_ang_recorded = [] 

	for kk in range(0,organized_data.shape[0]):
		frames_recorded = []; leng_recorded = [] 
		x_recorded = []; y_recorded = [] 
		wid_recorded = []; ang_recorded = [] 
		for jj in range(0,organized_data.shape[1]):
			if organized_data[kk,jj] > 0:
				frames_recorded.append(jj)
				leng_recorded.append(organized_data[kk,jj])
				x_recorded.append(organized_x[kk,jj])
				y_recorded.append(organized_y[kk,jj])
				wid_recorded.append(organized_data_wid[kk,jj])
				ang_recorded.append(organized_data_ang[kk,jj])
			
		ALL_frames_recorded.append(frames_recorded)
		ALL_leng_recorded.append(leng_recorded) 
		ALL_record_size.append(len(frames_recorded))
		ALL_x_pos.append(x_recorded)
		ALL_y_pos.append(y_recorded)
		ALL_wid_recorded.append(wid_recorded)
		ALL_ang_recorded.append(ang_recorded)

	# --> selectively identify timeseries where 75% is present 
	min_exists = num_frames*keep_thresh
	ALL_frames_above_thresh = []; ALL_leng_above_thresh = []; ALL_idx_above_thresh = [] 
	ALL_x_pos_above_thresh = []; ALL_y_pos_above_thresh = [] 
	ALL_wid_above_thresh = []; ALL_ang_above_thresh = [] 

	for kk in range(0,len(ALL_frames_recorded)):
		if ALL_record_size[kk] > min_exists:
			ALL_frames_above_thresh.append(ALL_frames_recorded[kk])
			ALL_leng_above_thresh.append(ALL_leng_recorded[kk])
			ALL_idx_above_thresh.append(kk)
			ALL_x_pos_above_thresh.append(ALL_x_pos[kk])
			ALL_y_pos_above_thresh.append(ALL_y_pos[kk])
			ALL_wid_above_thresh.append(ALL_wid_recorded[kk]) 
			ALL_ang_above_thresh.append(ALL_ang_recorded[kk])
		
	##########################################################################################
	# --> do some smoothing and fill in gaps with GPR 
	##########################################################################################
	ALL_frames_GPR = []; ALL_leng_GPR = [] 
	ALL_wid_GPR = []; ALL_ang_GPR = [] 
	ALL_x_pos_GPR = []; ALL_y_pos_GPR = [] 

	# --> use GPR to interpolate sarcomere length 
	for kk in range(0,len(ALL_frames_above_thresh)):
		kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 10.0)) + 1.0 * WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 10.0))
		#  model.kernel_ < -- give the fitted kernel 
		model = GaussianProcessRegressor(kernel=kernel,normalize_y=True)
		xdata = np.asarray(ALL_frames_above_thresh[kk])
		ydata = np.asarray(ALL_leng_above_thresh[kk])

		model.fit(xdata.reshape(-1, 1),ydata)
		x_hat = np.linspace(0,num_frames-1,num_frames)
		y_hat = model.predict(x_hat.reshape(-1,1))
	
		ALL_frames_GPR.append(x_hat)
		ALL_leng_GPR.append(y_hat)

	# --> use GPR to interpolate sarcomere width
	for kk in range(0,len(ALL_frames_above_thresh)):
		kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 10.0)) + 1.0 * WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 10.0))
		#  model.kernel_ < -- give the fitted kernel 
		model = GaussianProcessRegressor(kernel=kernel,normalize_y=True)
		xdata = np.asarray(ALL_frames_above_thresh[kk])
		ydata = np.asarray(ALL_wid_above_thresh[kk])
		model.fit(xdata.reshape(-1, 1),ydata)
		y_hat = model.predict(x_hat.reshape(-1,1))
		ALL_wid_GPR.append(y_hat)

	# --> use GPR to interpolate sarcomere angle 
	for kk in range(0,len(ALL_frames_above_thresh)):
		kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 10.0)) + 1.0 * WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 10.0))
		#  model.kernel_ < -- give the fitted kernel 
		model = GaussianProcessRegressor(kernel=kernel,normalize_y=True)
		xdata = np.asarray(ALL_frames_above_thresh[kk])
		ydata = np.asarray(ALL_ang_above_thresh[kk])
		model.fit(xdata.reshape(-1, 1),ydata)
		y_hat = model.predict(x_hat.reshape(-1,1))
		ALL_ang_GPR.append(y_hat)

	# --> use GPR to interpolate x position
	for kk in range(0,len(ALL_frames_above_thresh)):
		kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 10.0)) + 1.0 * WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 10.0))
		#  model.kernel_ < -- give the fitted kernel 
		model = GaussianProcessRegressor(kernel=kernel,normalize_y=True)
		xdata = np.asarray(ALL_frames_above_thresh[kk])
		ydata = np.asarray(ALL_x_pos_above_thresh[kk])
		model.fit(xdata.reshape(-1, 1),ydata)
		y_hat = model.predict(x_hat.reshape(-1,1))
		ALL_x_pos_GPR.append(y_hat)

	# --> use GPR to interpolate y position
	for kk in range(0,len(ALL_frames_above_thresh)):
		kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 10.0)) + 1.0 * WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 10.0))
		#  model.kernel_ < -- give the fitted kernel 
		model = GaussianProcessRegressor(kernel=kernel,normalize_y=True)
		xdata = np.asarray(ALL_frames_above_thresh[kk])
		ydata = np.asarray(ALL_y_pos_above_thresh[kk])
		model.fit(xdata.reshape(-1, 1),ydata)
		y_hat = model.predict(x_hat.reshape(-1,1))
		ALL_y_pos_GPR.append(y_hat)

	##########################################################################################
	# convert from distance to strains
	##########################################################################################
	arr_wid = np.asarray(ALL_wid_GPR)
	arr_ang = np.asarray(ALL_ang_GPR)
	arr_frames = np.asarray(ALL_frames_GPR)
	arr_leng = np.asarray(ALL_leng_GPR)
	ALL_x_pos_GPR = np.asarray(ALL_x_pos_GPR)
	ALL_y_pos_GPR = np.asarray(ALL_y_pos_GPR)
	all_normalized = np.zeros(arr_leng.shape)

	for kk in range(0,arr_frames.shape[0]):
		me = np.mean(arr_leng[kk,:])
		all_normalized[kk,:] = (arr_leng[kk,:] - me)/me

	##########################################################################################
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	# Save everything  
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	##########################################################################################
	if save_for_vis:
		tag_vis = 'for_plotting_'
	else:
		tag_vis = ''
	
	sarc_data_normalized_fname = out_time + '/' + tag_vis + 'tracking_results_frames.txt'
	np.savetxt(sarc_data_normalized_fname, arr_frames)
	sarc_data_normalized_fname = out_time + '/' + tag_vis + 'tracking_results_leng.txt'
	np.savetxt(sarc_data_normalized_fname, all_normalized)
	sarc_data_fname = out_time + '/' + tag_vis + 'tracking_results_leng_NOT_NORMALIZED.txt'
	np.savetxt(sarc_data_fname, arr_leng)
	sarc_data_fname = out_time + '/' + tag_vis + 'tracking_results_wid.txt'
	np.savetxt(sarc_data_fname,ALL_wid_GPR)
	sarc_data_fname = out_time + '/' + tag_vis + 'tracking_results_ang.txt'
	np.savetxt(sarc_data_fname,ALL_ang_GPR)
	sarc_data_fname = out_time + '/' + tag_vis + 'tracking_results_x_pos.txt'
	np.savetxt(sarc_data_fname,ALL_x_pos_GPR)
	sarc_data_fname = out_time + '/' + tag_vis + 'tracking_results_y_pos.txt'
	np.savetxt(sarc_data_fname,ALL_y_pos_GPR)


	sarc_idx_fname = out_time + '/' + tag_vis + 'tracking_results_sarc_idx_above_thresh.txt'
	np.savetxt(sarc_idx_fname, np.asarray(ALL_idx_above_thresh))

	plot_info_frames_fname = out_time + '/' + tag_vis + 'plotting_all_frames.pkl'
	with open(plot_info_frames_fname, 'wb') as f:
		pickle.dump(ALL_frames_above_thresh,f)

	plot_info_x_pos_fname = out_time + '/' + tag_vis + 'plotting_all_x.pkl'
	with open(plot_info_x_pos_fname, 'wb') as f:
		pickle.dump(ALL_x_pos_above_thresh, f)

	plot_info_y_pos_fname = out_time + '/' + tag_vis + 'plotting_all_y.pkl'
	with open(plot_info_y_pos_fname, 'wb') as f:
		pickle.dump(ALL_y_pos_above_thresh, f)

	return

