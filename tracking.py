import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  
import trackpy as tp
import os
import trackpy.predict
import sys
import glob

##########################################################################################
# Input info and set up 
##########################################################################################
def run_all_tracking(folder_name,tp_depth): 
	"""Run all tracking -- z disks and sarcomeres."""
	num_frames = len(glob.glob('ALL_MOVIES_MATRICES/' + folder_name + '_matrices/*.npy'))

	external_folder_name = 'ALL_MOVIES_PROCESSED'
	if not os.path.exists(external_folder_name): os.makedirs(external_folder_name)

	out_track = external_folder_name + '/' + folder_name + '/tracking_results'

	if not os.path.exists(external_folder_name + '/' + folder_name): os.makedirs(external_folder_name + '/' + folder_name)
	if not os.path.exists(out_track): os.makedirs(out_track)

	##########################################################################################
	# Helper functions
	##########################################################################################

	##########################################################################################
	def fakeframe_bands(frame):
		"""Create a pandas dataframe that captures the z-disks."""
		if frame < 10: file_root = '/frame-000%i'%(frame)
		elif frame < 100: file_root = '/frame-00%i'%(frame)
		else: file_root = '/frame-0%i'%(frame)
		filename = 'ALL_MOVIES_PROCESSED/' + folder_name + '/segmented_bands' + file_root + '_bands.txt'
		file = np.loadtxt(filename)
		cent_idx1 = file[:,0]
		cent_idx2 = file[:,1]
		end1_idx1 = file[:,2]
		end1_idx2 = file[:,3]
		end2_idx1 = file[:,4]
		end2_idx2 = file[:,5]
		fake_mass = np.ones((file.shape[0]))*11
		orig_idx_all = np.arange(0,file.shape[0],1)
		return pd.DataFrame(dict( y=cent_idx2, x=cent_idx1, orig_idx=orig_idx_all, end1_idx1=end1_idx1, end1_idx2=end1_idx2, end2_idx1=end2_idx1,end2_idx2=end2_idx2,mass=fake_mass,frame=frame))

	##########################################################################################
	def fakeframe_sarc(frame):
		"""Create a pandas dataframe that captures the sarcomeres."""
		if frame < 10: file_root = '/frame-000%i'%(frame)
		elif frame < 100: file_root = '/frame-00%i'%(frame)
		else: file_root = '/frame-0%i'%(frame)
		filename = 'ALL_MOVIES_PROCESSED/' + folder_name + '/segmented_sarc' + file_root  + '_sarc_data.txt'
		file = np.loadtxt(filename)
		ZLID1 = file[:,0]
		ZLID2 = file[:,1]
		x_cent = file[:,2]
		y_cent = file[:,3]
		length = file[:,4]
		width = file[:,5]
		angle = file[:,6]
		fake_mass = np.ones((file.shape[0]))*11
		orig_idx_all = np.arange(0,file.shape[0],1)
		return pd.DataFrame(dict( y=y_cent, x=x_cent, orig_idx=orig_idx_all, ZLID1=ZLID1,ZLID2=ZLID2,length=length,width=width,angle=angle,mass=fake_mass,frame=frame))

	##########################################################################################
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	# Track z-disks 
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	##########################################################################################
	# load all of the segmented z-disks into a features array 
	frames = []
	for kk in range(0,num_frames):
		frames.append(fakeframe_bands(kk))

	features = pd.concat(frames)

	# Run tracking --> using the trackpy package 
	# http://soft-matter.github.io/trackpy/v0.3.0/tutorial/prediction.html
	t = tp.link_df(features, tp_depth, memory=int(num_frames)) 
	t1 = tp.filter_stubs(t, int(num_frames*.10))

	# Extract the results from tracking 
	frame = t1.frame.to_numpy()
	xall = t1.x.to_numpy()
	yall = t1.y.to_numpy()
	orig_idx = t1.orig_idx.to_numpy()
	particle = t1.particle.to_numpy()
	end1_idx1 = t1.end1_idx1.to_numpy()
	end1_idx2 = t1.end1_idx2.to_numpy()
	end2_idx1 = t1.end2_idx1.to_numpy()
	end2_idx2 = t1.end2_idx2.to_numpy()

	save_data = np.zeros((frame.shape[0],9))
	save_data[:,0] = frame
	save_data[:,1] = orig_idx
	save_data[:,2] = particle
	save_data[:,3] = xall
	save_data[:,4] = yall 
	save_data[:,5] = end1_idx1
	save_data[:,6] = end1_idx2
	save_data[:,7] = end2_idx1
	save_data[:,8] = end2_idx2

	np.savetxt(out_track + '/'+'tracking_results_zdisks.txt',save_data)

	##########################################################################################
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	# Track sarcomeres
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	##########################################################################################
	# load all of the segmented sarcomeres into a features array 
	features = []
	for kk in range(0,num_frames):
		features.append(fakeframe_sarc(kk))

	features = pd.concat(features)

	# Run tracking --> using the trackpy package 
	# http://soft-matter.github.io/trackpy/v0.3.0/tutorial/prediction.html
	t = tp.link_df(features, tp_depth, memory=int(num_frames))
	t1 = tp.filter_stubs(t, int(num_frames*.10))

	frame = t1.frame.to_numpy()
	xall = t1.x.to_numpy()
	yall = t1.y.to_numpy()
	orig_idx = t1.orig_idx.to_numpy()
	particle = t1.particle.to_numpy()
	ZLID1 = t1.ZLID1.to_numpy()
	ZLID2 = t1.ZLID2.to_numpy()
	leng_all = t1.length.to_numpy()
	wid_all = t1.width.to_numpy()
	ang_all = t1.angle.to_numpy()

	save_data = np.zeros((frame.shape[0],10))
	save_data[:,0] = frame
	save_data[:,1] = orig_idx
	save_data[:,2] = particle
	save_data[:,3] = xall
	save_data[:,4] = yall 
	save_data[:,5] = ZLID1
	save_data[:,6] = ZLID2
	save_data[:,7] = leng_all
	save_data[:,8] = wid_all
	save_data[:,9] = ang_all

	np.savetxt(out_track + '/'+'tracking_results_sarcomeres.txt',save_data)
	
	return

