import pickle
import cv2
from skimage.filters import threshold_otsu, threshold_local
from skimage import measure
from scipy import ndimage
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import imageio
import scipy
from scipy import signal
from skimage.feature import peak_local_max
from scipy.signal import find_peaks
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from scipy.signal import find_peaks
import csv
import pandas as pd
import random 
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, leaves_list
import networkx as nx
import scipy
from scipy.linalg import polar
from numpy import linalg as LA
import time_series as ts
import moviepy.editor as mp
##########################################################################################
# visualization on the images 
##########################################################################################

##########################################################################################
def visualize_segmentation(folder_name, gaussian_filter_size=1,frame_num=0,include_eps=False):
	"""Visualize the results of z-disk and sarcomere segmentation."""
	external_folder_name = 'ALL_MOVIES_PROCESSED'
	if not os.path.exists(external_folder_name):
		os.makedirs(external_folder_name)

	out_analysis = external_folder_name + '/' + folder_name + '/analysis'
	if not os.path.exists(external_folder_name + '/' + folder_name): os.makedirs(external_folder_name + '/' + folder_name)
	if not os.path.exists(out_analysis): os.makedirs(out_analysis)

	# --> visualize segmentation
	raw_img = np.load('ALL_MOVIES_MATRICES/' + folder_name + '_matrices/frame-%04d.npy'%(frame_num))
	# plot of segmented z disks
	box = -1
	laplacian = cv2.Laplacian(raw_img,cv2.CV_64F)
	laplacian = ndimage.gaussian_filter(laplacian, gaussian_filter_size)
	contour_thresh = threshold_otsu(laplacian)
	contour_image = laplacian
	contours =  measure.find_contours(contour_image,contour_thresh)

	total = 0
	contour_list = [] 
	for n, contour in enumerate(contours):
		total += 1
		if contour.shape[0] >= 8:
			contour_list.append(contour)

	band_data = np.loadtxt('ALL_MOVIES_PROCESSED/' + folder_name + '/segmented_bands/frame-%04d_bands.txt'%(frame_num))
	z_disc_x = band_data[:,0]
	z_disc_y = band_data[:,1]

	# --> import sarcomeres 
	sarc_data = np.loadtxt('ALL_MOVIES_PROCESSED/' + folder_name + '/segmented_sarc/frame-%04d_sarc_data.txt'%(frame_num))
	sarc_x = sarc_data[:,2]
	sarc_y = sarc_data[:,3]

	fig, axs = plt.subplots(1,2,figsize=(10,5))
	axs[0].imshow(raw_img, cmap=plt.cm.gray); axs[0].set_title('z-disks -- frame %i, %i found'%(frame_num,len(contour_list)))
	for kk in range(0,len(contour_list)):
		cont = contour_list[kk]
		axs[0].plot(cont[:,1],cont[:,0])
	
	axs[0].set_xticks([]); axs[0].set_yticks([])
	axs[1].imshow(raw_img, cmap=plt.cm.gray); axs[1].set_title('sarcomeres -- frame %i, %i found'%(frame_num,sarc_x.shape[0]))
	axs[1].plot(sarc_y,sarc_x,'r*',markersize=3)
	axs[1].set_xticks([]); axs[1].set_yticks([])
	plt.savefig(out_analysis + '/visualize_segmentation_%04d'%(frame_num))
	if include_eps:
		plt.savefig(out_analysis + '/visualize_segmentation_%04d.eps'%(frame_num))
	return

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
def visualize_contract_anim_movie(folder_name,re_run_timeseries=False, use_re_run_timeseries=False, keep_thresh=0.75,include_eps=False,single_frame=False):
	"""Visualize the results of tracking."""
	external_folder_name = 'ALL_MOVIES_PROCESSED'
	if not os.path.exists(external_folder_name):
		os.makedirs(external_folder_name)

	out_analysis = external_folder_name + '/' + folder_name + '/analysis'
	if not os.path.exists(external_folder_name + '/' + folder_name): os.makedirs(external_folder_name + '/' + folder_name)
	if not os.path.exists(out_analysis): os.makedirs(out_analysis)

	num_frames = len(glob.glob('ALL_MOVIES_MATRICES/' + folder_name + '_matrices/*.npy'))
	if single_frame:
		num_frames = 1
	
	if use_re_run_timeseries:
		tag_vis = 'for_plotting_'
		if re_run_timeseries:
			ts.timeseries_all(folder_name, keep_thresh, True)
	else:
		tag_vis = ''
	
	plot_info_frames_fname = 'ALL_MOVIES_PROCESSED/' + folder_name + '/timeseries/' + tag_vis + 'plotting_all_frames.pkl'
	ALL_frames_above_thresh = pickle.load( open( plot_info_frames_fname  , "rb" ) )
	plot_info_x_pos_fname = 'ALL_MOVIES_PROCESSED/' + folder_name + '/timeseries/' + tag_vis + 'plotting_all_x.pkl'
	ALL_x_pos_above_thresh = pickle.load( open( plot_info_x_pos_fname  , "rb" ) )
	plot_info_y_pos_fname = 'ALL_MOVIES_PROCESSED/' + folder_name + '/timeseries/' + tag_vis + 'plotting_all_y.pkl'
	ALL_y_pos_above_thresh = pickle.load( open( plot_info_y_pos_fname  , "rb" ) )
	sarc_data_normalized_fname = 'ALL_MOVIES_PROCESSED/' + folder_name + '/timeseries/' + tag_vis + 'tracking_results_leng.txt'
	all_normalized = np.loadtxt(sarc_data_normalized_fname)
	
	if use_re_run_timeseries:
		out_plots = out_analysis + '/for_plotting_contract_anim'
	else:
		out_plots = out_analysis + '/contract_anim'

	if not os.path.exists(out_plots): os.makedirs(out_plots)

	# --> plot every frame, plot every sarcomere according to normalized fraction length 
	color_matrix = np.zeros(all_normalized.shape)
	for kk in range(0,all_normalized.shape[0]):
		for jj in range(0,all_normalized.shape[1]):
			of = all_normalized[kk,jj]
			if of < -.2: color_matrix[kk,jj] = 0
			elif of > .2: color_matrix[kk,jj] = 1
			else: color_matrix[kk,jj] = of*2.5 + .5

	img_list = [] 
	for t in range(0,num_frames):
		if t < 10: file_root = '/frame-000%i'%(t)
		elif t < 100: file_root = '/frame-00%i'%(t)
		else: file_root = '/frame-0%i'%(t)
		img = get_frame_matrix(folder_name,t)

		plt.figure()
		plt.imshow(img, cmap=plt.cm.gray)
		for kk in range(0,all_normalized.shape[0]):
			if t in ALL_frames_above_thresh[kk]:
				ix = np.argwhere(np.asarray(ALL_frames_above_thresh[kk]) == t)[0][0]
				col = (1-color_matrix[kk,t], 0 , color_matrix[kk,t])
				yy = ALL_y_pos_above_thresh[kk][ix]
				xx = ALL_x_pos_above_thresh[kk][ix]
				plt.scatter(yy,xx,s=15,color=col,marker='o')
	
		ax = plt.gca()
		ax.set_xticks([]); ax.set_yticks([])
		plt.savefig(out_plots + '/' + file_root + '_length')
		if include_eps:
			plt.savefig(out_plots + '/' + file_root + '_length.eps')
		plt.close()
		img_list.append(imageio.imread(out_plots + '/' + file_root + '_length.png'))
	
	if num_frames > 1:
		imageio.mimsave(out_plots + '/contract_anim.gif', img_list)
	
# 	clip = mp.VideoFileClip(out_plots + '/contract_anim.gif')
# 	clip.write_videofile( 'Kehan_Tracked_Movies/' + folder_name + '.mp4') # put all movies in one folder
	
	return

##########################################################################################
# plot the spatial graph 
##########################################################################################

##########################################################################################
def visualize_spatial_graph(folder_name,include_eps=False):
	external_folder_name = 'ALL_MOVIES_PROCESSED'
	if not os.path.exists(external_folder_name):
		os.makedirs(external_folder_name)

	out_analysis = external_folder_name + '/' + folder_name + '/analysis'
	if not os.path.exists(external_folder_name + '/' + folder_name): os.makedirs(external_folder_name + '/' + folder_name)
	if not os.path.exists(out_analysis): os.makedirs(out_analysis)
	
	######################################################################################
	out_graph = 'ALL_MOVIES_PROCESSED' + '/' + folder_name + '/graph'
	with open(out_graph + '/graph.pkl', 'rb') as f: G = pickle.load(f)
	with open(out_graph + '/pos.pkl', 'rb') as f: pos = pickle.load(f)

	# plot spatial graph 
	G2 = nx.Graph()
	nodes_list = list(G.nodes())

	for kk in range(0,len(nodes_list)): G2.add_node(nodes_list[kk])
	edges_list = list(G.edges())
	orient_list = [] 

	for kk in range(0,len(edges_list)):
		# define the angle of the edge
		node_1_ix = edges_list[kk][0]
		node_2_ix = edges_list[kk][1]
		x1 = pos[node_1_ix][0]
		x2 = pos[node_2_ix][0]
		y1 = pos[node_1_ix][1]
		y2 = pos[node_2_ix][1]
		rad = ((x1-x2)**2.0 + (y1-y2)**2.0)**0.5
		x_val = (x2-x1)/rad
		y_val = (y2-y1)/rad
		ang = np.abs(np.dot([1,0],[x_val,y_val]))
		orient_list.append(ang)
		G2.add_edge(node_1_ix,node_2_ix,weight=ang)

	# for each node, determine local alignment -- 
	node_val_list = [] 

	for kk in range(0,len(nodes_list)):
		ix = nodes_list[kk]
		ed_li = list(G.edges(ix))
		val = 0
		num = 0
		for jj in range(0,len(ed_li)):
			for ii in range(jj+1,len(ed_li)):
				node_1a_ix = ed_li[jj][0]
				node_1b_ix = ed_li[jj][1]
				node_2a_ix = ed_li[ii][0]
				node_2b_ix = ed_li[ii][1]
			
				x1a = pos[node_1a_ix][0]
				x1b = pos[node_1b_ix][0]
				y1a = pos[node_1a_ix][1]
				y1b = pos[node_1b_ix][1]
			
				x2a = pos[node_2a_ix][0]
				x2b = pos[node_2b_ix][0]
				y2a = pos[node_2a_ix][1]
				y2b = pos[node_2b_ix][1]
			
				rad1 = ((x1a-x1b)**2.0 + (y1a-y1b)**2.0)**0.5
				rad2 = ((x2a-x2b)**2.0 + (y2a-y2b)**2.0)**0.5
			
				vec1 = [(x1a-x1b)/rad1,(y1a-y1b)/rad1]
				vec2 = [(x2a-x2b)/rad2,(y2a-y2b)/rad2]
			
				val += np.abs(np.dot( vec1 , vec2 ))
				num += 1 
	
		if num > 0:
			node_val_list.append(val/num)
		else:
			node_val_list.append(0)
	
	plt.figure(figsize=(5,5))
	edges,weights = zip(*nx.get_edge_attributes(G2,'weight').items())
	nx.draw(G2,pos,node_color='k',node_size=10, width=2, edge_color=weights, edge_cmap = plt.cm.rainbow)

	x_list = []; y_list = [] 
	mi = np.min(node_val_list); ma = np.max(node_val_list)
	for kk in range(0,len(nodes_list)):
		ix = nodes_list[kk]
		x = pos[ix][0]
		y = pos[ix][1]
		val = 1 - ((node_val_list[kk] - mi) /(ma - mi)*0.75 + 0.25)
		if node_val_list[kk] > .9:
			plt.plot(x,y,'.',color=(val,val,val),ms=10)
		if node_val_list[kk] > .75:
			plt.plot(x,y,'.',color=(val,val,val),ms=7.5)
		else:
			plt.plot(x,y,'.',color=(val,val,val),ms=5)	
	
	######################################################################################
	plt.savefig(out_analysis + '/' + folder_name + '_spatial_graph')
	if include_eps:
		plt.savefig(out_analysis + '/' + folder_name + '_spatial_graph.eps')
	plt.close()
	return
	
##########################################################################################
# time series plots and analysis 
##########################################################################################

##########################################################################################
def DTWDistance(s1, s2):
	"""Compute distance based on dynamic time warping (DTW)"""
	DTW={}
	for i in range(len(s1)):
		DTW[(i, -1)] = float('inf')
	for i in range(len(s2)):
		DTW[(-1, i)] = float('inf')
		
	DTW[(-1, -1)] = 0
	for i in range(len(s1)):
		for j in range(len(s2)):
			dist= (s1[i]-s2[j])**2
			DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

	return np.sqrt(DTW[len(s1)-1, len(s2)-1])
	
##########################################################################################
def cluster_timeseries_plot_dendrogram(folder_name,compute_dist_DTW,compute_dist_euclidean=False):
	"""Cluster timeseries and plot a dendrogram that shows the clustering."""
	external_folder_name = 'ALL_MOVIES_PROCESSED'
	if not os.path.exists(external_folder_name):
		os.makedirs(external_folder_name)

	if compute_dist_DTW == False and compute_dist_euclidean == False: load_dist_DTW = True
	else: load_dist_DTW = False 
	
	out_analysis = external_folder_name + '/' + folder_name + '/analysis'
	if not os.path.exists(external_folder_name + '/' + folder_name): os.makedirs(external_folder_name + '/' + folder_name)
	if not os.path.exists(out_analysis): os.makedirs(out_analysis)

	num_frames = len(glob.glob('ALL_MOVIES_MATRICES/' + folder_name + '_matrices/*.npy'))

	sarc_data_normalized_fname = 'ALL_MOVIES_PROCESSED/' + folder_name + '/timeseries/tracking_results_frames.txt'
	arr_frames = np.loadtxt(sarc_data_normalized_fname)
	sarc_data_normalized_fname = 'ALL_MOVIES_PROCESSED/' + folder_name + '/timeseries/tracking_results_leng.txt'
	arr_leng = np.loadtxt(sarc_data_normalized_fname)
	
	X = arr_leng

	if compute_dist_DTW:
		num_sarc = X.shape[0]
		dist_mat = np.zeros((num_sarc,num_sarc))
		for kk in range(0,num_sarc):
			for jj in range(kk+1,num_sarc):
				dist_mat[kk,jj] = DTWDistance(X[kk,:],X[jj,:])
		np.savetxt( 'ALL_MOVIES_PROCESSED/' + folder_name + '/analysis/dist_mat_DTW.txt',dist_mat)
		dist_mat = dist_mat + dist_mat.T
	elif load_dist_DTW:
		dist_mat = np.loadtxt('ALL_MOVIES_PROCESSED/' + folder_name + '/analysis/dist_mat_DTW.txt')
		dist_mat = dist_mat + dist_mat.T
	elif compute_dist_euclidean:
		Y = pdist(X, 'euclidean')
		dist_mat = squareform(Y)
	
	dist_v = squareform(dist_mat)
	Z = linkage(dist_v , method='ward', metric='euclidean')
	ll = leaves_list(Z)

	# --> plot dendrogram 
	plt.figure(figsize=(9,30),frameon=False)
	plt.subplot(1,2,1)
	# dendrogram
	dn1 = dendrogram(Z,orientation='left',color_threshold=0, above_threshold_color='k') #,truncate_mode='lastp')
	ordered = dn1['leaves'] #from bottom to top 

	if compute_dist_DTW or load_dist_DTW:
		np.savetxt('ALL_MOVIES_PROCESSED/' + folder_name + '/analysis/dendrogram_order_DTW.txt',np.asarray(ordered))
	else:
		np.savetxt('ALL_MOVIES_PROCESSED/' + folder_name + '/analysis/dendrogram_order_euc.txt',np.asarray(ordered))

	ax = plt.gca()
	ax.xaxis.set_visible(False)
	plt.subplot(1,2,2)
	ax = plt.gca()

	for kk in range(0,len(ordered)):
		ix = ordered[kk]
		col = (1-kk/len(ordered), kk/len(ordered) , 1- kk/len(ordered))
		plt.plot(X[ix,:] + kk*.3,c=col)
	
	plt.tight_layout()
	plt.ylim((-.4,kk*.3+.35))
	plt.axis('off')

	if compute_dist_DTW or load_dist_DTW:
		plt.savefig('ALL_MOVIES_PROCESSED/' + folder_name + '/analysis/dendrogram_DTW.pdf')
	else:
		plt.savefig('ALL_MOVIES_PROCESSED/' + folder_name + '/analysis/dendrogram_euclidean.pdf')

	return

##########################################################################################
def plot_normalized_tracked_timeseries(folder_name,include_eps=False):
	"""Create a plot of the normalized tracked time series."""
	external_folder_name = 'ALL_MOVIES_PROCESSED'
	if not os.path.exists(external_folder_name):
		os.makedirs(external_folder_name)

	out_analysis = external_folder_name + '/' + folder_name + '/analysis'
	if not os.path.exists(external_folder_name + '/' + folder_name): os.makedirs(external_folder_name + '/' + folder_name)
	if not os.path.exists(out_analysis): os.makedirs(out_analysis)

	num_frames = len(glob.glob('ALL_MOVIES_MATRICES/' + folder_name + '_matrices/*.npy'))

	sarc_data_normalized_fname = 'ALL_MOVIES_PROCESSED/' + folder_name + '/timeseries/tracking_results_leng.txt'
	all_normalized = np.loadtxt(sarc_data_normalized_fname)

	plt.figure()
	plt.plot(all_normalized.T,linewidth=.25)
	plt.plot(np.median(all_normalized.T,axis=1),'k-',linewidth=3,label='median curve')
	plt.plot(np.mean(all_normalized.T,axis=1),'--',color=(.5,.5,.5),linewidth=3,label='mean curve')
	plt.legend()
	plt.legend()
	plt.xlabel('frame')
	plt.ylabel('normalized length')
	plt.title('timeseries data, tracked and normalized, %i sarcomeres'%(all_normalized.shape[0]))
	plt.ylim((-.1,.1))
	plt.legend
	plt.tight_layout()
	plt.savefig('ALL_MOVIES_PROCESSED/' + folder_name + '/analysis/timeseries_tracked_normalized')
	if include_eps:
		plt.savefig('ALL_MOVIES_PROCESSED/' + folder_name + '/analysis/timeseries_tracked_normalized.eps')
	return

##########################################################################################
def plot_untracked_absolute_timeseries(folder_name,include_eps=False):
	"""Create a plot of the un-tracked absolute sarcomere lengths."""
	external_folder_name = 'ALL_MOVIES_PROCESSED'
	if not os.path.exists(external_folder_name):
		os.makedirs(external_folder_name)

	out_analysis = external_folder_name + '/' + folder_name + '/analysis'
	if not os.path.exists(external_folder_name + '/' + folder_name): os.makedirs(external_folder_name + '/' + folder_name)
	if not os.path.exists(out_analysis): os.makedirs(out_analysis)

	num_frames = len(glob.glob('ALL_MOVIES_MATRICES/' + folder_name + '_matrices/*.npy'))

	ALL_PIX_LEN = []; med = []; ix = []; num_sarc = [] 

	for frame in range(0,num_frames):
		if frame < 10: file_root = '/frame-000%i'%(frame)
		elif frame < 100: file_root = '/frame-00%i'%(frame)
		else: file_root = '/frame-0%i'%(frame)
	
		fname = external_folder_name + '/' + folder_name + '/segmented_sarc/' + file_root + '_sarc_data.txt'
		data = np.loadtxt(fname)
		pix_len = data[:,4]
		ALL_PIX_LEN.append(pix_len)
		med.append(np.median(pix_len))
		ix.append(frame+1)
		num_sarc.append(len(pix_len))
	
	# --> create a violin plot of everything 
	plt.figure(figsize=(12,6))
	plt.subplot(3,1,1)
	ax = plt.gca()
	ax.violinplot(ALL_PIX_LEN)
	plt.plot(ix,med,'ro',label='median')
	plt.legend()
	plt.xlabel('frame')
	plt.ylabel('sarc len in pixels')
	plt.title(folder_name + ' absolute sarcomere length untracked')
	plt.subplot(3,1,2)
	plt.plot(ix,med,'k-')
	plt.plot(ix,med,'ro',label='median')
	plt.legend()
	plt.xlabel('frame')
	plt.ylabel('sarc len in pixels')
	plt.subplot(3,1,3)
	plt.plot(ix,num_sarc,'k-')
	plt.plot(ix,num_sarc,'go')
	plt.xlabel('frame')
	plt.ylabel('# sarc segmented')
	plt.savefig( external_folder_name + '/' + folder_name + '/analysis/absolute_sarc_length_untracked')
	if include_eps:
		plt.savefig( external_folder_name + '/' + folder_name + '/analysis/absolute_sarc_length_untracked.eps')
	return 

##########################################################################################
def compute_timeseries_individual_parameters(folder_name,include_eps=False):
	"""Compute and save timeseries time constants (contraction time, relaxation time, flat time, period, offset, etc.)."""
	external_folder_name = 'ALL_MOVIES_PROCESSED'
	if not os.path.exists(external_folder_name):
		os.makedirs(external_folder_name)

	out_analysis = external_folder_name + '/' + folder_name + '/analysis'
	if not os.path.exists(external_folder_name + '/' + folder_name): os.makedirs(external_folder_name + '/' + folder_name)
	if not os.path.exists(out_analysis): os.makedirs(out_analysis)

	num_frames = len(glob.glob('ALL_MOVIES_MATRICES/' + folder_name + '_matrices/*.npy'))

	input_distance = 10; input_width = 5 # <-- might need to adjust?  

	sarc_data_normalized_fname = external_folder_name + '/' + folder_name + '/timeseries/tracking_results_frames.txt'
	arr_frames = np.loadtxt(sarc_data_normalized_fname)
	sarc_data_normalized_fname = external_folder_name + '/' + folder_name + '/timeseries/tracking_results_leng.txt'
	arr_leng = np.loadtxt(sarc_data_normalized_fname)
	sarc_data_fname = external_folder_name + '/' + folder_name + '/timeseries/tracking_results_leng_NOT_NORMALIZED.txt'
	arr_leng_not_normalized = np.loadtxt(sarc_data_fname)

	pix_leng_median = []; pix_leng_mean = []; pix_leng_min = []; pix_leng_max = []; perc_sarc_short = [] 
	fra_mean_contract_time = []; fra_mean_relax_time = []; fra_mean_flat_time = []; fra_mean_period = []; fra_to_first = [] 
	idx_sarc = []; num_peak_all = [] 

	for zz in range(0,arr_frames.shape[0]):
		idx_sarc.append(zz)
		x = arr_frames[zz,:]
		data_pixels = arr_leng_not_normalized[zz,:]
		data = arr_leng[zz,:]
		data_med = signal.medfilt(data,5) # optional median filter 
		deriv = np.gradient(data,x)
		# go through and group into category by derivative 
		count_C = 0; count_R = 0; count_F = 0
		thresh_flat = 0.005*(np.max(data_med) - np.min(data_med))/0.2
		for kk in range(0,x.shape[0]):
			if deriv[kk] > thresh_flat: count_R += 1 
			elif deriv[kk] < -1.0*thresh_flat: count_C += 1
			else: count_F += 1 
	
		# detect peaks and valleys 
		th = .00; di = input_distance; wi = input_width # parameters
		# distance Required minimal horizontal distance (>= 1) in samples between neighbouring peaks. Smaller peaks are removed first until the condition is fulfilled for all remaining peaks.
		#widthnumber or ndarray or sequence, optional Required width of peaks in samples. Either a number, None, an array matching x or a 2-element sequence of the former. The first element is always interpreted as the minimal and the second, if supplied, as the maximal required width.
	
		peaks_U, _ = find_peaks(data_med,threshold=th,distance=di,width=wi)
		peaks_L, _ = find_peaks(-1.0*data_med,threshold=th,distance=di,width=wi)
	
		#num_peaks = 0.5 * peaks_U.shape[0] + 0.5 * peaks_L.shape[0]
		#num_peaks = peaks_L.shape[0]
		num_peaks = 0
		for kk in range(0,peaks_L.shape[0]):
			if data_med[peaks_L[kk]] < np.mean(data_med) - thresh_flat:
				num_peaks += 1
		if num_peaks == 0: num_peaks = 999999
		mean_C = count_C / num_peaks
		mean_R = count_R / num_peaks 
		mean_F = count_F / num_peaks 
	
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
		#		save everything                                                             #
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
		pix_leng_median.append(np.median(data_pixels))
		pix_leng_mean.append(np.mean(data_pixels))
		mi = np.min(data_pixels); pix_leng_min.append(mi)
		ma = np.max(data_pixels); pix_leng_max.append(ma)
		perc_sarc_short.append(  (ma - mi)/(ma) * 100 )
		fra_mean_contract_time.append(mean_C)
		fra_mean_relax_time.append(mean_R)
		fra_mean_flat_time.append(mean_F)
		fra_mean_period.append(x.shape[0] / num_peaks)
		if peaks_L.shape[0] > 0:
			fra_to_first.append(peaks_L[0])
		else:
			fra_to_first.append(0)
		num_peak_all.append(num_peaks)	
	
	### --> plot parameters
	plt.figure(figsize=(7,7))
	plt.subplot(2,2,1)
	plt.hist(fra_mean_contract_time)
	plt.plot([np.median(fra_mean_contract_time),np.median(fra_mean_contract_time)],[0,10],'r--')
	plt.xlabel('frames')
	plt.title('median_contract: %.2f'%(np.median(fra_mean_contract_time)))
	plt.tight_layout()
	plt.subplot(2,2,2)
	plt.hist(fra_mean_relax_time)
	plt.plot([np.median(fra_mean_relax_time),np.median(fra_mean_relax_time)],[0,10],'r--')
	plt.xlabel('frames')
	plt.title('median_relax: %.2f'%(np.median(fra_mean_relax_time)))
	plt.tight_layout()
	plt.subplot(2,2,3)
	plt.hist(fra_mean_flat_time)
	plt.plot([np.median(fra_mean_flat_time),np.median(fra_mean_flat_time)],[0,10],'r--')
	plt.xlabel('frames')
	plt.title('median_flat: %.2f'%(np.median(fra_mean_flat_time)))
	plt.tight_layout()
	plt.subplot(2,2,4)
	plt.hist(fra_mean_period)
	plt.plot([np.median(fra_mean_period),np.median(fra_mean_period)],[0,10],'r--')
	plt.xlabel('frames')
	plt.title('median_period: %.2f'%(np.median(fra_mean_period)))
	plt.tight_layout()
	plt.savefig(out_analysis + '/histogram_time_constants')
	if include_eps:
		plt.savefig(out_analysis + '/histogram_time_constants.eps')

	num_sarc = len(idx_sarc) 
	arr = np.zeros((num_sarc,12))
	arr[:,0] = np.asarray(idx_sarc) 
	arr[:,1] = np.asarray(pix_leng_median)
	arr[:,2] = np.asarray(pix_leng_mean)
	arr[:,3] = np.asarray(pix_leng_min)
	arr[:,4] = np.asarray(pix_leng_max)
	arr[:,5] = np.asarray(perc_sarc_short)
	arr[:,6] = np.asarray(fra_mean_contract_time)
	arr[:,7] = np.asarray(fra_mean_relax_time)
	arr[:,8] = np.asarray(fra_mean_flat_time)
	arr[:,9] = np.asarray(fra_mean_period)
	arr[:,10] = np.asarray(fra_to_first)
	arr[:,11] = np.asarray(num_peak_all)
	np.savetxt(out_analysis + '/timeseries_parameters_info.txt', arr)

	# --> save as excel spreadsheet 
	writer = pd.ExcelWriter(out_analysis + '/timeseries_parameters_info.xlsx', engine='xlsxwriter')
	all_col = ['idx', 'pix_leng_median', 'pix_leng_mean', 'pix_leng_min', 'pix_leng_max', 'perc_sarc_short', 'frames_mean_contract', 'frames_mean_relax', 'frames_mean_flat', 'frames_mean_period', 'frames_to_first', 'num_peaks']
	df = pd.DataFrame(np.asarray(arr), columns=all_col)
	df.to_excel(writer, sheet_name='summary_stats')
	arr = arr_leng
	df2 = pd.DataFrame(np.asarray(arr))
	df2.to_excel(writer, sheet_name='full_time_series', columns = arr_frames[0,:])
	writer.save()	
	return

##########################################################################################
def sample(mu_track,num_track,vals_all):
	"""Sample mu from the total population -- match #tracked."""
	num_run = 1000
	mu_samp = []
	for jj in range(0,num_run):
		ix = []
		for kk in range(0,num_track):
			ix.append(random.randint(0,len(vals_all)-1))
			
		samp = vals_all[ix]
		mu_samp.append(mu_track - np.mean(samp))
	return mu_samp

##########################################################################################
def compute_mu_ang(ang_list):
	"""Compute the mean of an angle."""
	x_total = 0
	y_total = 0
	for kk in range(0,len(ang_list)):
		ang = ang_list[kk]
		x_total += np.cos(ang)
		y_total += np.sin(ang)
		
	x_mean = x_total / len(ang_list)
	y_mean = y_total / len(ang_list)
	ang = np.arctan2(y_mean, x_mean)
	r = np.sqrt(x_mean**2.0 + y_mean**2.0)
	return ang, r

##########################################################################################
def sample_ang(mu_track_ang, mu_track_r,num_track,vals_all):
	"""Sample angle from the total population -- match #tracked."""
	num_run = 1000
	mu_samp_ang = []
	mu_samp_r = [] 
	for jj in range(0,num_run):
		ix = []
		for kk in range(0,num_track):
			ix.append(random.randint(0,len(vals_all)-1))
			
		samp = vals_all[ix]
		ang, r = compute_mu_ang(samp)
		mu_samp_ang.append(mu_track_ang - ang)
		mu_samp_r.append(mu_track_r - r)
	return mu_samp_ang, mu_samp_r
	
##########################################################################################
def compare_tracked_untracked(folder_name,include_eps=False):
	"""Compare the tracked and untracked populations by random sampling the untracked population."""
	external_folder_name = 'ALL_MOVIES_PROCESSED'
	if not os.path.exists(external_folder_name):
		os.makedirs(external_folder_name)

	out_analysis = external_folder_name + '/' + folder_name + '/analysis'
	if not os.path.exists(external_folder_name + '/' + folder_name): os.makedirs(external_folder_name + '/' + folder_name)
	if not os.path.exists(out_analysis): os.makedirs(out_analysis)

	num_frames = len(glob.glob('ALL_MOVIES_MATRICES/' + folder_name + '_matrices/*.npy'))

	ALL_PIX_LEN = []; ALL_PIX_WID = []; ALL_PIX_ANG = [] 
	med = []; ix = []; num_sarc = [] 

	for frame in range(0,num_frames):
		if frame < 10: file_root = '/frame-000%i'%(frame)
		elif frame < 100: file_root = '/frame-00%i'%(frame)
		else: file_root = '/frame-0%i'%(frame)
		fname = external_folder_name + '/' + folder_name + '/segmented_sarc/' + file_root + '_sarc_data.txt'
		data = np.loadtxt(fname)
		pix_len = data[:,4]; pix_wid = data[:,5]; pix_ang = data[:,6]
		ALL_PIX_LEN.append(pix_len); ALL_PIX_WID.append(pix_wid); ALL_PIX_ANG.append(pix_ang)
		med.append(np.median(pix_len)); ix.append(frame+1); num_sarc.append(len(pix_len))

	# --> import data 
	sarc_data_fname = external_folder_name + '/' + folder_name + '/timeseries/tracking_results_leng_NOT_NORMALIZED.txt'
	tracked_leng = np.loadtxt(sarc_data_fname)
	sarc_data_fname = external_folder_name + '/' + folder_name + '/timeseries/tracking_results_wid.txt'
	tracked_wid = np.loadtxt(sarc_data_fname)
	sarc_data_fname = external_folder_name + '/' + folder_name + '/timeseries/tracking_results_ang.txt'
	tracked_ang = np.loadtxt(sarc_data_fname)

	# --> compute the mean number NOT tracked 
	num_not = 0
	for kk in range(0,len(ALL_PIX_LEN)): num_not += len(ALL_PIX_LEN[kk])
	num_not = num_not / len(ALL_PIX_LEN); num_tracked = tracked_leng.shape[0]

	# --> sample the length from the tracked population 
	mu_samp_ALL_len = [] 
	for frame_num in range(0,num_frames):
		len_all = ALL_PIX_LEN[frame_num]
		len_tracked = list(tracked_leng[:,frame_num])
		mu_track = np.mean(len_tracked)
		num_track = len(len_tracked)
		vals_all = len_all
		mu_samp = sample(mu_track, num_track, vals_all)
		mu_samp_ALL_len.append(mu_samp)

	plt.figure(figsize=(25,5))
	plt.boxplot(mu_samp_ALL_len)
	plt.plot([0,num_frames],[-.5,-.5],'k--')
	plt.plot([0,num_frames],[.5,.5],'k--')
	plt.title('comparison of length in pixels, approx %i untracked, %i tracked'%(num_not,num_tracked))
	plt.xlabel('frame number')
	plt.ylabel(r'$\mu_{track}-\mu_{all}$')
	plt.savefig(out_analysis + '/length_compare_box_plots')
	if include_eps:
		plt.savefig(out_analysis + '/length_compare_box_plots.eps')

	# --> sample the width from the tracked population 
	mu_samp_ALL_wid = [] 
	for frame_num in range(0,num_frames):
		wid_all = ALL_PIX_WID[frame_num]
		wid_tracked = list(tracked_wid[:,frame_num])
		mu_track = np.mean(wid_tracked)
		num_track = len(wid_tracked)
		vals_all = wid_all
		mu_samp = sample(mu_track, num_track, vals_all)
		mu_samp_ALL_wid.append(mu_samp)

	plt.figure(figsize=(25,5))
	plt.boxplot(mu_samp_ALL_wid)
	plt.plot([0,num_frames],[-.5,-.5],'k--')
	plt.plot([0,num_frames],[.5,.5],'k--')
	plt.title('comparison of width in pixels, approx %i untracked, %i tracked'%(num_not,num_tracked))
	plt.xlabel('frame number')
	plt.ylabel(r'$\mu_{track}-\mu_{all}$')
	plt.savefig(out_analysis + '/width_compare_box_plots')
	if include_eps:
		plt.savefig(out_analysis + '/width_compare_box_plots.eps')

	# --> sample the angle from the tracked population 
	mu_samp_ALL_ang = []; mu_samp_ALL_rad = []
	for frame_num in range(0,num_frames):
		ang_all = ALL_PIX_ANG[frame_num]
		ang_tracked = list(tracked_ang[:,frame_num])
		mu_track_ang, mu_track_r = compute_mu_ang(ang_tracked)
		num_track = len(ang_tracked)
		vals_all = ang_all
		mu_samp_ang, mu_samp_r = sample_ang(mu_track_ang, mu_track_r,num_track,vals_all)
		mu_samp_ALL_ang.append(mu_samp_ang)
		mu_samp_ALL_rad.append(mu_samp_r)
	
	plt.figure(figsize=(25,10))
	plt.subplot(2,1,1)
	plt.boxplot(mu_samp_ALL_ang)
	plt.plot([0,num_frames],[-1*np.pi/8,-1*np.pi/8],'k--')
	plt.plot([0,num_frames],[np.pi/8,np.pi/8],'k--')
	plt.title('comparison of angle in radians, approx %i untracked, %i tracked'%(num_not,num_tracked))
	plt.xlabel('frame number')
	plt.ylabel(r'$\mu_{track}-\mu_{all}$')
	plt.subplot(2,1,2)
	plt.boxplot(mu_samp_ALL_rad)
	plt.plot([0,num_frames],[0,0],'r--',label='uniform')
	plt.plot([0,num_frames],[1,1],'k--',label='oriented')
	plt.title('comparison of angle radius in pixels, approx %i untracked, %i tracked'%(num_not,num_tracked))
	plt.xlabel('frame number')
	plt.ylabel(r'$\mu_{track}-\mu_{all}$')
	plt.legend()
	plt.savefig(out_analysis + '/angle_compare_box_plots')
	if include_eps:
		plt.savefig(out_analysis + '/angle_compare_box_plots.eps')
	return 

##########################################################################################

##########################################################################################
# compute time series correlations  -- on graph distance and euclidean distance 
##########################################################################################

##########################################################################################
def compute_cross_correlation(sig1, sig2):
	"""Compute the normalized cross correlation between two signals."""
	sig1_norm = (sig1 - np.mean(sig1)) / (np.std(sig1) * sig1.shape[0])
	sig2_norm = (sig2 - np.mean(sig2)) / (np.std(sig2))		
	val = np.correlate(sig1_norm,sig2_norm)
	return val

##########################################################################################
def dist_val2(subgraphs,node_1,node_2):
	"""Compute the network distance between two nodes."""
	for sg in subgraphs:
		node_1_in = sg.has_node(node_1)
		node_2_in = sg.has_node(node_2)
		if node_1_in and node_2_in:
			dist = nx.shortest_path_length(sg,source=node_1,target=node_2)
			return dist 
	return 99999

##########################################################################################
def get_euclid_dist_from_avg_pos(x_vec_1,y_vec_1,x_vec_2,y_vec_2):
	"""Return the average euclidian distance between two sarcomeres."""
	dist_vec = (( x_vec_1 - x_vec_2 )**2.0 + ( y_vec_1 - y_vec_2 )**2.0)**(1.0/2.0)
	return np.mean(dist_vec)

##########################################################################################
def preliminary_spatial_temporal_correlation_info(folder_name,compute_network_distances=True,include_eps=False):
	"""Perform a preliminary analysis of spatial/temporal correlation."""
	num_frames = len(glob.glob('ALL_MOVIES_MATRICES/' + folder_name + '_matrices/*.npy'))

	external_folder_name = 'ALL_MOVIES_PROCESSED'
	if not os.path.exists(external_folder_name):
		os.makedirs(external_folder_name)

	out_analysis = external_folder_name + '/' + folder_name + '/analysis'
	if not os.path.exists(external_folder_name + '/' + folder_name): os.makedirs(external_folder_name + '/' + folder_name)
	if not os.path.exists(out_analysis): os.makedirs(out_analysis)

	# --> import sarcomere 
	sarc_data_normalized_fname = external_folder_name + '/' + folder_name + '/timeseries/tracking_results_frames.txt'
	arr_frames = np.loadtxt(sarc_data_normalized_fname)
	sarc_data_normalized_fname = external_folder_name + '/' + folder_name + '/timeseries/tracking_results_leng.txt'
	arr_leng = np.loadtxt(sarc_data_normalized_fname)
	sarc_data_fname = external_folder_name + '/' + folder_name + '/timeseries/tracking_results_leng_NOT_NORMALIZED.txt'
	arr_leng_not_normalized = np.loadtxt(sarc_data_fname)

	# --> import raw image 
	raw_img = np.load('ALL_MOVIES_MATRICES/' + folder_name + '_matrices/frame-0000.npy')

	# --> import graph
	out_graph =  external_folder_name + '/' + folder_name + '/graph'
	with open(out_graph + '/graph.pkl', 'rb') as f: G = pickle.load(f)
	out_graph = folder_name + '/graph/basic_graph.png'
	graph = plt.imread(external_folder_name + '/' + folder_name + '/graph/basic_graph.png')

	# --> import sarcomere info
	sarc_data_fname = 'ALL_MOVIES_PROCESSED/' + folder_name + '/tracking_results/tracking_results_sarcomeres.txt'
	sarc_data = np.loadtxt(sarc_data_fname)

	# --> import sarcomere position info
	sarc_x_pos_data_fname = 'ALL_MOVIES_PROCESSED/' + folder_name + '/timeseries/tracking_results_x_pos.txt'
	sarc_x_pos_data = np.loadtxt(sarc_x_pos_data_fname )
	sarc_y_pos_data_fname = 'ALL_MOVIES_PROCESSED/' + folder_name + '/timeseries/tracking_results_y_pos.txt'
	sarc_y_pos_data = np.loadtxt(sarc_y_pos_data_fname )

	# --> import z-disc data 
	zdisc_data_fname = 'ALL_MOVIES_PROCESSED/' + folder_name + '/tracking_results/tracking_results_zdisks.txt'
	zdisc_data = np.loadtxt(zdisc_data_fname)
	particle = zdisc_data[:,2]

	# --> import index information 
	sarc_idx_fname = 'ALL_MOVIES_PROCESSED/' + folder_name + '/timeseries/tracking_results_sarc_idx_above_thresh.txt'
	sarc_idx = np.loadtxt(sarc_idx_fname)

	all_frames = sarc_data[:,0]; all_particles = sarc_data[:,2]
	all_z_1 = sarc_data[:,5]; all_z_2 = sarc_data[:,6]
	unique_particles = np.unique(all_particles).astype('int')
	organized_data_z1 = np.zeros((unique_particles.shape[0],num_frames))
	organized_data_z2 = np.zeros((unique_particles.shape[0],num_frames))

	for kk in range(0,sarc_data.shape[0]):
		part = int(all_particles[kk])
		frame = int(all_frames[kk])
		idx_in_frame = np.where(zdisc_data[:,0] == frame)
		disc_data = zdisc_data[idx_in_frame[0],:]
		part_idx = np.argmin(np.abs(unique_particles - part))
		ZLID1 = int(all_z_1[kk])
		ZLID2 = int(all_z_2[kk])
		orig_disc_idx = disc_data[:,1].astype(int)
		check = np.where(orig_disc_idx == ZLID1)[0]
		if check.shape[0] == 0:
			continue
		else:
			ZGID1_idx = check[0]
			ZGID1 = int(disc_data[ZGID1_idx,2])
		check = np.where(orig_disc_idx == ZLID2)[0]
		if check.shape[0] == 0:
			continue
		else:
			ZGID2_idx = check[0]
			ZGID2 = int(disc_data[ZGID2_idx,2])
	
		organized_data_z1[part_idx,frame] = ZGID1
		organized_data_z2[part_idx,frame] = ZGID2

	# --> for each sarcomere identify which z discs it belongs to 
	Z_disc_1 = []; Z_disc_2 = [] 
	
	for kk in range(0,sarc_idx.shape[0]):
		idx = int(sarc_idx[kk])
		z_idx_1 = organized_data_z1[idx,:]
		if np.sum(z_idx_1) == 0:
			z_idx_1 = z_idx_1
		else:
			z_idx_1 = z_idx_1[z_idx_1>0]
		z_idx_2 = organized_data_z2[idx,:]
		if np.sum(z_idx_2) == 0:
			z_idx_2 = z_idx_2
		else:
			z_idx_2 = z_idx_2[z_idx_2>0]
		Z_disc_1.append(int(scipy.stats.mode(z_idx_1)[0][0]))
		Z_disc_2.append(int(scipy.stats.mode(z_idx_2)[0][0]))

	# get graph distances and correlation scores 
	graph_dist_all = []; corr_score_all = []; euclid_dist_all = [] 
	if compute_network_distances:
		for jj in range(0,sarc_idx.shape[0]):
			for kk in range(jj+1,sarc_idx.shape[0]):
				jj_idx = [Z_disc_1[jj], Z_disc_2[jj]]
				kk_idx = [Z_disc_1[kk], Z_disc_2[kk]]
				dist_all_combos = [] 
				for j in jj_idx:
					for k in kk_idx:
						subgraphs = (G.subgraph(c).copy() for c in nx.connected_components(G))
						dist = dist_val2(subgraphs,j,k)
						dist_all_combos.append(dist)
			
				sig1 = arr_leng[jj,:]
				sig2 = arr_leng[kk,:]
				corr_score = compute_cross_correlation(sig1, sig2)
				corr_score_all.append(corr_score)	
				graph_dist_all.append( np.min(dist_all_combos) )
				x_vec_1 = sarc_x_pos_data[jj,:]; y_vec_1 = sarc_y_pos_data[jj,:]
				x_vec_2 = sarc_x_pos_data[kk,:]; y_vec_2 = sarc_y_pos_data[kk,:]
				euclid_dist = get_euclid_dist_from_avg_pos(x_vec_1,y_vec_1,x_vec_2,y_vec_2)
				euclid_dist_all.append(euclid_dist)

		np.savetxt(out_analysis + '/graph_dist_all.txt',np.asarray(graph_dist_all))
		np.savetxt(out_analysis + '/euclid_dist_all.txt',np.asarray(euclid_dist_all))
		np.savetxt(out_analysis + '/corr_score_all.txt',np.asarray(corr_score_all))
	else:
		graph_dist_all = np.loadtxt(out_analysis + '/graph_dist_all.txt')
		euclid_dist_all = np.loadtxt(out_analysis + '/euclid_dist_all.txt')
		corr_score_all = np.loadtxt(out_analysis + '/corr_score_all.txt')


	graph_dist_all = np.asarray(graph_dist_all).astype('int')
	euclid_dist_all = np.asarray(euclid_dist_all)
	corr_score_all = np.asarray(corr_score_all)

	########## --> make plot 
	plt.figure(figsize=(30,4))
	# raw image
	plt.subplot(1,5,1)
	plt.imshow(raw_img)
	ax = plt.gca()
	ax.set_xticks([]); ax.set_yticks([])
	plt.title(folder_name + ' raw image')
	plt.tight_layout()
	# graph
	plt.subplot(1,5,2)
	plt.imshow(graph)
	ax = plt.gca()
	ax.set_xticks([]); ax.set_yticks([])
	plt.title(folder_name + ' graph')
	plt.tight_layout()
	# histogram
	plt.subplot(1,5,3)
	n, bins, patches = plt.hist(corr_score_all,range=(-1,1),rwidth=.8,color=(.5,.5,.5))
	plt.xlim((-1.1,1.1))
	plt.xlabel('normalized cross-correlation')
	plt.title('timeseries comparison')
	ma = np.max(n)
	plt.plot([0,0],[0,ma],'g--',label='no correlation')
	plt.plot([np.median(corr_score_all),np.median(corr_score_all)],[0,ma],'b-',label='median: %.2f'%(np.median(corr_score_all)))
	plt.legend()
	plt.tight_layout()
	# euclidean
	plt.subplot(1,5,4)
	x_coord = []; y_coord = []; num_in_bin = []
	for kk in range(0,5):
		ix_1 = euclid_dist_all > kk*20 
		ix_2 = euclid_dist_all < (kk +1)*20
		ix = [] 
		for jj in range(0,np.asarray(euclid_dist_all).shape[0]):
			if ix_1[jj] == True and ix_2[jj] == True:
				ix.append(jj)
		x_coord.append(kk*20 + 5)
		me = np.mean(corr_score_all[ix])
		num_in_bin.append(len(corr_score_all[ix]))
		y_coord.append(me)

	plt.plot(x_coord,y_coord,'.',color=(1.0,.5,.5),markersize=20,label='binned means')
	maxi = np.max(x_coord)
	plt.plot([0,maxi],[0,0],'g--',label='no correlation')
	mean_all = np.mean(corr_score_all)
	plt.plot([0,maxi],[mean_all,mean_all],'b-',label='mean all: %.2f'%(mean_all))

	plt.xlabel('timeseries comparison wrt euclidian distance (pixels)')
	plt.ylabel('normalized cross-correlation')
	plt.grid(True)
	plt.title('timeseries comparison wrt distance')
	plt.legend()
	plt.ylim((-1.05,1.05))
	plt.tight_layout()
	# network
	plt.subplot(1,5,5)
	dist_bins = []
	for kk in range(0,5): dist_bins.append(kk)
	x_coord = []; y_coord = []; num_in_bin = [ ]
	for di in dist_bins:
		ix = graph_dist_all == int(di)
		corr_score = corr_score_all[ix]
		if corr_score.shape[0] > 3:
			x_coord.append(di)		
			y_coord.append(np.mean(corr_score))
			num_in_bin.append(len(corr_score))

	ix = graph_dist_all < 9999
	corr_score = corr_score_all[ix]
	mean_connected = np.mean(corr_score)
	mean_all = np.mean(corr_score_all)

	plt.plot(x_coord,y_coord,'.',color=(1.0,.5,.5),markersize=20,label='binned means')
	maxi = np.max(dist_bins)
	plt.plot([0,maxi],[mean_connected, mean_connected],'r--',label='mean connected: %.2f'%(mean_connected))
	plt.plot([0,maxi],[0,0],'g--',label='no correlation')
	plt.plot([0,maxi],[mean_all,mean_all],'b-',label='mean all: %.2f'%(mean_all))

	plt.legend(loc=4)
	plt.xlabel('distance along network')
	plt.ylabel('normalized cross-correlation')
	plt.grid(True)
	plt.title('timeseries comparison wrt network distance')
	plt.ylim((-1.05,1.05))
	plt.tight_layout()

	plt.savefig(out_analysis + '/preliminary_spatial_analysis')
	if include_eps:
		plt.savefig(out_analysis + '/preliminary_spatial_analysis.eps')
	return 


##########################################################################################
# compute F 
##########################################################################################

##########################################################################################
def compute_F_whole_movie(folder_name,include_eps=False):
	"""Compute and return the average deformation gradient for the whole movie."""
	# set up folders
	external_folder_name = 'ALL_MOVIES_PROCESSED'
	if not os.path.exists(external_folder_name):
		os.makedirs(external_folder_name)

	out_analysis = external_folder_name + '/' + folder_name + '/analysis'
	if not os.path.exists(external_folder_name + '/' + folder_name): os.makedirs(external_folder_name + '/' + folder_name)
	if not os.path.exists(out_analysis): os.makedirs(out_analysis)

	# compute Lambda from x_pos and y_pos 
	x_pos = np.loadtxt('ALL_MOVIES_PROCESSED/' + folder_name + '/timeseries/tracking_results_x_pos.txt')
	y_pos = np.loadtxt('ALL_MOVIES_PROCESSED/' + folder_name + '/timeseries/tracking_results_y_pos.txt')

	num_sarc = x_pos.shape[0]
	num_time = x_pos.shape[1]
	num_vec = int((num_sarc * num_sarc - num_sarc) / 2.0)

	Lambda_list = []
	for tt in range(0,num_time):
		Lambda = np.zeros((2,num_vec))
		ix = 0
		for kk in range(0,num_sarc):
			for jj in range(kk+1,num_sarc):
				x_vec = x_pos[kk,tt] - x_pos[jj,tt]
				y_vec = y_pos[kk,tt] - y_pos[jj,tt]
				Lambda[0,ix] = x_vec
				Lambda[1,ix] = y_vec 
				ix += 1 

		Lambda_list.append(Lambda)

	F_list = []; F11_list = []; F22_list = []; F12_list = []; F21_list = []
	J_list = []  
	for tt in range(0,num_time):
		Lambda_0 = Lambda_list[0]
		Lambda_t = Lambda_list[tt]
		term_1 = np.dot( Lambda_t , np.transpose(Lambda_0) )
		term_2 = np.linalg.inv( np.dot( Lambda_0 , np.transpose(Lambda_0) ) )
		F = np.dot(term_1 , term_2)
		F_vec = [F[0,0],F[0,1],F[1,0],F[1,1]]
		F_list.append(F_vec)
		F11_list.append(F[0,0] - 1.0)
		F22_list.append(F[1,1] - 1.0)
		F12_list.append(F[0,1])
		F21_list.append(F[1,0])
		J_list.append(F[0,0]*F[1,1] - F[0,1]*F[1,0])

	np.savetxt(out_analysis + '/recovered_F.txt',np.asarray(F_list))
	plt.figure(figsize=(10,5))
	plt.subplot(1,2,1)
	plt.plot(F11_list,'r--',linewidth=5, label='F11 recovered')
	plt.plot(F22_list,'g--',linewidth=4, label='F22 recovered')
	plt.plot(F12_list,'c:',label='F12 recovered')
	plt.plot(F21_list,'b:',label='F21 recovered')
	plt.legend()
	plt.title('recovered deformation gradient')
	plt.xlabel('frames');
	plt.subplot(1,2,2)
	plt.plot(J_list,'k-',label='Jacobian')
	plt.xlabel('frames');
	plt.legend()
	plt.title('det of deformation gradient')
	plt.savefig(out_analysis + '/recovered_F_plot')
	if include_eps:
		plt.savefig(out_analysis + '/recovered_F_plot.eps')
	return

##########################################################################################
def adjust_F_if_movie_starts_not_contracted(folder_name,include_eps=False):
	"""Adjust and return the average deformation gradient for the whole movie -- useful if first frame is not the relaxed state."""
	external_folder_name = 'ALL_MOVIES_PROCESSED'
	if not os.path.exists(external_folder_name):
		os.makedirs(external_folder_name)

	out_analysis = external_folder_name + '/' + folder_name + '/analysis'
	if not os.path.exists(external_folder_name + '/' + folder_name): os.makedirs(external_folder_name + '/' + folder_name)
	if not os.path.exists(out_analysis): os.makedirs(out_analysis)
	
	F_list = np.loadtxt(out_analysis + '/recovered_F.txt')
	J_list = [] #F_vec = [F[0,0],F[0,1],F[1,0],F[1,1]]
	for kk in range(0,F_list.shape[0]):
		F00 = F_list[kk,0]; F01 = F_list[kk,1]; F10 = F_list[kk,2]; F11 = F_list[kk,3]
		J_list.append(F00*F11 - F01*F10)
	
	arg_max = np.argmax(J_list)
	
	# compute Lambda from x_pos and y_pos 
	x_pos = np.loadtxt('ALL_MOVIES_PROCESSED/' + folder_name + '/timeseries/tracking_results_x_pos.txt')
	y_pos = np.loadtxt('ALL_MOVIES_PROCESSED/' + folder_name + '/timeseries/tracking_results_y_pos.txt')

	num_sarc = x_pos.shape[0]
	num_time = x_pos.shape[1]
	num_vec = int((num_sarc * num_sarc - num_sarc) / 2.0)

	Lambda_list = []
	for tt in range(0,num_time):
		Lambda = np.zeros((2,num_vec))
		ix = 0
		for kk in range(0,num_sarc):
			for jj in range(kk+1,num_sarc):
				x_vec = x_pos[kk,tt] - x_pos[jj,tt]
				y_vec = y_pos[kk,tt] - y_pos[jj,tt]
				Lambda[0,ix] = x_vec
				Lambda[1,ix] = y_vec 
				ix += 1 

		Lambda_list.append(Lambda)

	F_list = []; F11_list = []; F22_list = []; F12_list = []; F21_list = []
	J_list = []  
	for tt in range(0,num_time):
		Lambda_0 = Lambda_list[arg_max]
		Lambda_t = Lambda_list[tt]
		term_1 = np.dot( Lambda_t , np.transpose(Lambda_0) )
		term_2 = np.linalg.inv( np.dot( Lambda_0 , np.transpose(Lambda_0) ) )
		F = np.dot(term_1 , term_2)
		F_vec = [F[0,0],F[0,1],F[1,0],F[1,1]]
		F_list.append(F_vec)
		F11_list.append(F[0,0] - 1.0)
		F22_list.append(F[1,1] - 1.0)
		F12_list.append(F[0,1])
		F21_list.append(F[1,0])
		J_list.append(F[0,0]*F[1,1] - F[0,1]*F[1,0])

	np.savetxt(out_analysis + '/recovered_F.txt',np.asarray(F_list))
	plt.figure(figsize=(10,5))
	plt.subplot(1,2,1)
	plt.plot(F11_list,'r--',linewidth=5, label='F11 recovered')
	plt.plot(F22_list,'g--',linewidth=4, label='F22 recovered')
	plt.plot(F12_list,'c:',label='F12 recovered')
	plt.plot(F21_list,'b:',label='F21 recovered')
	plt.legend()
	plt.title('recovered deformation gradient')
	plt.xlabel('frames');
	plt.subplot(1,2,2)
	plt.plot(J_list,'k-',label='Jacobian')
	plt.xlabel('frames');
	plt.legend()
	plt.title('det of deformation gradient')
	plt.savefig(out_analysis + '/recovered_F_plot')
	if include_eps:
		plt.savefig(out_analysis + '/recovered_F_plot.eps')
	return

##########################################################################################
def analyze_J_full_movie(folder_name,include_eps=False):
	"""Analyze the Jacobian -- report timeseries parmeters. Must first run compute_F_whole_movie()."""
	# set up folders
	external_folder_name = 'ALL_MOVIES_PROCESSED'
	if not os.path.exists(external_folder_name):
		os.makedirs(external_folder_name)

	out_analysis = external_folder_name + '/' + folder_name + '/analysis'
	if not os.path.exists(external_folder_name + '/' + folder_name): os.makedirs(external_folder_name + '/' + folder_name)
	if not os.path.exists(out_analysis): os.makedirs(out_analysis)
	
	# import the deformation gradient. 
	F_list = np.loadtxt(out_analysis + '/recovered_F.txt')
	num_frames = F_list.shape[0]; x = [] 
	J_list = []
	for kk in range(0,num_frames):
		F00 = F_list[kk,0]; F01 = F_list[kk,1]; F10 = F_list[kk,2]; F11 = F_list[kk,3]
		J_list.append(F00*F11 - F01*F10)
		x.append(kk)
	
	J_list = np.asarray(J_list)
	x = np.asarray(x)
	
	# compute the parameters of the timeseries
	plt.figure(figsize=(4,4))
	plt.plot(J_list,'k-')
	
	data = J_list
	data_med = signal.medfilt(data,5)
	deriv = np.gradient(data,x)
	count_C = 0; count_R = 0; count_F = 0
	thresh_flat = 0.01*(np.max(J_list) - np.min(J_list))
	
	pix_leng_median = []; pix_leng_mean = []; pix_leng_min = []; pix_leng_max = []; perc_sarc_short = [] 
	fra_mean_contract_time = []; fra_mean_relax_time = []; fra_mean_flat_time = []; fra_mean_period = []; fra_to_first = [] 
	idx_sarc = []; num_peak_all = [] 
	for kk in range(0,x.shape[0]):
		if deriv[kk] > thresh_flat: 
			count_R += 1 
			plt.plot(x[kk],J_list[kk],'o',color=(.5,.5,.5))
		elif deriv[kk] < -1.0*thresh_flat: 
			count_C += 1
			plt.plot(x[kk],J_list[kk],'o',color=(.5,0,0))
		else: 
			count_F += 1 
			plt.plot(x[kk],J_list[kk],'o',color=(0,0,.5))
	
	# detect peaks and valleys 
	input_distance = 10; input_width = 5 
	th = .00; di = input_distance; wi = input_width # parameters
	peaks_U, _ = find_peaks(data_med,threshold=th,distance=di,width=wi)
	peaks_L, _ = find_peaks(-1.0*data_med,threshold=th,distance=di,width=wi)
	#num_peaks = 0.5 * peaks_U.shape[0] + 0.5 * peaks_L.shape[0]
	num_peaks = peaks_L.shape[0]
	if num_peaks == 0: num_peaks = 999999
	mean_C = count_C / num_peaks
	mean_R = count_R / num_peaks 
	mean_F = count_F / num_peaks 
	plt.grid()
	#plt.plot(x[peaks_U],data[peaks_U],'rx',markersize=10)	
	plt.plot(x[peaks_L],data[peaks_L],'rx',markersize=13)	
	plt.title('frames contract: %i, relax: %i, flat: %i'%(count_C,count_R,count_F))
	plt.xlabel('frame number')
	plt.ylabel('determinate of average F')
	plt.tight_layout()
	
	plt.savefig(out_analysis + '/recovered_F_plot_timeseries')
	if include_eps:
		plt.savefig(out_analysis + '/recovered_F_plot_timeseries.eps')
		
	return

##########################################################################################
def visualize_F_full_movie(folder_name,include_eps=False):
	"""Visualize the eigenvalues of F -- plot timeseries next to the movie. Must first run compute_F_whole_movie()."""
	external_folder_name = 'ALL_MOVIES_PROCESSED'
	if not os.path.exists(external_folder_name):
		os.makedirs(external_folder_name)

	out_analysis = external_folder_name + '/' + folder_name + '/analysis/F_movie'
	if not os.path.exists(external_folder_name + '/' + folder_name): os.makedirs(external_folder_name + '/' + folder_name)
	if not os.path.exists(out_analysis): os.makedirs(out_analysis)
	
	# import the deformation gradient. 
	F_list = np.loadtxt(external_folder_name + '/' + folder_name + '/analysis/recovered_F.txt')
	num_frames = F_list.shape[0]; x = [] 
	J_list = []
	R_list = []
	U_list = [] 
	F_list_mat = [] 
	lambda_1_list = []; vec_1_list = [] 
	lambda_2_list = []; vec_2_list = [] 
	th_list = [] 
	for kk in range(0,num_frames):
		F00 = F_list[kk,0]; F01 = F_list[kk,1]; F10 = F_list[kk,2]; F11 = F_list[kk,3]
		J_list.append(F00*F11 - F01*F10)
		x.append(kk)
		R, U = polar(np.asarray([[F00,F01],[F10,F11]]))
		R_list.append(R); U_list.append(U); F_list_mat.append(np.asarray([[F00,F01],[F10,F11]]))
		w, v = LA.eig(U)
		lambda_1_list.append(np.min(w)); lambda_2_list.append(np.max(w))
		v = np.dot(R, v)
		vec_1_list.append(v[:,np.argmin(w)]); vec_2_list.append(v[:,np.argmax(w)])
		th_list.append(np.arccos(v[0,0]))
	
	J_list = np.asarray(J_list)
	x = np.asarray(x)
	J_min = np.min(J_list)
		
	img_list = [] 
	# --> plot 
	for kk in range(0,num_frames):
		raw_img = get_frame_matrix(folder_name, kk)
		x_pos_mean = raw_img.shape[0]/2.0; y_pos_mean = raw_img.shape[1]/2.0
		plt.figure(figsize=(10*.7,5*.7))
		plt.subplot(1,2,1)
		plt.imshow(raw_img, cmap=plt.cm.gray)
		rad = .2*np.min([raw_img.shape[0],raw_img.shape[1]]); th = np.linspace(0,2.0*np.pi,100)
		plt.plot([y_pos_mean-rad*vec_1_list[kk][1],y_pos_mean+rad*vec_1_list[kk][1]],[x_pos_mean-rad*vec_1_list[kk][0],x_pos_mean+rad*vec_1_list[kk][0]],'-',color=(255/255,204/255,203/255),linewidth=0.3)
		plt.plot([y_pos_mean-rad*vec_2_list[kk][1],y_pos_mean+rad*vec_2_list[kk][1]],[x_pos_mean-rad*vec_2_list[kk][0],x_pos_mean+rad*vec_2_list[kk][0]],'-',color=(0.5,0.5,0.5),linewidth=0.3)
		#plt.plot([y_pos_mean,y_pos_mean],[x_pos_mean-rad,x_pos_mean+rad],'-',color=(255/255,204/255,203/255),linewidth=0.2)
		# add in eigenvector directions
		x_vec = []; y_vec = [] ; x_vec_circ = []; y_vec_circ = [] 
		scale = np.asarray([[.9,0],[0,.9]])
		for jj in range(0,100):
			v = np.asarray([rad*np.cos(th[jj]),rad*np.sin(th[jj])])
			#v_def = np.dot(np.dot(F_list_mat[jj],scale),v)
			nest1 = np.dot(F_list_mat[kk],F_list_mat[kk])
			nest2 = np.dot(F_list_mat[kk],nest1)
			nest3 = np.dot(F_list_mat[kk],nest2)
			nest4 = np.dot(F_list_mat[kk],nest3)
			nest5 = np.dot(F_list_mat[kk],nest4)
			nest6 = np.dot(F_list_mat[kk],nest5)
			nest7 = np.dot(F_list_mat[kk],nest6)
			nest8 = np.dot(F_list_mat[kk],nest7)
			v_def = np.dot(nest8,v)
			x_vec.append(v_def[0] + x_pos_mean); y_vec.append(v_def[1] + y_pos_mean)
			x_vec_circ.append(x_pos_mean + v[0]); y_vec_circ.append(y_pos_mean + v[1])
		
		plt.plot(y_vec_circ,x_vec_circ,'-',color=(255/255,204/255,203/255),linewidth=0.3)
		plt.plot(y_vec,x_vec,'-',color=(255/255,204/255,203/255),linewidth=1.0)
		
		ax = plt.gca()
		ax.set_xticks([]); ax.set_yticks([]);
		
		plt.subplot(1,2,2)
		plt.plot(x,lambda_1_list,'-',color='k',linewidth=1,label='λ1')
		plt.plot(x,lambda_2_list,'-',color=(0.5,0.5,0.5),linewidth=1,label='λ2')
		plt.plot(x[kk],lambda_1_list[kk],'o',mfc=(.7,0,0),mec=(0,0,0),markersize=7)
		plt.plot(x[kk],lambda_2_list[kk],'o',mfc=(.7,0,0),mec=(0.5,0.5,0.5),markersize=7)
		plt.xlabel('frame number')
		plt.legend()
		plt.tight_layout()
		plt.savefig(out_analysis + '/frame_%04d'%(kk))
		if include_eps:
			plt.savefig(out_analysis + '/frame_%i.eps'%(kk))
		plt.close()
		img_list.append(imageio.imread(out_analysis + '/frame_%04d.png'%(kk)))
	
	imageio.mimsave(out_analysis + '/F_anim.gif', img_list)
	
	return
	
##########################################################################################
def save_lambda_from_F(folder_name,include_eps=False):
	"""Visualize the eigenvalues of F -- plot timeseries next to the movie. Must first run compute_F_whole_movie()."""
	external_folder_name = 'ALL_MOVIES_PROCESSED'
	if not os.path.exists(external_folder_name):
		os.makedirs(external_folder_name)

	out_analysis = external_folder_name + '/' + folder_name + '/analysis'
	if not os.path.exists(external_folder_name + '/' + folder_name): os.makedirs(external_folder_name + '/' + folder_name)
	if not os.path.exists(out_analysis): os.makedirs(out_analysis)
	
	# import the deformation gradient. 
	F_list = np.loadtxt(external_folder_name + '/' + folder_name + '/analysis/recovered_F.txt')
	num_frames = F_list.shape[0]; x = [] 
	J_list = []
	R_list = []
	U_list = [] 
	F_list_mat = [] 
	lambda_1_list = []; vec_1_list = [] 
	lambda_2_list = []; vec_2_list = [] 
	th_list = [] 
	for kk in range(0,num_frames):
		F00 = F_list[kk,0]; F01 = F_list[kk,1]; F10 = F_list[kk,2]; F11 = F_list[kk,3]
		J_list.append(F00*F11 - F01*F10)
		x.append(kk)
		R, U = polar(np.asarray([[F00,F01],[F10,F11]]))
		R_list.append(R); U_list.append(U); F_list_mat.append(np.asarray([[F00,F01],[F10,F11]]))
		w, v = LA.eig(U)
		lambda_1_list.append(np.min(w)); lambda_2_list.append(np.max(w))
		v = np.dot(R, v)
		vec_1_list.append(v[:,np.argmin(w)]); vec_2_list.append(v[:,np.argmax(w)])
		th_list.append(np.arccos(v[0,0]))
	
	J_list = np.asarray(J_list)
	x = np.asarray(x)
	J_min = np.min(J_list)
		
	# --> plot 
	kk = np.argmin(J_list)
	raw_img = get_frame_matrix(folder_name, kk)
	x_pos_mean = raw_img.shape[0]/2.0; y_pos_mean = raw_img.shape[1]/2.0
	plt.figure(figsize=(10*.7,5*.7))
	plt.subplot(1,2,1)
	plt.imshow(raw_img, cmap=plt.cm.gray)
	rad = .2*np.min([raw_img.shape[0],raw_img.shape[1]]); th = np.linspace(0,2.0*np.pi,100)
	plt.plot([y_pos_mean-rad*vec_1_list[kk][1],y_pos_mean+rad*vec_1_list[kk][1]],[x_pos_mean-rad*vec_1_list[kk][0],x_pos_mean+rad*vec_1_list[kk][0]],'-',color=(255/255,204/255,203/255),linewidth=0.3)
	plt.plot([y_pos_mean-rad*vec_2_list[kk][1],y_pos_mean+rad*vec_2_list[kk][1]],[x_pos_mean-rad*vec_2_list[kk][0],x_pos_mean+rad*vec_2_list[kk][0]],'-',color=(0.5,0.5,0.5),linewidth=0.3)
	#plt.plot([y_pos_mean,y_pos_mean],[x_pos_mean-rad,x_pos_mean+rad],'-',color=(255/255,204/255,203/255),linewidth=0.2)
	# add in eigenvector directions
	x_vec = []; y_vec = [] ; x_vec_circ = []; y_vec_circ = [] 
	scale = np.asarray([[.9,0],[0,.9]])
	for jj in range(0,100):
		v = np.asarray([rad*np.cos(th[jj]),rad*np.sin(th[jj])])
		#v_def = np.dot(np.dot(F_list_mat[jj],scale),v)
		nest1 = np.dot(F_list_mat[kk],F_list_mat[kk])
		nest2 = np.dot(F_list_mat[kk],nest1)
		nest3 = np.dot(F_list_mat[kk],nest2)
		nest4 = np.dot(F_list_mat[kk],nest3)
		nest5 = np.dot(F_list_mat[kk],nest4)
		nest6 = np.dot(F_list_mat[kk],nest5)
		nest7 = np.dot(F_list_mat[kk],nest6)
		nest8 = np.dot(F_list_mat[kk],nest7)
		v_def = np.dot(nest8,v)
		x_vec.append(v_def[0] + x_pos_mean); y_vec.append(v_def[1] + y_pos_mean)
		x_vec_circ.append(x_pos_mean + v[0]); y_vec_circ.append(y_pos_mean + v[1])
	
	plt.plot(y_vec_circ,x_vec_circ,'-',color=(255/255,204/255,203/255),linewidth=0.3)
	plt.plot(y_vec,x_vec,'-',color=(255/255,204/255,203/255),linewidth=1.0)
	
	ax = plt.gca()
	ax.set_xticks([]); ax.set_yticks([]);
	
	plt.subplot(1,2,2)
	plt.plot(x,lambda_1_list,'-',color='k',linewidth=1,label='λ1')
	plt.plot(x,lambda_2_list,'-',color=(0.5,0.5,0.5),linewidth=1,label='λ2')
	plt.plot(x[kk],lambda_1_list[kk],'o',mfc=(.7,0,0),mec=(0,0,0),markersize=7)
	plt.plot(x[kk],lambda_2_list[kk],'o',mfc=(.7,0,0),mec=(0.5,0.5,0.5),markersize=7)
	plt.xlabel('frame number')
	plt.legend()
	plt.tight_layout()
	plt.savefig(out_analysis + '/' + folder_name + '_plot_lambda_frame_%04d'%(kk))
	if include_eps:
		plt.savefig(out_analysis + '/' + folder_name + '_plot_lambda_frame_%i.eps'%(kk))
	plt.close()
	
	lambda_list = np.zeros((len(lambda_1_list),2))
	lambda_list[:,0] = np.asarray(lambda_1_list)
	lambda_list[:,1] = np.asarray(lambda_2_list)	
	
	np.savetxt(out_analysis + 'lambda_all.txt',lambda_list)
	return





