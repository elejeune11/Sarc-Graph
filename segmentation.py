import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.filters import threshold_otsu, threshold_local
from skimage import measure
from scipy import ndimage
from scipy.spatial import distance
from collections import Counter
import os 
import pickle
import sys
import glob

##########################################################################################
# Input info and set up 
##########################################################################################
def segmentation_all(folder_name, gaussian_filter_size=1):
	"""Run all segmentation -- z-disks and sarcomeres"""
	num_frames = len(glob.glob('ALL_MOVIES_MATRICES/' + folder_name + '_matrices/*.npy'))

	external_folder_name = 'ALL_MOVIES_PROCESSED'
	if not os.path.exists(external_folder_name):
		os.makedirs(external_folder_name)

	out_bands = external_folder_name + '/' + folder_name + '/segmented_bands'
	out_sarc = external_folder_name + '/' + folder_name  + '/segmented_sarc'

	if not os.path.exists(external_folder_name + '/' + folder_name): os.makedirs(external_folder_name + '/' + folder_name)
	if not os.path.exists(out_bands): os.makedirs(out_bands)
	if not os.path.exists(out_sarc): os.makedirs(out_sarc)

	##########################################################################################
	# Helper functions
	##########################################################################################

	##########################################################################################
	def get_frame_matrix(frame):
		"""Get the npy matrix for a frame of the movie."""
		if frame < 10: file_root = '_matrices/frame-000%i'%(frame)
		elif frame < 100: file_root = '_matrices/frame-00%i'%(frame)
		else: file_root = '_matrices/frame-0%i'%(frame)
		root = 'ALL_MOVIES_MATRICES/' + folder_name + file_root  + '.npy'
		raw_img = np.load(root)
		return raw_img

	##########################################################################################
	def process_band(cont):
		"""Process the contour and return important properties. Units of pixels."""
		num = cont.shape[0]
		# coordinate 1 of center 
		center_idx1 = np.sum(cont[:,0]) / num
		# coordinate 2 of center 
		center_idx2 = np.sum(cont[:,1]) / num
		# perimeter 
		perim = num
		# find the maximum distance between points in the contour and identify the coordinates
		dist_mat = np.zeros((num,num))
		for kk in range(0,num):
			for jj in range(kk+1,num):
				dist_mat[kk,jj] =  ((cont[kk,0] - cont[jj,0])**2.0 + (cont[kk,1] - cont[jj,1])**2.0)**(1.0/2.0)
	
		leng = np.max(dist_mat)
		args = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
		idx_kk = args[0]
		idx_jj = args[1] 
		vec_x = cont[idx_kk,0] - cont[idx_jj,0]
		vec_y = cont[idx_kk,1] - cont[idx_jj,1]
		unit_vector = [vec_x/leng, vec_y/leng] 
		# identify end_1 and end_2 -- coordinates of the ends 
		end_1 = [cont[idx_kk,0],cont[idx_kk,1]] 
		end_2 = [cont[idx_jj,0],cont[idx_jj,1]] 

		return center_idx1, center_idx2, unit_vector, leng, perim, end_1, end_2

	##########################################################################################
	def compute_length_from_contours(cont1,cont2):
		"""Compute the length between two z disks from two contours"""
		c1_x = np.mean(cont1[:,0])
		c1_y = np.mean(cont1[:,1])
		c2_x = np.mean(cont2[:,0])
		c2_y = np.mean(cont2[:,1])
		return ((c1_x - c2_x)**2.0 + (c1_y - c2_y)**2.0)**0.5
	
	##########################################################################################
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	# Segment z-disks 
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	##########################################################################################
	for frame in range(0,num_frames):
		if frame < 10: file_root = '/frame-000%i'%(frame)
		elif frame < 100: file_root = '/frame-00%i'%(frame)
		else: file_root = '/frame-0%i'%(frame)
		# get the raw image matrix -- pre-processed in "file_pre_processing.py"
		raw_img = get_frame_matrix(frame)
	
		# compute the laplacian of the image and then use that to find the contours
		box = -1
		laplacian = cv2.Laplacian(raw_img,cv2.CV_64F)
		laplacian = ndimage.gaussian_filter(laplacian, gaussian_filter_size)
		contour_thresh = threshold_otsu(laplacian)
		contour_image = laplacian
		contours =  measure.find_contours(contour_image,contour_thresh)
	
		# create a list of all contours large enough (in pixels) to surround an area
		total = 0
		contour_list = [] 
		for n, contour in enumerate(contours):
			total += 1
			if contour.shape[0] >= 8:
				contour_list.append(contour)
			
		# compute properties of each contour i.e. z disk
		num_contour = len(contour_list) 
		unit_vector_list = []; band_center_idx1 = []; band_center_idx2 = [] 
		end_1_list = []; end_2_list = []; leng_list = [] 
		for kk in range(0,num_contour):
			con = contour_list[kk]
			center_idx1, center_idx2, unit_vector, leng, perim, end_1, end_2 = process_band(con)
			unit_vector_list.append(unit_vector)
			end_1_list.append(end_1)
			end_2_list.append(end_2)
			band_center_idx1.append(center_idx1)
			band_center_idx2.append(center_idx2)
			leng_list.append(leng)

		# save info per band: center x, center y, end_1x,y, end_2 x,y
		num_bands = len(contour_list)
		info = np.zeros((num_bands,6))
		info[:,0] = np.asarray(band_center_idx1)
		info[:,1] = np.asarray(band_center_idx2)
		info[:,2:4] = np.asarray(end_1_list)
		info[:,4:6] = np.asarray(end_2_list)
		np.savetxt(out_bands  + file_root + '_bands.txt',info)
	
		# save contour_list --> pickle the file to come back to later 
		with open(out_bands + file_root + '_raw_contours.pkl', 'wb') as f:
			pickle.dump(contour_list, f)

	##########################################################################################
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	# Segment sarcomeres
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	##########################################################################################
	for t in range(0,num_frames):
		# identify sarcomeres from the segmented z-disks -- open the z-disks file 
		if t < 10: file_root = '/frame-000%i'%(t)
		elif t < 100: file_root = '/frame-00%i'%(t)
		else: file_root = '/frame-0%i'%(t)
		root = folder_name + file_root 
		filename_all = out_bands  + file_root + '_bands.txt'
		segmented_data = np.loadtxt(filename_all)
		with open(out_bands + file_root + '_raw_contours.pkl', 'rb') as f:
			contour_list = pickle.load(f)
		disc_data = segmented_data
		coords = disc_data[:,0:2]
		ends_1 = disc_data[:,2:4]
		ends_2 = disc_data[:,4:6]
		num_discs = disc_data.shape[0]
	
		# --> get median shortest distance between points 
		actual_dist = distance.cdist(coords,coords,'euclidean') + np.eye(num_discs)*999999
		min_list = [] 
		for kk in range(0,actual_dist.shape[0]):
			min_list.append(np.min(actual_dist[kk,:]))

		median_neigh = np.median(min_list)
		upper_limit = np.sort(min_list)[int(0.9*len(min_list))]

		# create ghost points matrix 
		num_ghost_pts = num_discs*2
		ghost_pts = np.zeros((num_ghost_pts,2))
		for kk in range(0,num_discs):
			idx1 = kk*2
			idx2 = kk*2 + 1 
			# create the unit vector from the end points
			x1 = ends_1[kk,0]
			y1 = ends_1[kk,1]
			x2 = ends_2[kk,0]
			y2 = ends_2[kk,1]
			mag = ((x1-x2)**2.0 + (y1-y2)**2.0)**0.5
			unit_vec_x = (x1-x2)/mag
			unit_vec_y = (y1-y2)/mag
			x_gap = -1.0*unit_vec_y
			y_gap = unit_vec_x
			ghost_pts[idx1,0] = coords[kk,0] - x_gap*median_neigh/2.0
			ghost_pts[idx1,1] = coords[kk,1] - y_gap*median_neigh/2.0
			ghost_pts[idx2,0] = coords[kk,0] + x_gap*median_neigh/2.0
			ghost_pts[idx2,1] = coords[kk,1] + y_gap*median_neigh/2.0

		ghost_dist = distance.cdist(ghost_pts,ghost_pts,'euclidean') 

		# distance correction for things matching to themselves 
		for kk in range(0,num_ghost_pts):
			for jj in range(0,num_ghost_pts):
				if kk == jj:
					ghost_dist[kk,jj] = 999999
				if kk % 2 == 0 and kk + 1 == jj:
					ghost_dist[kk,jj] = 999999
				if kk % 2 == 1 and kk - 1 == jj:
					ghost_dist[kk,jj] = 999999

		# ghost match
		ghost_match = np.zeros((num_ghost_pts))
		for kk in range(0,num_ghost_pts):
			ghost_match[kk] = int(np.argmin(ghost_dist[kk,:]))

		# unique ghost match
		match_list = [] 
		for kk in range(0,num_ghost_pts):
			if kk < ghost_match[kk]:
				match_list.append((kk,int(ghost_match[kk])))
			else:
				match_list.append((int(ghost_match[kk]),kk))

		c = Counter(match_list)
		match_list_unique = [] 
		for item in c:
			reps = c[item]
			if reps > 1:
				match_list_unique.append(item)
	
		# extract information and save 
		num_unique = len(match_list_unique)
		sarc_data = []

		for kk in range(0,num_unique):
			gp1 = int(match_list_unique[kk][0])
			gp2 = int(match_list_unique[kk][1])
			pt1 = int((gp1 - gp1 % 2) / 2.0)
			pt2 = int((gp2 - gp2 % 2) / 2.0)

			dist_between_ghost_pts = ((ghost_pts[gp1,0]-ghost_pts[gp2,0])**2.0 + (ghost_pts[gp1,1]-ghost_pts[gp2,1])**2.0)**(1.0/2.0)

			x1 = coords[pt1,0]; y1 = coords[pt1,1]
			x2 = coords[pt2,0]; y2 = coords[pt2,1]

			if dist_between_ghost_pts < upper_limit:
				# get ID1 
				ID1 = pt1
				# get ID2
				ID2 = pt2
				# get center_x
				center_x = 0.5*(x1 + x2)
				# get center_y
				center_y = 0.5*(y1 + y2)
				# get length
				length = compute_length_from_contours(contour_list[pt1],contour_list[pt2])
				# data from ends
				e1x1 = ends_1[pt1,0]; e1x2 = ends_2[pt1,0]
				e1y1 = ends_1[pt1,1]; e1y2 = ends_2[pt1,1]
				e2x1 = ends_1[pt2,0]; e2x2 = ends_2[pt2,0]
				e2y1 = ends_1[pt2,1]; e2y2 = ends_2[pt2,1]
				# get width 
				wid1 = ((e1x1-e1x2)**2.0 + (e1y1-e1y2)**2.0)**0.5
				wid2 = ((e2x1-e2x2)**2.0 + (e2y1-e2y2)**2.0)**0.5
				width = 0.5*(wid1 + wid2)
				# get angle 
				angle = np.arctan2( y2 - y1 , x2 - x1 )
				if angle < 0:
					angle += np.pi

				sarc_data.append([ID1,ID2,center_x,center_y,length,width,angle])

		np.savetxt(out_sarc + file_root + '_sarc_data.txt', np.asarray(sarc_data))

	return


