import numpy as np
import matplotlib.pyplot as plt
import geom_fcns as geo
import render_fcns as ren
import os 
import sys

##########################################################################################
# investigation #1 -- sinusoidal w/ increasing amplitude 
##########################################################################################
# higher amp            straight
#    \       \       \   |
#     |       |       |  |
#    /       /       /   |
#   |       |       |    |
#    \       \       \   |
#     |       |       |  |
#    /       /       /   |
#   |       |       |    |
#    \       \       \   |
#     |       |      |   |
##########################################################################################

##########################################################################################
# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
# create geometry 
# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
##########################################################################################

case_list = [1,2,3]
res_list = [2,1,.75,.5] 


for zzz in range(0,len(case_list)):
	for xxx in range(0,len(res_list)):
		case = case_list[zzz]
		res = res_list[xxx]
		folder_name = 'synth_case_%i_val_%i'%(case,int(res*100))
		if not os.path.exists(folder_name):
			os.makedirs(folder_name)

		##########################################################################################
		# define undeformed geometry 
		##########################################################################################
		if case == 1:
			x_cent = 0; y_cent = 0; z_cent =  0; ellipse_a = 20; ellipse_b = 10; 
		elif case == 2:
			x_cent = 0; y_cent = 0; z_cent =  0; ellipse_a = 20; ellipse_b = 20; 
		elif case == 3:
			x_cent = 0; y_cent = 0; z_cent =  0; ellipse_a = 20; ellipse_b = 25; 
		th_min = 0; th_max = np.pi*2.0; num_sarc = 50
		sarc_list_1 = geo.sarc_list_ellipse_seg(x_cent, y_cent, z_cent, ellipse_a, ellipse_b, th_min, th_max, num_sarc)

		sarc_list = sarc_list_1 

		##########################################################################################
		# define and apply the deformation gradient F_homog_iso
		##########################################################################################
		max_contract = 0.15

		val_list = []
		for kk in range(0,5):  val_list.append(0)
		for kk in range(0,20): val_list.append(-1.0*max_contract*np.sin(kk/40*np.pi*2))
		for kk in range(0,5):  val_list.append(0)
		for kk in range(0,20): val_list.append(-1.0*max_contract*np.sin(kk/40*np.pi*2))
		for kk in range(0,5):  val_list.append(0)
		for kk in range(0,20): val_list.append(-1.0*max_contract*np.sin(kk/40*np.pi*2))
		for kk in range(0,5):  val_list.append(0)

		x0 = 0; y0 = 0; z0 = 0
		x_zone_1 = 5; x_zone_2 = 15
		F_fcn = geo.transform_helper_F_homog_iso

		sarc_list_ALL = geo.sarc_list_ALL_transform_F( sarc_list, val_list, x0, y0, z0, x_zone_1, x_zone_2, F_fcn)

		##########################################################################################
		# save geometry 
		##########################################################################################
		geo.plot_3D_geom(folder_name,sarc_list,'k','r-')
		geo.pickle_sarc_list_ALL(folder_name,sarc_list_ALL)
		sarc_array, sarc_array_normalized, x_pos_array, y_pos_array = geo.get_ground_truth(sarc_list_ALL)
		geo.plot_ground_truth_timeseries(sarc_array_normalized, folder_name)

		##########################################################################################
		# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
		# render
		# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
		##########################################################################################
		is_normal_radius = True; is_normal_height = True
		avg_radius = 1.5; avg_height = .5
		parameter_radius = 0.005; parameter_height = 0.002
		radius_list, height_list = ren.z_disk_props( sarc_list, is_normal_radius, is_normal_height, avg_radius, avg_height, parameter_radius, parameter_height)

		# --> begin loop, render each frame 
		num_frames = len(val_list)
		img_list = []

		for frame in range(0,num_frames):
			sarc_list = sarc_list_ALL[frame]
			# only keep sarcomeres that are within the frame
			z_lower = -1; z_upper = 1 
			sarc_list_in_slice, radius_list_in_slice, height_list_in_slice = ren.sarc_list_in_slice_fcn(sarc_list, radius_list, height_list, z_lower, z_upper)
			# turn into a 3D matrix of points
			x_lower = -30; x_upper = 30
			y_lower = -30; y_upper = 30
			z_lower = -5;  z_upper = 5
			val = res
			dim_x = int((x_upper-x_lower)/2*6*val); dim_y = int((y_upper-y_lower)/2*6*val); dim_z = int(5)
			mean_rad = radius_list_in_slice; mean_hei = height_list_in_slice
			bound_x = 10; bound_y = 10; bound_z = 10; val = 100
			matrix = ren.slice_to_matrix(sarc_list_in_slice,dim_x,dim_y,dim_z,x_lower,x_upper,y_lower,y_upper,z_lower,z_upper, mean_rad, mean_hei, bound_x, bound_y, bound_z,val)
			# add random 
			mean = 10; std_random = 1
			matrix = ren.random_val(matrix,mean,std_random)
			# add blur
			sig = 1
			matrix_blur = ren.matrix_gaussian_blur_fcn(matrix,sig)
			# convert matrix to image 
			slice_lower = 1; slice_upper = 4
			image = ren.matrix_to_image(matrix_blur,slice_lower,slice_upper)
			# image list 
			img_list.append(image)

		ren.save_img_stills(img_list,folder_name)
		ren.still_to_avi(folder_name,num_frames,False)
		ren.ground_truth_movie(folder_name,num_frames,img_list,sarc_array_normalized, x_pos_array, y_pos_array,x_lower,x_upper,y_lower,y_upper,dim_x,dim_y)
		#ren.still_to_avi(folder_name_render,num_frames,True)

		np.savetxt(folder_name + '/' + folder_name + '_GT_sarc_array_normalized.txt',sarc_array_normalized)
		np.savetxt(folder_name + '/' + folder_name + '_GT_x_pos_array.txt',(x_pos_array - x_lower)/(x_upper-x_lower)*dim_x)
		np.savetxt(folder_name + '/' + folder_name + '_GT_y_pos_array.txt',(y_pos_array - y_lower)/(y_upper-y_lower)*dim_y)




