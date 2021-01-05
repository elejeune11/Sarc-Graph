import numpy as np
import matplotlib.pyplot as plt
import geom_fcns as geo
import render_fcns as ren
import os 
import sys

##########################################################################################
# investigation #3 -- ellipses close to each other with inhomogeneous contraction 
##########################################################################################
#		 \  \  \
#		  \  \  \
#		   |  |  |
#		   |  |  |
#		   |  |  |
#		   |  |  |
#		  /  /  /
#		 /  /  /
##########################################################################################

folder_name = 'synthetic_data_S3'
if not os.path.exists(folder_name):
	os.makedirs(folder_name)

##########################################################################################
# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
# create geometry 
# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
##########################################################################################

##########################################################################################
# define undeformed geometry 
##########################################################################################

x_cent = 20; y_cent = 0; z_cent = 0
ellipse_a = 20; ellipse_b = 5; th_min = 0.2; th_max = np.pi - 0.2; num_sarc = 15
sarc_list_1 = geo.sarc_list_ellipse_seg(x_cent, y_cent, z_cent, ellipse_a, ellipse_b, th_min, th_max, num_sarc)

x_cent = 20; y_cent = -4; z_cent = 0
ellipse_a = 20; ellipse_b = 5; th_min = 0.5; th_max = np.pi - 0.5; num_sarc = 15
sarc_list_2 = geo.sarc_list_ellipse_seg(x_cent, y_cent, z_cent, ellipse_a, ellipse_b, th_min, th_max, num_sarc)

x_cent = 20; y_cent = -8; z_cent = 0
ellipse_a = 20; ellipse_b = 5; th_min = 0.75; th_max = np.pi - 0.75; num_sarc = 10
sarc_list_3 = geo.sarc_list_ellipse_seg(x_cent, y_cent, z_cent, ellipse_a, ellipse_b, th_min, th_max, num_sarc)

sarc_list = sarc_list_1 + sarc_list_2 + sarc_list_3

##########################################################################################
# define and apply the deformation gradient F_homog_iso
##########################################################################################

val_list = []
v = .01
v2 = 40
for kk in range(0,v2): val_list.append(kk*v)
for kk in range(0,v2): val_list.append(v*v2 - kk*v) 
x0 = 20; y0 = 0; z0 = 0; x_zone_1 = 5; x_zone_2 = 15
F_fcn = geo.transform_helper_F_nonhomog_aniso

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
avg_radius = 1.0; avg_height = 0.5
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
	x_lower = -10; x_upper = 50
	y_lower = -30; y_upper = 30
	z_lower = -5;  z_upper = 5
	dim_x = int((x_upper-x_lower)/2*6); dim_y = int((y_upper-y_lower)/2*6); dim_z = int(4)
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
	slice_lower = 0; slice_upper = 4
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




