import numpy as np
import matplotlib.pyplot as plt
import geom_fcns as geo
import render_fcns as ren
import os 
import sys

##########################################################################################
# investigation #5 -- multiple disorganized chains not in the same plane 
##########################################################################################
#			\
#    ---     |
#        \  /
#          |  \      |
#           \  \     |
#            |  \    |
#           /    \   |
#   ----------------------------
#          |         |
#           \
#            | 
##########################################################################################

folder_name = 'synthetic_data_S5'
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

x_end_1 = 0; y_end_1 = 0; z_end_1 = 0; x_end_2 = 40; y_end_2 = 0; z_end_2 = 0
sin_amplitude = 1; sin_period = 10; num_sarc = 15
sarc_sin_1 = geo.sarc_list_sin_seg_y(x_end_1,y_end_1,z_end_1,x_end_2,y_end_2,z_end_2,sin_amplitude,sin_period,num_sarc)

x_end_1 = 0; y_end_1 = 10; z_end_1 = 1.5; x_end_2 = 40; y_end_2 = 10; z_end_2 = -1.5
sin_amplitude = .25; sin_period = 5; num_sarc = 15
sarc_sin_2 = geo.sarc_list_sin_seg_y(x_end_1,y_end_1,z_end_1,x_end_2,y_end_2,z_end_2,sin_amplitude,sin_period,num_sarc)

x_end_1 = 20; y_end_1 = -30; z_end_1 = -1.5; x_end_2 = 25; y_end_2 = 30; z_end_2 = 1.5
num_sarc = 25
sarc_lin_3 = geo.sarc_list_line_seg(x_end_1,y_end_1,z_end_1,x_end_2,y_end_2,z_end_2,num_sarc)

x_cent = 20; y_cent = -20; z_cent = .25
ellipse_a = 20; ellipse_b = 30; th_min = np.pi/2.0; th_max = np.pi - .2; num_sarc = 15
sarc_eli_4 = geo.sarc_list_ellipse_seg(x_cent, y_cent, z_cent, ellipse_a, ellipse_b, th_min, th_max, num_sarc)

##########################################################################################
# define and apply the deformation gradient F_homog_iso
##########################################################################################

x0 = 20; y0 = 0; z0 = 0; x_zone_1 = 5; x_zone_2 = 15

# --> segment 1 --------------------------------------------------------------------------
val_list = []
v = .01; v2 = 40
for kk in range(0,v2): val_list.append(kk*v)
for kk in range(0,v2): val_list.append(v*v2 - kk*v) 
F_fcn = geo.transform_helper_F_nonhomog_aniso
sarc_list_ALL_1 = geo.sarc_list_ALL_transform_F( sarc_sin_1, val_list, x0, y0, z0, x_zone_1, x_zone_2, F_fcn)

# --> segment 2 --------------------------------------------------------------------------

val_list = []
v = -.01; v2 = 40
for kk in range(0,v2): val_list.append(kk*v)
for kk in range(0,v2): val_list.append(v*v2 - kk*v) 
F_fcn = geo.transform_helper_F_nonhomog_aniso
sarc_list_ALL_2 = geo.sarc_list_ALL_transform_F( sarc_sin_2, val_list, x0, y0, z0, x_zone_1, x_zone_2, F_fcn)

# --> segment 3 --------------------------------------------------------------------------
val_list = []
v = .005; v2 = 40
for kk in range(0,v2): val_list.append(kk*v)
for kk in range(0,v2): val_list.append(v*v2 - kk*v) 
F_fcn = geo.transform_helper_F_homog_iso
sarc_list_ALL_3 = geo.sarc_list_ALL_transform_F( sarc_lin_3, val_list, x0, y0, z0, x_zone_1, x_zone_2, F_fcn)

# --> segment 4 --------------------------------------------------------------------------

val_list = []
v = .01; v2 = 40
for kk in range(0,int(v2/2)): val_list.append(kk*v)
v = .005
for kk in range(0,int(v2/2)): val_list.append(.01*v2/2 + kk*v)
v = .01
for kk in range(0,int(v2/2)): val_list.append(.01*v2/2 + .005*v2/2 - kk*v) 
v = .005
for kk in range(0,int(v2/2)): val_list.append(.01*v2/2 + .005*v2/2 - .01*v2/2 - kk*v) 
F_fcn = geo.transform_helper_F_nonhomog_aniso
sarc_list_ALL_4 = geo.sarc_list_ALL_transform_F( sarc_eli_4, val_list, x0, y0, z0, x_zone_1, x_zone_2, F_fcn)

# --> put all segments together ----------------------------------------------------------
sarc_list = sarc_sin_1 + sarc_sin_2 + sarc_lin_3 + sarc_eli_4

sarc_list_ALL = []
for kk in range(0,v2*2):
	SL = sarc_list_ALL_1[kk] + sarc_list_ALL_2[kk] + sarc_list_ALL_3[kk] + sarc_list_ALL_4[kk] 
	sarc_list_ALL.append(SL)

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









