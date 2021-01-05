import numpy as np
import matplotlib.pyplot as plt
import geom_fcns as geo
import render_fcns as ren
import os 
import sys
##########################################################################################
# investigation #4 -- messier example with asynchronous contraction 
##########################################################################################
folder_name = 'synthetic_data_S4'
if not os.path.exists(folder_name):
	os.makedirs(folder_name)

##########################################################################################
#                . -- .
#            *             *
#        *    _               *
#      *     | \     |    |     *
#	  *      |  |    |    |      *
#     *      | /     |    |      *
#	  *      | \     |    |      *
#      *     |  |    |    |     *
#        *   |_/     \____/   *
#           *              *
#               '  --  '
##########################################################################################

##########################################################################################
# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
# create geometry 
# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
##########################################################################################

##########################################################################################
# define undeformed geometry 
##########################################################################################
# seg 1  -- ellipse -- outer ring
x_cent = 15; y_cent = 3; z_cent = .1
ellipse_a = 20; ellipse_b = 25; th_min = .1; th_max = np.pi*2.0
num_sarc = 40
sl_1 = geo.sarc_list_ellipse_seg(x_cent, y_cent, z_cent, ellipse_a, ellipse_b, th_min, th_max, num_sarc)

# seg 2  -- inner
x_cent = 10; y_cent = -10; z_cent = 0
ellipse_a = 5; ellipse_b = 9; th_min = .2; th_max = np.pi - .1
num_sarc = 10
sl_2 = geo.sarc_list_ellipse_seg(x_cent, y_cent, z_cent, ellipse_a, ellipse_b, th_min, th_max, num_sarc)

# seg 3  -- inner
x_cent = 20; y_cent = -10; z_cent = 0
ellipse_a = 5; ellipse_b = 9; th_min = .2; th_max = np.pi - .1
num_sarc = 10
sl_3 = geo.sarc_list_ellipse_seg(x_cent, y_cent, z_cent, ellipse_a, ellipse_b, th_min, th_max, num_sarc)

# seg 4  -- inner
x_end_1 = 5;  y_end_1 = -10; z_end_1 = .1
x_end_2 = 25; y_end_2 = -10; z_end_2 = -.1
num_sarc = 10
sl_4 = geo.sarc_list_line_seg(x_end_1,y_end_1,z_end_1,x_end_2,y_end_2,z_end_2,num_sarc)

# seg 5  -- inner
x_cent = 3; y_cent = 10.5; z_cent = 0
ellipse_a = 20; ellipse_b = 7; th_min = .1; th_max = np.pi/2.0 
num_sarc = 10
sl_5 = geo.sarc_list_ellipse_seg(x_cent, y_cent, z_cent, ellipse_a, ellipse_b, th_min, th_max, num_sarc)

# seg 6  -- inner
x_cent = 3; y_cent = 10; z_cent = 0
ellipse_a = 20; ellipse_b = 7; th_min = -.1; th_max = -np.pi/2.0 
num_sarc = 10
sl_6 = geo.sarc_list_ellipse_seg(x_cent, y_cent, z_cent, ellipse_a, ellipse_b, th_min, th_max, num_sarc)

sarc_list_A = sl_1

sarc_list_B = sl_2 + sl_3 + sl_4 + sl_5 + sl_6 

sarc_list = sarc_list_A  + sarc_list_B

##########################################################################################
# define and apply the deformation gradient F_homog_iso
##########################################################################################
val_list_A = []
v = .003
v2 = 40
for kk in range(0,15): val_list_A.append(kk*v)
for kk in range(0,5):  val_list_A.append(15*v) 
for kk in range(0,15): val_list_A.append(15*v - kk*v)
for kk in range(0,15): val_list_A.append(kk*v)
for kk in range(0,5):  val_list_A.append(15*v) 
for kk in range(0,15): val_list_A.append(15*v - kk*v)
for kk in range(0,5):  val_list_A.append(15*v) 
for kk in range(0,5):  val_list_A.append(kk*v)

x0 = 15; y0 = 0; z0 = 0
x_zone_1 = 10; x_zone_2 = 20
F_fcn = geo.transform_helper_F_homog_iso
sarc_list_ALL_A = geo.sarc_list_ALL_transform_F( sarc_list_A, val_list_A, x0, y0, z0, x_zone_1, x_zone_2, F_fcn)

val_list_B = [] 
for kk in range(0,int(v2*2)): val_list_B.append(np.sin(kk/20)*.1)

F_fcn = geo.transform_helper_F_nonhomog_aniso
sarc_list_ALL_B = geo.sarc_list_ALL_transform_F( sarc_list_B, val_list_B, x0, y0, z0, x_zone_1, x_zone_2, F_fcn)

sarc_list_ALL = []
for kk in range(0,len(val_list_A)):
	SL = sarc_list_ALL_A[kk] + sarc_list_ALL_B[kk]
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
avg_radius = 2.0; avg_height = 1
parameter_radius = 0.25; parameter_height = 0.1
radius_list_A, height_list_A = ren.z_disk_props( sarc_list_A, is_normal_radius, is_normal_height, avg_radius, avg_height, parameter_radius, parameter_height)

is_normal_radius = True; is_normal_height = True
avg_radius = 2.0 / 3.0; avg_height = .5
parameter_radius = 0.005; parameter_height = 0.002
radius_list_B, height_list_B = ren.z_disk_props( sarc_list_B, is_normal_radius, is_normal_height, avg_radius, avg_height, parameter_radius, parameter_height)

radius_list = radius_list_A + radius_list_B
height_list = height_list_A + height_list_B 

num_frames = len(val_list_A)

img_list = []

for frame in range(0,num_frames):
	sarc_list = sarc_list_ALL[frame]
	# only keep sarcomeres that are within the frame
	z_lower = -10; z_upper = 10 
	sarc_list_in_slice, radius_list_in_slice, height_list_in_slice = ren.sarc_list_in_slice_fcn(sarc_list, radius_list, height_list, z_lower, z_upper)
	# turn into a 3D matrix of points
	x_lower = -15; x_upper = 40
	y_lower = -30; y_upper = 40
	z_lower = -5; z_upper = 5
	dim_x = int((x_upper-x_lower)/2.0*6); dim_y = int((y_upper-y_lower)/2.0*6)
	dim_z = int(4)
	mean_rad = radius_list_in_slice; mean_hei = height_list_in_slice
	bound_x = 10; bound_y = 10; bound_z = 10; val = 100
	matrix = ren.slice_to_matrix(sarc_list_in_slice,dim_x,dim_y,dim_z,x_lower,x_upper,y_lower,y_upper,z_lower,z_upper, mean_rad, mean_hei, bound_x, bound_y, bound_z,val)
	# add random 
	mean = 10; std = 1
	matrix = ren.random_val(matrix,mean,std)
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


