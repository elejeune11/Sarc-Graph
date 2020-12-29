import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

##########################################################################################
# 	FUNCTIONS FOR GEOMETRY I.E. CREATE AND ANIMATE GEOMETRY 
##########################################################################################

##########################################################################################
# Functions to create "sarc_list" a list of sarcomeres with the end coordinates specified 
##########################################################################################
# function: sarc_list_line_seg
# function: sarc_list_ellipse_seg
# function: sarc_list_sin_seg_y
# function: sarc_list_sin_spin_seg
# function: sarc_list_branch_seg

##########################################################################################
# Functions (and helper functions) to transform "sarc_list"
# deformation gradient based transformations
# displacement based transformations 
##########################################################################################
# function: transform_helper_F_homog_iso
# function: transform_helper_F_nonhomog_aniso
# function: transform_helper_Rotation_x
# function: transform_helper_Rotation_y
# function: transform_helper_Rotation_z
# function: transform

# function: displace_random_nudge
# function: displace_constant_value

# function: sarc_list_ALL_transform_F
# function: sarc_list_ALL_displace_nudge

##########################################################################################
# Functions to run script, save, and make plots  
##########################################################################################
# function: get_len
# function: get_pos
# function: pickle_sarc_list_ALL
# function: get_ground_truth
# function: plot_3D_geom
# function: plot_ground_truth_timeseries

##########################################################################################
# sarc_list functions: define a baseline geometry for a single continuous segment 
##########################################################################################
##########################################################################################
def sarc_list_line_seg(x_end_1,y_end_1,z_end_1,x_end_2,y_end_2,z_end_2,num_sarc):
	"""Create a sarc_list for a straight line segment."""
	x_vec = np.linspace(x_end_1,x_end_2,num_sarc+1)
	y_vec = np.linspace(y_end_1,y_end_2,num_sarc+1)
	z_vec = np.linspace(z_end_1,z_end_2,num_sarc+1)
	sarc_list = [] 
	for kk in range(0,num_sarc):
		pt_1 = (x_vec[kk],y_vec[kk],z_vec[kk])
		pt_2 = (x_vec[kk+1],y_vec[kk+1],z_vec[kk+1])
		sarc = [pt_1,pt_2]
		sarc_list.append(sarc)
		
	return sarc_list
	
##########################################################################################
def sarc_list_ellipse_seg(x_cent, y_cent, z_cent, ellipse_a, ellipse_b, th_min, th_max, num_sarc):
	"""Create a sarc_list for an ellipse segment that sweeps through theta."""
	# parametric equation for an ellipse
	# x = ellipse_a * cos(th) + x_cent
	# y = ellipse_b * sin(th) + y_cent
	# z = z_cent
	sarc_list = [] 
	th_vec = np.linspace(th_min,th_max,num_sarc+1)
	for kk in range(0,num_sarc):
		th = th_vec[kk]
		pt_1 = (ellipse_a*np.cos(th) + x_cent , ellipse_b*np.sin(th) + y_cent , z_cent)
		th = th_vec[kk+1]
		pt_2 = (ellipse_a*np.cos(th) + x_cent , ellipse_b*np.sin(th) + y_cent , z_cent)
		sarc = [pt_1,pt_2]
		sarc_list.append(sarc)
		
	return sarc_list
	
##########################################################################################
def sarc_list_sin_seg_y(x_end_1,y_end_1,z_end_1,x_end_2,y_end_2,z_end_2,sin_amplitude,sin_period,num_sarc):
	"""Create a sarc_list for a segment with sinusoidal fluctuations in y only."""
	sarc_list = [] 
	leng = ((x_end_1-x_end_2)**2.0 + (y_end_1-y_end_2)**2.0 + (z_end_1-z_end_2)**2.0)**0.5
	th_min = 0.0
	th_max = leng / sin_period * 2.0 * np.pi 
	th_vec = np.linspace(th_min,th_max,num_sarc+1)
	x_vec = np.linspace(x_end_1, x_end_2,num_sarc+1)
	y_vec = np.linspace(y_end_1, y_end_2,num_sarc+1)
	z_vec = np.linspace(z_end_1, z_end_2,num_sarc+1)
	for kk in range(0,num_sarc):
		th = th_vec[kk]
		si = np.sin(th/sin_period*(2*np.pi))*sin_amplitude
		pt_1 = (x_vec[kk],y_vec[kk] + si, z_vec[kk])
		th = th_vec[kk+1]
		si = np.sin(th/sin_period*(2*np.pi))*sin_amplitude
		pt_2 = (x_vec[kk+1],y_vec[kk+1] + si, z_vec[kk+1])
		sarc = [pt_1,pt_2]
		sarc_list.append(sarc)
		
	return sarc_list
	
##########################################################################################
def sarc_list_sin_spin_seg(x_end_1,y_end_1,z_end_1,x_end_2,y_end_2,z_end_2,sin_amplitude,sin_period,num_sarc): 
	"""Create a sarc_list for a segment with sinusoidal fluctuations in y and z."""
	sarc_list = [] 
	leng = ((x_end_1-x_end_2)**2.0 + (y_end_1-y_end_2)**2.0 + (z_end_1-z_end_2)**2.0)**0.5
	th_min = 0.0
	th_max = leng / sin_period * 2.0 * np.pi 
	th_vec = np.linspace(th_min,  th_max, num_sarc+1)
	x_vec = np.linspace(x_end_1, x_end_2, num_sarc+1)
	y_vec = np.linspace(y_end_1, y_end_2, num_sarc+1)
	z_vec = np.linspace(z_end_1, z_end_2, num_sarc+1)
	for kk in range(0,num_sarc):
		th = th_vec[kk]
		si = np.sin( th / (2*np.pi) * sin_period ) * sin_amplitude
		pt_1 = (x_vec[kk],y_vec[kk] + si*np.sin(th/(2*np.pi)*per), z_vec[kk] + si*np.cos(th/(2*np.pi)*per))
		th = th_vec[kk+1]
		si = np.sin( th / (2*np.pi) * sin_period) * sin_amplitude
		pt_2 = (x_vec[kk+1],y_vec[kk+1] + si*np.sin(th/(2*np.pi)*sin_period), z_vec[kk+1] + si*np.cos(th/(2*np.pi)*sin_period))
		sarc = [pt_1,pt_2]
		sarc_list.append(sarc)
		
	return sarc_list
	
##########################################################################################
def sarc_list_branch_seg(x_end_1,y_end_1,z_end_1,x_end_2,y_end_2,z_end_2,x_end_3,y_end_3,z_end_3,x_end_4,y_end_4,z_end_4,num_sarc13, num_sarc23, num_sarc34):
	"""Create 3 sarc_lists for a Y branch segment."""
	sarc_list1 = sarc_list_line_seg(x_end_1,y_end_1,z_end_1,x_end_3,y_end_3,z_end_3,num_sarc13)
	sarc_list2 = sarc_list_line_seg(x_end_2,y_end_2,z_end_2,x_end_3,y_end_3,z_end_3,num_sarc23)
	sarc_list3 = sarc_list_line_seg(x_end_3,y_end_3,z_end_3,x_end_4,y_end_4,z_end_4,num_sarc34)
	return sarc_list1, sarc_list2, sarc_list3
##########################################################################################

##########################################################################################
##########################################################################################
# make alterations to a baseline geometry -- nudge or deformation grad
##########################################################################################
##########################################################################################

##########################################################################################
# deformation gradient based transformation 
##########################################################################################

##########################################################################################
def transform_helper_F_homog_iso( x, y, z, val, x1, x2): 
	"""Create an isotropic homogeneous deformation gradient F."""
	F = np.asarray([[val + 1,0,0],[0,val + 1,0],[0,0,val + 1]])
	return F

##########################################################################################
def transform_helper_F_nonhomog_aniso( x, y, z, val, x_zone_1, x_zone_2):
	"""Create an inhomogeneous deformation gradient F (can replace this with any function)."""
	if x < x_zone_1:
		val_adjusted = val
	elif x > x_zone_2:
		val_adjusted = val * (-1)
	else:
		val_adjusted = val - (x - x_zone_1)/(x_zone_2-x_zone_1)*(2*val)
	
	F = np.asarray([[val_adjusted + 1 ,0,0],[0,val_adjusted + 1 ,0],[0,0,val_adjusted + 1]])
	return F

##########################################################################################
def transform_helper_Rotation_x(th):
	"""Create a rotation tensor about the x axis."""
	Rx = np.asarray([[1,0,0],[0,np.cos(th),-1.0*np.sin(th)],[0,np.sin(th),np.cos(th)]])
	return Rx

##########################################################################################
def transform_helper_Rotation_y(th):
	"""Create a rotation tensor about the y axis."""
	Ry = np.asarray([[np.cos(th),0,np.sin(th)],[0,1,0],[-1.0*np.sin(th),0,np.cos(th)]])
	return Ry

##########################################################################################
def transform_helper_Rotation_z(th):
	"""Create a rotation tensor about the z axis."""
	Rz = np.asarray([[np.cos(th), -1.0*np.sin(th),0],[np.sin(th),np.cos(th),0],[0,0,1]])
	return Rz

##########################################################################################
def transform( sarc_list, F_fcn, val, x0, y0, z0, x1, x2):
	"""Transform every point in sarc_list by a defined deformation gradient."""
	sarc_list_transformed = [] 
	num_sarc = len(sarc_list)
	for kk in range(0,num_sarc):
		s1 = sarc_list[kk][0]
		s2 = sarc_list[kk][1]
		v1 = np.asarray([s1[0] - x0, s1[1] - y0, s1[2] - z0])
		v2 = np.asarray([s2[0] - x0, s2[1] - y0, s2[2] - z0])
		# find the position of the sarcomere
		x =  0.5*s1[0] + 0.5*s2[0]
		y =  0.5*s1[1] + 0.5*s2[1]
		z =  0.5*s1[2] + 0.5*s2[2]
		# use the position of the sarcomere to find F
		F = F_fcn(x,y,z,val,x1,x2)
		# multiply the deformation gradient by the old position vector
		v1_new = np.dot(F,v1)
		v2_new = np.dot(F,v2)
		# create new sarcomere pair 
		s1 = (v1_new[0] + x0,v1_new[1] + y0,v1_new[2] + z0)
		s2 = (v2_new[0] + x0,v2_new[1] + y0,v2_new[2] + z0)
		s = [s1, s2]
		sarc_list_transformed.append(s)
		
	return sarc_list_transformed

##########################################################################################
# displace sarcomere list 
##########################################################################################

##########################################################################################
def displace_random_nudge(sarc_list, is_normal, random_parameter_x, random_parameter_y, random_parameter_z):
	"""Displace every point in sarc_list by a small random nudge."""
	# make sure that edges -- i.e. z disks -- get transformed together --
	num_sarc = len(sarc_list)
	for kk in range(0,num_sarc-1):
		if is_normal: # normally distributed parameter
			nudge_x = np.random.normal(0,random_parameter_x)
			nudge_y = np.random.normal(0,random_parameter_y)
			nudge_z = np.random.normal(0,random_parameter_z)
		else: # uniformly distributed parameter 
			nudge_x = (np.random.random(1)[0] - .5) * random_parameter_x * 2.0
			nudge_y = (np.random.random(1)[0] - .5) * random_parameter_y * 2.0
			nudge_z = (np.random.random(1)[0] - .5) * random_parameter_z * 2.0
			
		s1 = sarc_list[kk][1] 
		s2 = sarc_list[kk+1][0]
		s10 = s1[0] + nudge_x; s20 = s2[0] + nudge_x
		s11 = s1[1] + nudge_y; s21 = s2[1] + nudge_y
		s12 = s1[2] + nudge_z; s22 = s2[2] + nudge_z
		s1 = (s10, s11, s12)
		s2 = (s20, s21, s22)
		s = [s1,s2]
		sarc_list[kk][1] = s1
		sarc_list[kk+1][0] = s2
		
	return sarc_list
	
##########################################################################################
def displace_constant_value(sarc_list, x_disp, y_disp, z_disp):
	"""Displace every point in sarc_list by a small uniform nudge.""" 
	sarc_list_disp = [] 
	for kk in range(0,len(sarc_list)):
		s1 = sarc_list[kk][0] 
		s2 = sarc_list[kk][1]
		s1[0] += x_disp; s2[0] += x_disp
		s1[1] += y_disp; s2[1] += y_disp
		s1[2] += z_disp; s2[2] += z_disp
		s = [s1, s2]
		sarc_list_disp.append(s)
		
	return sarc_list_disp

##########################################################################################
##########################################################################################
# make alterations to a baseline geometry -- over all frames of the movie 
# sarc_list_ALL is a list of all sarc_list -- one per movie frame 
##########################################################################################
##########################################################################################

##########################################################################################
def sarc_list_ALL_transform_F( sarc_list, val_list, x0, y0, z0, x_zone_1, x_zone_2, F_fcn):
	"""Transform every sarcomere in sarc_list to sarc_list_ALL following val_list and F_fcn."""
	num_frames = len(val_list)
	sarc_list_ALL = [] 
	for kk in range(0,num_frames):
		val = val_list[kk]
		sarc_list2 = transform( sarc_list, F_fcn, val, x0, y0, z0, x_zone_1, x_zone_2)
		sarc_list_ALL.append(sarc_list2)
		
	return sarc_list_ALL

##########################################################################################
def sarc_list_ALL_displace_nudge( sarc_list_ALL, is_normal, random_parameter_x, random_parameter_y, random_parameter_z):
	"""Displace every point in every frame of sarc_list_ALL by a nudge."""
	sarc_list_ALL_nudge = [] 
	for step in range(0,len(sarc_list_ALL)):
			sarc_list = sarc_list_ALL[step]
			sarc_list = displace_random_nudge( sarc_list, is_normal, random_parameter_x, random_parameter_y, random_parameter_z)
			sarc_list_ALL_nudge.append(sarc_list)
			
	sarc_list_ALL = sarc_list_ALL_nudge
	return sarc_list_ALL

##########################################################################################
##########################################################################################
# organizational functions -- plot and save information
##########################################################################################
##########################################################################################

##########################################################################################
def get_len(p_1,p_2):
	"""Get the length of a sarcomere from end points."""
	return ((p_1[0]-p_2[0])**2.0 + (p_1[1]-p_2[1])**2.0 + (p_1[2]-p_2[2])**2.0)**(1.0/2.0)

##########################################################################################
def get_pos(p_1,p_2):
	"""Get the position of a sarcomere from end points."""
	x = 0.5*p_1[0] + 0.5*p_2[0]
	y = 0.5*p_1[1] + 0.5*p_2[1]
	return x, y

##########################################################################################
def pickle_sarc_list_ALL(folder_name,sarc_list_ALL):
	"""Pickle sarc_list_ALL to save it for rendering."""
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)
	sarc_fname = folder_name + '/sarc_list_ALL.pkl'
	with open(sarc_fname, 'wb') as f:
		pickle.dump(sarc_list_ALL,f)
	return

##########################################################################################
def get_ground_truth(sarc_list_ALL):
	"""Get the ground truth in a format that can be compared to tracking results."""
	# re-arrange sarc_list_ALL into len_list_ALL
	len_list_ALL = []
	x_list_ALL = [] 
	y_list_ALL = [] 
	for jj in range(0,len(sarc_list_ALL)):
		sarc_list = sarc_list_ALL[jj]
		len_list = []
		x_list =[] 
		y_list = [] 
		for kk in range(0,len(sarc_list)):
			p_1 = sarc_list[kk][0]
			p_2 = sarc_list[kk][1]
			sarc_len = get_len(p_1,p_2)
			len_list.append(sarc_len)
			x,y = get_pos(p_1,p_2)
			x_list.append(x)
			y_list.append(y)
			
		len_list_ALL.append(len_list)
		x_list_ALL.append(x_list)
		y_list_ALL.append(y_list)
		
	# re-arrange by sarcomere 
	len_list_by_sarc = [] 
	x_pos_array = [] 
	y_pos_array = [] 
	num_sarc = len(len_list)
	for kk in range(0,num_sarc):
		len_list = [] 
		x_list = [] 
		y_list = [] 
		for jj in range(0,len(len_list_ALL)):
			len_list.append(len_list_ALL[jj][kk])
			x_list.append(x_list_ALL[jj][kk])
			y_list.append(y_list_ALL[jj][kk])
			
		len_list_by_sarc.append(len_list)
		x_pos_array.append(x_list)
		y_pos_array.append(y_list)
		
	# make into an array
	sarc_array = np.asarray(len_list_by_sarc)
	x_pos_array = np.asarray(x_pos_array)
	y_pos_array = np.asarray(y_pos_array)
	# make normalized lengths array
	sarc_array_normalized = np.zeros(sarc_array.shape)
	num_sarc = sarc_array_normalized.shape[0]
	for kk in range(0,num_sarc):
		sarc_array_normalized[kk,:] = (sarc_array[kk,:]-np.mean(sarc_array[kk,:]))/np.mean(sarc_array[kk,:])
	
	return sarc_array, sarc_array_normalized, x_pos_array, y_pos_array

##########################################################################################
def plot_3D_geom(folder_name,sarc_list,zm,sm):
	"""Make a 3D plot of the ground truth from sarc_list."""
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	
	for kk in range(0,len(sarc_list)):
		pt_1 = sarc_list[kk][0]
		pt_2 = sarc_list[kk][1]
		ax.scatter(pt_1[0],pt_1[1],pt_1[2],zm)
		ax.scatter(pt_2[0],pt_2[1],pt_2[2],zm)
		ax.plot([pt_1[0], pt_2[0]],[pt_1[1], pt_2[1]],[pt_1[2],pt_2[2]], sm)
	
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)
	plt.savefig(folder_name + '/ground_truth_geom.png')
	return

##########################################################################################
def plot_ground_truth_timeseries(sarc_array_normalized, folder_name):
	"""Plot all of the ground truth timeseries."""
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)
	plt.figure()
	num_sarc = sarc_array_normalized.shape[0]
	for kk in range(0,num_sarc):
		plt.plot(sarc_array_normalized[kk,:])
	
	sarc_array_mean = np.mean(sarc_array_normalized,axis=0)
	plt.plot(sarc_array_mean,'k--',linewidth=3)
	plt.xlabel('frame')
	plt.ylabel('normalized sarc length')
	plt.title('num sarc: %i'%(num_sarc))
	plt.savefig(folder_name + '/ground truth time series')
	return





