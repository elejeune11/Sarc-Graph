import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from scipy.ndimage import gaussian_filter, median_filter
from matplotlib.animation import FuncAnimation
##########################################################################################
# 	FUNCTIONS FOR RENDERING I.E. GO FROM GEOMETRY TO FINAL IMAGES
##########################################################################################

# function: z_disk_props
# function: sarc_list_in_slice_fcn
# function: return_x_y_z_mat
# function: point_in_cyl
# function: binary_box
# function: slice_to_matrix
# function: matrix_gaussian_blur_fcn
# function: matrix_median_blur_fcn
# function: random_val
# function: cloud_image
# function: matrix_to_image
# function: save_img_stil
# function: still_to_avi
# function: ground_truth_movie

##########################################################################################

##########################################################################################
def z_disk_props( sarc_list, is_normal_radius, is_normal_height, avg_radius, avg_height, parameter_radius, parameter_height):
	"""Create z disk properties, z disks are modeled as cylinders with radius R and height H. Once cylinder per sarcomere (s1)."""
	radius_list = []
	height_list = [] 
	for kk in range(0,len(sarc_list)):
		if is_normal_radius:
			rad = avg_radius + np.random.normal(0,parameter_radius)
		else:
			rad = avg_radius + (np.random.random(1)[0] - .5) * parameter_radius * 2.0
		if is_normal_height:
			hei = avg_height + np.random.normal(0,parameter_height)
		else:
			hei = avg_height + (np.random.random(1)[0] - .5) * parameter_height * 2.0
			
		radius_list.append(rad)
		height_list.append(hei)
		
	return radius_list, height_list

##########################################################################################
def sarc_list_in_slice_fcn(sarc_list, radius_list, height_list, z_lower, z_upper): 
	"""Check to see if sarcomere is within a slice in the z dimension."""
	sarc_list_in_slice = [] 
	radius_list_in_slice = [] 
	height_list_in_slice = [] 
	num_sarc = len(sarc_list)
	for kk in range(0,num_sarc):
		z = 0.5*( sarc_list[kk][0][2] + sarc_list[kk][1][2] )
		if z > z_lower and z < z_upper:
			sarc_list_in_slice.append(sarc_list[kk])
			radius_list_in_slice.append(radius_list[kk])
			height_list_in_slice.append(height_list[kk])
			
	return sarc_list_in_slice, radius_list_in_slice, height_list_in_slice

##########################################################################################
def return_x_y_z_mat(matrix, x_lower, x_upper, y_lower, y_upper, z_lower, z_upper):
	"""Helper function that returns the X, Y, and Z coordinates of a matrix."""
	matrix_X = np.zeros(matrix.shape)
	matrix_Y = np.zeros(matrix.shape)
	matrix_Z = np.zeros(matrix.shape)
	num_x = matrix.shape[0]
	num_y = matrix.shape[1]
	num_z = matrix.shape[2]
	for ii in range(0,num_x):
		for jj in range(0,num_y):
			for kk in range(0,num_z):
				matrix_X[ii,jj,kk] = ii / num_x * (x_upper - x_lower) + x_lower
				matrix_Y[ii,jj,kk] = jj / num_y * (y_upper - y_lower) + y_lower
				matrix_Z[ii,jj,kk] = kk / num_z * (z_upper - z_lower) + z_lower
				
	return matrix_X, matrix_Y, matrix_Z

##########################################################################################
def point_in_cyl(pt_x,pt_y,pt_z,cyl_p1,cyl_p2,cyl_rad):
	"""Helper function that returns 1 if a point is inside a cylinder, 0 otherwise."""
	q = np.asarray([pt_x,pt_y,pt_z])
	p1 = np.asarray([cyl_p1[0],cyl_p1[1],cyl_p1[2]])
	p2 = np.asarray([cyl_p2[0],cyl_p2[1],cyl_p2[2]])
	check_1 = np.dot(q-p1,p2-p1)
	check_2 = np.dot(q-p2,p2-p1)
	if check_1 >=0 and check_2 <= 0:
		rad = np.linalg.norm(np.cross( q-p1, p2-p1 )) / np.linalg.norm(p2-p1)
		if rad <= cyl_rad:
			return 1 
		else:
			return 0
	else:
		return 0 

##########################################################################################
def binary_box(matrix_X,matrix_Y,matrix_Z,cyl_p1,cyl_p2,cyl_rad): 
	"""Helper function that returns a binary matrix if the point is inside the cylinder.""" 
	num_x = matrix_X.shape[0]
	num_y = matrix_Y.shape[1]
	num_z = matrix_Z.shape[2]
	bin_box = np.zeros((num_x,num_y,num_z))	
	for ii in range(0,num_x):
		for jj in range(0,num_y):
			for kk in range(0,num_z):
				x = matrix_X[ii,jj,kk]
				y = matrix_Y[ii,jj,kk]
				z = matrix_Z[ii,jj,kk]
				bin_box[ii,jj,kk] = point_in_cyl(x,y,z,cyl_p1,cyl_p2,cyl_rad)
				
	return bin_box

##########################################################################################
def slice_to_matrix(sarc_list,dim_x,dim_y,dim_z,x_lower,x_upper,y_lower,y_upper,z_lower,z_upper, mean_rad, mean_hei, bound_x, bound_y, bound_z, val):
	"""Create a 3D matrix where each sarcomere is represented as voxels."""
	matrix = np.zeros((dim_x,dim_y,dim_z))
	matrix_X, matrix_Y, matrix_Z = return_x_y_z_mat(matrix, x_lower, x_upper, y_lower, y_upper, z_lower, z_upper)
	# for each, only add s1 (adding s2 would be redundant)
	num_sarc = len(sarc_list)
	for kk in range(0,num_sarc):
		s1 = sarc_list[kk][0]
		s1 = np.asarray([s1[0],s1[1],s1[2]])
		s2 = sarc_list[kk][1]
		s2 = np.asarray([s2[0],s2[1],s2[2]])
		vec = (s2 - s1) / np.linalg.norm(s2-s1)
		rad = mean_rad[kk]
		hei = mean_hei[kk]
		p1 = s1 + vec * hei/2.0
		p2 = s1 - vec * hei/2.0
		
		cent_x = int((s1[0]  - x_lower)/(x_upper-x_lower) * dim_x)
		cent_y = int((s1[1]  - y_lower)/(y_upper-y_lower) * dim_y)
		cent_z = int((s1[2]  - z_lower)/(z_upper-z_lower) * dim_z)
		
		lower_x = np.max([cent_x - bound_x, 0])
		upper_x = np.min([cent_x + bound_x, dim_x-1])
		lower_y = np.max([cent_y - bound_y, 0])
		upper_y = np.min([cent_y + bound_y, dim_y-1])
		lower_z = np.max([cent_z - bound_z, 0])
		upper_z = np.min([cent_z + bound_z, dim_z-1])
				
		mm_x = matrix_X[lower_x:upper_x,lower_y:upper_y,lower_z:upper_z]
		mm_y = matrix_Y[lower_x:upper_x,lower_y:upper_y,lower_z:upper_z]
		mm_z = matrix_Z[lower_x:upper_x,lower_y:upper_y,lower_z:upper_z]
		
		bin_box = binary_box(mm_x,mm_y,mm_z,p1,p2,rad)
		matrix[lower_x:upper_x,lower_y:upper_y,lower_z:upper_z] += bin_box*val
		
		if kk == num_sarc - 1:
			s1 = sarc_list[kk][0]
			s1 = np.asarray([s1[0],s1[1],s1[2]])
			s2 = sarc_list[kk][1]
			s2 = np.asarray([s2[0],s2[1],s2[2]])
			vec = (s2 - s1) / np.linalg.norm(s2-s1)
			rad = mean_rad[kk]
			hei = mean_hei[kk]
			p1 = s2 + vec * hei/2.0
			p2 = s2 - vec * hei/2.0
		
			cent_x = int((s1[0]  - x_lower)/(x_upper-x_lower) * dim_x)
			cent_y = int((s1[1]  - y_lower)/(y_upper-y_lower) * dim_y)
			cent_z = int((s1[2]  - z_lower)/(z_upper-z_lower) * dim_z)
		
			lower_x = np.max([cent_x - bound_x, 0])
			upper_x = np.min([cent_x + bound_x, dim_x-1])
			lower_y = np.max([cent_y - bound_y, 0])
			upper_y = np.min([cent_y + bound_y, dim_y-1])
			lower_z = np.max([cent_z - bound_z, 0])
			upper_z = np.min([cent_z + bound_z, dim_z-1])
				
			mm_x = matrix_X[lower_x:upper_x,lower_y:upper_y,lower_z:upper_z]
			mm_y = matrix_Y[lower_x:upper_x,lower_y:upper_y,lower_z:upper_z]
			mm_z = matrix_Z[lower_x:upper_x,lower_y:upper_y,lower_z:upper_z]
		
			bin_box = binary_box(mm_x,mm_y,mm_z,p1,p2,rad)
			matrix[lower_x:upper_x,lower_y:upper_y,lower_z:upper_z] += bin_box*val
		
	return matrix

##########################################################################################
def matrix_gaussian_blur_fcn(matrix,sig):
	"""Function to apply gaussian blur to the matrix that represents sarcomeres as voxels."""
	matrix_blur = gaussian_filter(matrix, sigma=sig)
	return  matrix_blur

##########################################################################################
def matrix_median_blur_fcn(matrix,size):
	"""Function to apply median blur to the matrix that represents sarcomeres as voxels."""
	matrix_blur = median_filter(matrix_blur, size=size)
	return  matrix_blur

##########################################################################################
def random_val(matrix,mean,std):
	"""Function to apply normally distributed random noise to the matrix that represents sarcomeres as voxels."""
	mat = np.random.normal(mean,std,matrix.shape)
	matrix += mat
	return matrix 

##########################################################################################
def cloud_image(a,b,x0,y0,matrix,val):
	for ii in range(0,matrix.shape[0]):
		for jj in range(0,matrix.shape[1]):
			for kk in range(0,matrix.shape[2]):
				if ((ii-x0)/a)**2.0 + ((jj - y0)/b)**2.0 < 1:
					matrix[ii,jj,kk] += val*10
					
	return matrix
	
##########################################################################################
def matrix_to_image(matrix,slice_lower,slice_upper):
	"""Convert the 3D matrix into a projected 2D image matrix."""
	matrix = matrix[:,:,slice_lower:slice_upper]
	image = np.sum(matrix,axis=2)
	return image

##########################################################################################
def save_img_stills(image_list,folder_name):
	"""Save image stills with correct matplotlib settings."""
	folder_name_render = folder_name + '/render'
	if not os.path.exists(folder_name_render):
		os.makedirs(folder_name_render)
	num_images = len(image_list)
	for step in range(0,num_images):
		image = image_list[step]
		plt.figure()
		plt.imshow(image)
		plt.axis('off')
		ax = plt.gca()
		ax.set_xticks([]); ax.set_yticks([])
		if step < 10:
			plt.savefig(folder_name_render + '/frame_00%i.png'%(step),bbox_inches = 'tight',transparent=True,pad_inches = 0)
		elif step < 100:
			plt.savefig(folder_name_render + '/frame_0%i.png'%(step),bbox_inches = 'tight',transparent=True,pad_inches = 0)
		else:
			plt.savefig(folder_name_render + '/frame_%i.png'%(step),bbox_inches = 'tight',transparent=True,pad_inches = 0)
		plt.close()
		
	return

##########################################################################################
def still_to_avi(folder_name,num_frames,is_GT):
	"""Convert still images to an avi."""
	folder_name_render = folder_name + '/render'
	if is_GT == True:
		video_name = folder_name + '/ground_truth_movie/GT_' + folder_name + '.avi' 
	else:
		video_name = folder_name + '/' + folder_name + '.avi'
	img_list = [] 
	for kk in range(0,num_frames):
		if kk < 10:
			fname = 'frame_00%i.png'%(kk)
		elif kk < 100:
			fname = 'frame_0%i.png'%(kk)
		else:
			fname = 'frame_%i.png'%(kk)
		img_list.append(fname)
	
	images = [img for img in img_list]
	
	if is_GT == True:
		frame = cv2.imread(os.path.join(folder_name + '/ground_truth_movie', images[0]))
	else:
		frame = cv2.imread(os.path.join(folder_name + '/render', images[0]))
		
	height, width, layers = frame.shape

	video = cv2.VideoWriter(video_name, 0, 30, (width,height))

	for image in images:
		if is_GT == True:
			video.write(cv2.imread(os.path.join(folder_name + '/ground_truth_movie', image)))
		else:
			video.write(cv2.imread(os.path.join(folder_name + '/render', image)))
	
	cv2.destroyAllWindows()
	video.release()
	return 

##########################################################################################
def ground_truth_movie(folder_name,num_frames,img_list,sarc_array_normalized, x_pos_array, y_pos_array,x_lower,x_upper,y_lower,y_upper,dim_x,dim_y):
	"""Make the ground truth movie from the geometry."""
	folder_name_GT = folder_name + '/ground_truth_movie'
	if not os.path.exists(folder_name_GT):
		os.makedirs(folder_name_GT)
	all_normalized = sarc_array_normalized
	color_matrix = np.zeros(all_normalized.shape)
	for kk in range(0,all_normalized.shape[0]):
		for jj in range(0,all_normalized.shape[1]):
			of = all_normalized[kk,jj]
			if of < -.2:
				color_matrix[kk,jj] = 0
			elif of > .2:
				color_matrix[kk,jj] = 1
			else:
				color_matrix[kk,jj] = of*2.5 + .5
	
	for t in range(0,num_frames):
		img = img_list[t]
		
		plt.figure()
		plt.imshow(img)
		for kk in range(0,all_normalized.shape[0]):
			col = (1 - color_matrix[kk,t], 0, color_matrix[kk,t])
			yy = (y_pos_array[kk,t] - y_lower)/(y_upper-y_lower)*dim_y
			xx = (x_pos_array[kk,t] - x_lower)/(x_upper-x_lower)*dim_x
			plt.plot(yy,xx,'.',c=col)
			
		ax = plt.gca()
		ax.set_xticks([]); ax.set_yticks([])
		plt.axis('off')
		if t < 10:
			plt.savefig(folder_name_GT + '/frame_00%i.png'%(t),bbox_inches = 'tight',transparent=True,pad_inches = 0)
		elif t < 100:
			plt.savefig(folder_name_GT + '/frame_0%i.png'%(t),bbox_inches = 'tight',transparent=True,pad_inches = 0)
		else:
			plt.savefig(folder_name_GT + '/frame_%i.png'%(t),bbox_inches = 'tight',transparent=True,pad_inches = 0)
		plt.close()
	return
	
	