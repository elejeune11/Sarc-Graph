import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os 
import pickle
import sys
import glob
from scipy.linalg import polar
from numpy import linalg as LA
import moviepy.editor as mp
import imageio
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
def compute_frame_OOP(folder_name,frame_num):
	"""Given a specific frame, compute Orientation Order Parameter (OOP) of the frame."""
	num_frames = len(glob.glob('ALL_MOVIES_MATRICES/' + folder_name + '_matrices/*.npy'))
	out_file = 'ALL_MOVIES_PROCESSED' + '/' + folder_name + '/segmented_sarc'
	ang = [] 
	dat_fname = out_file + '/frame-%04d_sarc_data.txt'%(frame_num)
	dat = np.loadtxt(dat_fname)
	ang_dat = dat[:,6]
	for jj in range(0,ang_dat.shape[0]):
		val = ang_dat[jj]
		ang.append(val)
	
	mat = np.zeros((2,2))
	for kk in range(0,len(ang)):
		x = np.cos(ang[kk])
		y = np.sin(ang[kk])
		vec = np.asarray([x,y])
		n = np.outer(vec,vec)
		mat += 2.0*n - np.asarray([[1,0],[0,1]])
	
	mat = mat / len(ang)
	
	u, v = np.linalg.eig(mat)
	
	OOP = np.max(u)
	OOP_vec = v[:,np.argmax(u)]
	
	return OOP, OOP_vec

##########################################################################################
def compute_frame_F(folder_name,frame_0,frame_t):
	"""Compute the average deformation gradient given frame 0 and current frame."""
	x_pos = np.loadtxt('ALL_MOVIES_PROCESSED/' + folder_name + '/timeseries/tracking_results_x_pos.txt')
	y_pos = np.loadtxt('ALL_MOVIES_PROCESSED/' + folder_name + '/timeseries/tracking_results_y_pos.txt')
	
	num_sarc = x_pos.shape[0]
	num_time = x_pos.shape[1]
	num_vec = int((num_sarc * num_sarc - num_sarc) / 2.0)
	
	Lambda_0 = np.zeros((2,num_vec))
	ix = 0
	for kk in range(0,num_sarc):
		for jj in range(kk+1,num_sarc):
			x_vec = x_pos[kk,frame_0] - x_pos[jj,frame_0]
			y_vec = y_pos[kk,frame_0] - y_pos[jj,frame_0]
			Lambda_0[0,ix] = x_vec
			Lambda_0[1,ix] = y_vec 
			ix += 1 
			
	Lambda_t = np.zeros((2,num_vec))
	ix = 0
	for kk in range(0,num_sarc):
		for jj in range(kk+1,num_sarc):
			x_vec = x_pos[kk,frame_t] - x_pos[jj,frame_t]
			y_vec = y_pos[kk,frame_t] - y_pos[jj,frame_t]
			Lambda_t[0,ix] = x_vec
			Lambda_t[1,ix] = y_vec 
			ix += 1 
	
	term_1 = np.dot( Lambda_t , np.transpose(Lambda_0) )
	term_2 = np.linalg.inv( np.dot( Lambda_0 , np.transpose(Lambda_0) ) )
	F = np.dot(term_1 , term_2)
	J = F[0,0]*F[1,1] - F[0,1]*F[1,0]
	
	return F, J


##########################################################################################
def compute_all_OOP(folder_name):
	"""Compute OOP for every frame."""
	num_frames = len(glob.glob('ALL_MOVIES_MATRICES/' + folder_name + '_matrices/*.npy'))
	OOP_list = []; OOP_vec_list = [] 
	for kk in range(0,num_frames):
		OOP, OOP_vec = compute_frame_OOP(folder_name,kk)
		OOP_list.append(OOP)
		OOP_vec_list.append(OOP_vec)
	
	return OOP_list, OOP_vec_list

##########################################################################################
def compute_all_F(folder_name, reference_frame):
	"""Compute F and J for every frame."""
	num_frames = len(glob.glob('ALL_MOVIES_MATRICES/' + folder_name + '_matrices/*.npy'))
	F_list = []; J_list = []
	for kk in range(0,num_frames):
		F, J = compute_frame_F(folder_name,reference_frame,kk)
		F_list.append(F)
		J_list.append(J)
		
	return F_list, J_list

##########################################################################################
def compute_all_F_adjusted(folder_name):
	"""Compute F and J for every frame. Reference frame is most relaxed frame."""
	F_list, J_list = compute_all_F(folder_name, 0)
	reference_frame = np.argmax(J_list)
	F_list, J_list = compute_all_F(folder_name, reference_frame)
	return F_list, J_list, reference_frame

##########################################################################################
def visualize_OOP_and_F_timeseries(OOP_list,J_list,folder_name):
	"""Plot timeseries."""
	external_folder_name = 'ALL_MOVIES_PROCESSED'
	out_analysis = external_folder_name + '/' + folder_name + '/analysis'
	plt.figure()
	plt.subplot(1,2,1)
	plt.plot(OOP_list)
	plt.xlabel('frame number')
	plt.ylabel('OOP')
	plt.tight_layout()
	plt.subplot(1,2,2)
	plt.plot(J_list)
	plt.xlabel('frame number')
	plt.ylabel('average deformation J')
	plt.tight_layout()
	plt.savefig(out_analysis + '/OOP_J_timeseries')
	return 

##########################################################################################
def visualize_OOP_and_F_on_image(folder_name, frame_num, F_list, OOP_vec_list, OOP_list):
	"""Plot the OOP and F visualize don the image"""
	external_folder_name = 'ALL_MOVIES_PROCESSED'
	out_analysis = external_folder_name + '/' + folder_name + '/analysis'
	F = F_list[frame_num]
	J = F[0,0]*F[1,1] - F[0,1]*F[1,0]
	R, U = polar(F)
	w, v = LA.eig(U)
	v = np.dot(R, v)
	vec_1 = v[:,np.argmin(w)]
	vec_2 = v[:,np.argmax(w)]
	raw_img = get_frame_matrix(folder_name, frame_num)
	x_pos_mean = raw_img.shape[0]/2.0; y_pos_mean = raw_img.shape[1]/2.0
	plt.figure(figsize=(5,5))
	plt.imshow(raw_img, cmap=plt.cm.gray)
	rad = .2*np.min([raw_img.shape[0],raw_img.shape[1]]); th = np.linspace(0,2.0*np.pi,100)
	plt.plot([y_pos_mean-rad*vec_1[1],y_pos_mean+rad*vec_1[1]],[x_pos_mean-rad*vec_1[0],x_pos_mean+rad*vec_1[0]],'-',color=(255/255,204/255,203/255),linewidth=0.3)
	plt.plot([y_pos_mean-rad*vec_2[1],y_pos_mean+rad*vec_2[1]],[x_pos_mean-rad*vec_2[0],x_pos_mean+rad*vec_2[0]],'-',color=(0.5,0.5,0.5),linewidth=0.3)
	
	x_vec = []; y_vec = [] ; x_vec_circ = []; y_vec_circ = [] 
	scale = np.asarray([[.9,0],[0,.9]])
	for jj in range(0,100):
		v = np.asarray([rad*np.cos(th[jj]),rad*np.sin(th[jj])])
		#v_def = np.dot(np.dot(F_list_mat[jj],scale),v)
		nest1 = np.dot(F,F); nest2 = np.dot(F,nest1); nest3 = np.dot(F,nest2)
		nest4 = np.dot(F,nest3); nest5 = np.dot(F,nest4); nest6 = np.dot(F,nest5)
		nest7 = np.dot(F,nest6); nest8 = np.dot(F,nest7)
		v_def = np.dot(nest8,v)
		x_vec.append(v_def[0] + x_pos_mean); y_vec.append(v_def[1] + y_pos_mean)
		x_vec_circ.append(x_pos_mean + v[0]); y_vec_circ.append(y_pos_mean + v[1])
	
	plt.plot(y_vec_circ,x_vec_circ,'-',color=(255/255,204/255,203/255),linewidth=0.3)
	plt.plot(y_vec,x_vec,'-',color=(255/255,204/255,203/255),linewidth=1.0)
	
	OOP_vec = OOP_vec_list[frame_num]
	rad_OOP = rad*OOP_list[frame_num]
	plt.plot([y_pos_mean - rad_OOP*OOP_vec[1],y_pos_mean + rad_OOP*OOP_vec[1]],[x_pos_mean - rad_OOP*OOP_vec[0],x_pos_mean + rad_OOP*OOP_vec[0]],'r-',linewidth=5)
	
	plt.title('J: %.3f, OOP:%.3f, frame: %i'%(J,OOP_list[frame_num],frame_num))
	ax = plt.gca()
	ax.set_xticks([]); ax.set_yticks([]);
	plt.savefig(out_analysis + '/OOP_J_on_img')
	return 

##########################################################################################
def compute_s(y_vec):
	y_max = np.max(y_vec)
	y_min = np.min(y_vec)
	s = (y_max - y_min) / (y_max + 1)
	return s

##########################################################################################


##########################################################################################
def compute_s_median(y_mat):
	s_list = [] 
	for kk in range(0,y_mat.shape[0]):
		s = compute_s(y_mat[kk,:])
		s_list.append(s)
	return np.median(s_list), s_list

##########################################################################################
def compute_shortening(folder_name):
	"""Compute \bar{s} and s_avg, two measures of sarcomere shortening."""
	external_folder_name = 'ALL_MOVIES_PROCESSED/'
	out_analysis = external_folder_name + '/' + folder_name + '/analysis'
	# timeseries data
	fname_leng = external_folder_name + folder_name + '/timeseries/tracking_results_leng.txt'
	dat_leng = np.loadtxt(fname_leng)
	dat_avg = np.mean(dat_leng,axis=0)
	s_til, s_list = compute_s_median(dat_leng)
	s_avg = compute_s(dat_avg)
	np.savetxt(out_analysis + '/s_til.txt', np.asarray([s_til]))
	np.savetxt(out_analysis + '/s_avg.txt', np.asarray([s_avg]))
	return s_til, s_avg, s_list


##########################################################################################
def compute_metrics(folder_name):
	"""Compute metrics, OOP, Ciso and C||."""
	external_folder_name = 'ALL_MOVIES_PROCESSED'
	out_analysis = external_folder_name + '/' + folder_name + '/analysis'
	
	F_list, J_list, reference_frame = compute_all_F_adjusted(folder_name)
	with open(out_analysis + '/F_list.pkl', 'wb') as f:
		pickle.dump(F_list, f)
	with open(out_analysis + '/J_list.pkl', 'wb') as f:
		pickle.dump(J_list, f)
		
	OOP_list, OOP_vec_list = compute_all_OOP(folder_name)
	with open(out_analysis + '/OOP_list.pkl', 'wb') as f:
		pickle.dump(OOP_list, f)
	with open(out_analysis + '/OOP_vec_list.pkl', 'wb') as f:
		pickle.dump(OOP_vec_list, f)
	
	max_contract_frame = np.argmin(J_list)
	visualize_OOP_and_F_timeseries(OOP_list,J_list,folder_name)
	visualize_OOP_and_F_on_image(folder_name,max_contract_frame, F_list, OOP_vec_list, OOP_list)
	selected_frame = np.argmin(J_list)
	OOP_selected = OOP_list[selected_frame]
	J = J_list[selected_frame]
	F = F_list[selected_frame]
	avg_contract = 1.0 - np.sqrt(J)
	v = OOP_vec_list[selected_frame]
	v0 = np.dot(np.linalg.inv(F),v)
	v_abs = np.sqrt((v[0])**2.0 + (v[1])**2.0)
	v0_abs = np.sqrt((v0[0])**2.0 + (v0[1])**2.0)
	avg_aligned_contract = (v0_abs - v_abs)/v0_abs
	
	s_til, s_avg, s_list = compute_shortening(folder_name)
	
	np.savetxt(out_analysis + '/OOP.txt', np.asarray([OOP_selected]))
	np.savetxt(out_analysis + '/C_iso.txt',np.asarray([avg_contract]))
	np.savetxt(out_analysis + '/C_OOP.txt',np.asarray([avg_aligned_contract]))
	np.savetxt(out_analysis + '/s_til.txt',np.asarray([s_til]))
	np.savetxt(out_analysis + '/s_avg.txt',np.asarray([s_avg]))
	
	return OOP_selected, avg_contract, avg_aligned_contract, s_til, s_avg

##########################################################################################
def compute_metrics_load_state(folder_name):
	"""Compute metrics, OOP, Ciso and C||. Start from loaded """
	external_folder_name = 'ALL_MOVIES_PROCESSED'
	out_analysis = external_folder_name + '/' + folder_name + '/analysis'
	
	with open(out_analysis + '/F_list.pkl', 'rb') as f: F_list = pickle.load(f)
	with open(out_analysis + '/J_list.pkl', 'rb') as f: J_list = pickle.load(f)
	with open(out_analysis + '/OOP_list.pkl', 'rb') as f: OOP_list = pickle.load(f)
	with open(out_analysis + '/OOP_vec_list.pkl', 'rb') as f: OOP_vec_list = pickle.load(f)
	
	max_contract_frame = np.argmin(J_list)
	visualize_OOP_and_F_timeseries(OOP_list,J_list,folder_name)
	visualize_OOP_and_F_on_image(folder_name,max_contract_frame, F_list, OOP_vec_list, OOP_list)
	selected_frame = np.argmin(J_list)
	OOP_selected = OOP_list[selected_frame]
	J = J_list[selected_frame]
	F = F_list[selected_frame]
	avg_contract = 1.0 - np.sqrt(J)
	v = OOP_vec_list[selected_frame]
	v0 = np.dot(np.linalg.inv(F),v)
	v_abs = np.sqrt((v[0])**2.0 + (v[1])**2.0)
	v0_abs = np.sqrt((v0[0])**2.0 + (v0[1])**2.0)
	avg_aligned_contract = (v0_abs - v_abs)/v0_abs
	
	np.savetxt(out_analysis + '/OOP.txt', np.asarray([OOP_selected]))
	np.savetxt(out_analysis + '/C_iso.txt',np.asarray([avg_contract]))
	np.savetxt(out_analysis + '/C_OOP.txt',np.asarray([avg_aligned_contract]))
	
	return OOP_selected, avg_contract, avg_aligned_contract
	
##########################################################################################
def visualize_lambda_as_functional_metric(folder_name, include_eps=False):
	"""Plot lambda 1 and lambda 2 along with a movie of the cell deforming with tracked sarcomeres marked."""
	external_folder_name = 'ALL_MOVIES_PROCESSED/'
	out_analysis = external_folder_name + '/' + folder_name + '/analysis'
	# timeseries data
	fname_leng = external_folder_name + folder_name + '/timeseries/tracking_results_leng.txt'
	dat_leng = np.loadtxt(fname_leng)
	avg_leng = np.mean(dat_leng,axis=0)
	##########################################################################################
	plot_info_frames_fname = 'ALL_MOVIES_PROCESSED/' + folder_name + '/timeseries/' + 'plotting_all_frames.pkl'
	ALL_frames_above_thresh = pickle.load( open( plot_info_frames_fname  , "rb" ) )
	plot_info_x_pos_fname = 'ALL_MOVIES_PROCESSED/' + folder_name + '/timeseries/' + 'plotting_all_x.pkl'
	ALL_x_pos_above_thresh = pickle.load( open( plot_info_x_pos_fname  , "rb" ) )
	plot_info_y_pos_fname = 'ALL_MOVIES_PROCESSED/' + folder_name + '/timeseries/' + 'plotting_all_y.pkl'
	ALL_y_pos_above_thresh = pickle.load( open( plot_info_y_pos_fname  , "rb" ) )
	sarc_data_normalized_fname = 'ALL_MOVIES_PROCESSED/' + folder_name + '/timeseries/' + 'tracking_results_leng.txt'
	all_normalized = np.loadtxt(sarc_data_normalized_fname)
	color_matrix = np.zeros(all_normalized.shape)
	for kk in range(0,all_normalized.shape[0]):
		for jj in range(0,all_normalized.shape[1]):
			of = all_normalized[kk,jj]
			if of < -.1: color_matrix[kk,jj] = 0
			elif of > .1: color_matrix[kk,jj] = 1
			else: color_matrix[kk,jj] = of*5 + .5
	##########################################################################################
	out_plots = out_analysis + '/summary_plot'
	if not os.path.exists(out_plots): os.makedirs(out_plots)

	# F data
	F_list = np.loadtxt(external_folder_name + '/' + folder_name + '/analysis/recovered_F.txt')
	num_frames = F_list.shape[0]; x = [] 
	lambda_1_list = []; vec_1_list = [] 
	lambda_2_list = []; vec_2_list = [] 
	J_list = []; F_list_mat = [] 
	for kk in range(0,num_frames):
		F00 = F_list[kk,0]; F01 = F_list[kk,1]; F10 = F_list[kk,2]; F11 = F_list[kk,3]
		J_list.append(F00*F11 - F01*F10)
		x.append(kk)
		R, U = polar(np.asarray([[F00,F01],[F10,F11]]))
		w, v = LA.eig(U)
		lambda_1_list.append(np.min(w)); lambda_2_list.append(np.max(w))
		v = np.dot(R, v)
		vec_1_list.append(v[:,np.argmin(w)]); vec_2_list.append(v[:,np.argmax(w)])
		F_list_mat.append(np.asarray([[F00,F01],[F10,F11]]))

	##########################################################################################
	img_list = [] 

	for kk in range(0,num_frames):
		t = kk 
		if t < 10: file_root = '/frame-000%i'%(t)
		elif t < 100: file_root = '/frame-00%i'%(t)
		else: file_root = '/frame-0%i'%(t)
	
		fig = plt.figure(figsize=(10*.7,5*.7))
		gs = fig.add_gridspec(2,2)

		ax1 = fig.add_subplot(gs[:,0])

		raw_img = get_frame_matrix(folder_name, kk)
		x_pos_mean = raw_img.shape[0]/2.0; y_pos_mean = raw_img.shape[1]/2.0
		plt.imshow(raw_img, cmap=plt.cm.gray)

		##########################################################################################
		for zz in range(0,all_normalized.shape[0]):
			if kk in ALL_frames_above_thresh[zz]:
				ix = np.argwhere(np.asarray(ALL_frames_above_thresh[zz]) == kk)[0][0]
				col = (1-color_matrix[zz,kk], 0 , color_matrix[zz,kk])
				yy = ALL_y_pos_above_thresh[zz][ix]
				xx = ALL_x_pos_above_thresh[zz][ix]
				plt.scatter(yy,xx,s=3,color=col,marker='o')

		##########################################################################################

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
		##########################################################################################
		##########################################################################################
		ax = fig.add_subplot(gs[0,1])
		ax.set_title('average deformation')
		ax.plot(x,lambda_1_list,'-',color='k',linewidth=1,label='λ1')
		ax.plot(x,lambda_2_list,'-',color=(0.5,0.5,0.5),linewidth=1,label='λ2')
		ax.plot(x[kk],lambda_1_list[kk],'o',mfc=(.7,0,0),mec=(0,0,0),markersize=7)
		ax.plot(x[kk],lambda_2_list[kk],'o',mfc=(.7,0,0),mec=(0.5,0.5,0.5),markersize=7)
		ax.set_xlim((np.min(x)-2,np.max(x)+2))
		plt.legend(loc='upper right')
		#ax.set_ylabel('avg deformation')
	
		ax2 = fig.add_subplot(gs[1,1])
		#ax2.set_ylabel('sarc length')
		ax2.set_title('normalized sarcomere length')
		ax2.plot(dat_leng.T,linewidth=5/dat_leng.shape[0],color=(0.75,0.75,0.75),alpha=.75)
		ax2.plot(x,avg_leng,'-',color=(0,0,0),linewidth=1,label='mean')
		val = np.max(np.abs(avg_leng))
		ax2.set_ylim((-2*val,2*val))
		ax2.set_xlim((np.min(x)-2,np.max(x)+2))

		ax2.plot(x[kk],avg_leng[kk],'o',mfc=(.7,0,0),mec=(0,0,0),markersize=7)

		plt.xlabel('frame number')
		plt.legend(loc='upper right')
		plt.tight_layout()
	
		plt.savefig(out_plots + '/' + file_root + '_summary')
		if include_eps or kk == np.argmin(J_list):
			plt.savefig(out_plots + '/' + 'frame-%i'%(t) + '_summary.eps')
		plt.close()
		img_list.append(plt.imread(out_plots + '/' + file_root + '_summary.png'))


	imageio.mimsave(out_plots + '/summary.gif', img_list, loop = 10)

	clip = mp.VideoFileClip(out_plots + '/summary.gif')
	clip.write_videofile(out_plots + '/summary.mp4')
