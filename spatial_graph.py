import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os 
import pickle
import sys
import glob

##########################################################################################
# Input info and set up 
##########################################################################################
def create_spatial_graph(folder_name,include_eps=False):
	"""Create spatial graph."""
	num_frames = len(glob.glob('ALL_MOVIES_MATRICES/' + folder_name + '_matrices/*.npy'))

	external_folder_name = 'ALL_MOVIES_PROCESSED'
	if not os.path.exists(external_folder_name): os.makedirs(external_folder_name)

	out_graph =  external_folder_name + '/' + folder_name + '/graph'
	if not os.path.exists(external_folder_name + '/' + folder_name): os.makedirs(external_folder_name + '/' + folder_name)
	if not os.path.exists(out_graph): os.makedirs(out_graph)

	##########################################################################################
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	# Set up graph 
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	##########################################################################################

	G=nx.Graph()

	disc_data_fname = external_folder_name + '/' + folder_name + '/tracking_results/tracking_results_zdisks.txt'
	disc_data_all = np.loadtxt(disc_data_fname)
	particle = disc_data_all[:,2]
	unique_particle = np.unique(particle).astype('int')

	pos = {}
	for kk in range(0,unique_particle.shape[0]):
		# compute mean position of the unique particle 
		idx = unique_particle[kk]
		all_instances = np.where(disc_data_all[:,2].astype('int') == idx)
		all_instances = all_instances[0].astype('int')
		y_mean = -1.0*np.mean(disc_data_all[all_instances,3])
		x_mean = np.mean(disc_data_all[all_instances,4])
		G.add_node(idx, xpos = x_mean, ypos = y_mean)
		pos.update({idx:(x_mean,y_mean)})

	sarc_data_fname = external_folder_name + '/' + folder_name + '/tracking_results/tracking_results_sarcomeres.txt'
	sarc_data_all = np.loadtxt(sarc_data_fname)

	# --> go though every frame 
	for frame in range(0,num_frames):
		# isolate disc_data in each frame
		idx_in_frame = np.where(disc_data_all[:,0] == frame)
		disc_data = disc_data_all[idx_in_frame[0],:]
	
		# isolate sarc_data in each frame 
		idx_in_frame = np.where(sarc_data_all[:,0] == frame)
		sarc_data = sarc_data_all[idx_in_frame[0],:]
	
		# go through every sarcomere
		for kk in range(0,sarc_data.shape[0]):
			# find each original ID
			ZLID1 = int(sarc_data[kk,5])
			ZLID2 = int(sarc_data[kk,6])
			# look for the ZLID in disc_data
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
		
			# add edge to the network -- if the edge already exists increment it's weight up by 1 
			if G.has_edge(ZGID1,ZGID2):
				G[ZGID1][ZGID2]['weight'] += 1
			else:
				G.add_edge(ZGID1, ZGID2, weight = 1 )

	# --> remove every edge that has weight less than 10% of frames
	weight_cutoff = np.floor(0.10*num_frames)
	edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
	for edge in edges:
		weight = G[edge[0]][edge[1]]['weight']
		if weight < weight_cutoff:
			G.remove_edge(edge[0],edge[1])

	# --> remove every node that is unconnected 
	isolated_nodes = list(nx.isolates(G))
	G.remove_nodes_from(isolated_nodes)

	##########################################################################################
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	# Plot graph and save output information  
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	##########################################################################################

	# https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html
	plt.figure(figsize=(15,15))
	edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())

	if num_frames == 1:
		nx.draw(G,pos,node_color='g',node_size=2100, width=5, edge_color=weights, edge_cmap = plt.cm.autumn)	
	else:
		nx.draw(G,pos,node_color='g',node_size=200, width=5, edge_color=weights, edge_cmap = plt.cm.coolwarm)		

	plt.axis('equal')
		
	plt.savefig(out_graph + '/basic_graph')
	if include_eps:
		plt.savefig(out_graph + '/basic_graph.eps')

	with open(out_graph + '/graph.pkl', 'wb') as f:
		pickle.dump(G, f)

	with open(out_graph + '/pos.pkl', 'wb') as f:
		pickle.dump(pos, f)

	with open(out_graph + '/graph.pkl', 'rb') as f:
		G = pickle.load(f)
	
	with open(out_graph + '/pos.pkl', 'rb') as f:
		pos = pickle.load(f)
