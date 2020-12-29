import file_pre_processing as fpp
import segmentation as seg
import tracking as track
import spatial_graph as sg
import timeseries as ts
import analysis_tools as at
##########################################################################################
folder_name_list = ['synthetic_data_1','synthetic_data_2','synthetic_data_3','real_data_Sample1','real_data_Sample2']
##########################################################################################
for folder_name in folder_name_list:
	include_eps = False
	##########################################################################################
	##########################################################################################
	##########################################################################################
	# Convert the movie into a folder of .npy arrays, one for each frame 
	##########################################################################################
	fpp.file_pre_processing(folder_name)
	print(folder_name,"file pre processing complete")
	# ##########################################################################################
	# # Run segmentation
	# ##########################################################################################
	gaussian_filter_size = 1
	seg.segmentation_all(folder_name, gaussian_filter_size)
	print(folder_name,"segmentation complete")
	# ##########################################################################################
	# # Run tracking
	# ##########################################################################################
	tp_depth = 4
	track.run_all_tracking(folder_name,tp_depth)
	print(folder_name,"tracking complete")
	##########################################################################################
	# Create spatial graph
	##########################################################################################
	sg.create_spatial_graph(folder_name)
	print(folder_name,"spatial graph complete")
	##########################################################################################
	# Process timeseries 
	##########################################################################################
	keep_thresh = 0.75
	ts.timeseries_all(folder_name, keep_thresh)
	print(folder_name,"timeseries complete")
	##########################################################################################
	# Additional analysis 
	##########################################################################################
	# --> visualize segmentation
	gaussian_filter_size = 1
	at.visualize_segmentation(folder_name, gaussian_filter_size,include_eps)
	print(folder_name,"visualize segmentation complete")

	# # --> visualize contract anim movie 
	at.visualize_contract_anim_movie(folder_name,include_eps)
	print(folder_name,"visualize contract anim movie complete")

	# --> perform timeseries clustering 
	compute_dist_DTW = True; compute_dist_euclidean = False
	at.cluster_timeseries_plot_dendrogram(folder_name,compute_dist_DTW,compute_dist_euclidean)
	print(folder_name,"cluster timeseries complete")

	# # --> plot normalized tracked timeseries 
	at.plot_normalized_tracked_timeseries(folder_name,include_eps)
	print(folder_name,"plot tracked timeseries complete")

	# --> plot untracked absolute timeseries
	at.plot_untracked_absolute_timeseries(folder_name,include_eps)
	print(folder_name,"plot absolute timeseries complete")

	# --> compute timeseries individual parameters
	at.compute_timeseries_individual_parameters(folder_name,include_eps)
	print(folder_name,"compute timeseries parameters complete")

	# --> compare tracked and untracked samples 
	at.compare_tracked_untracked(folder_name,include_eps)
	print(folder_name,"compare tracked/untracked complete")

	# --> perform preliminary spatial/temporal analysis 
	compute_network_distances = True
	at.preliminary_spatial_temporal_correlation_info(folder_name,compute_network_distances,include_eps)
	print(folder_name,"preliminary spatial/temporal analysis complete")

	# --> create and plot F 
	at.compute_F_whole_movie(folder_name,include_eps)
	print(folder_name,"compute F complete")


