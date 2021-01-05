# Sarc-Graph
Segmentation, tracking, and analysis of sarcomeres in hiPSC-CMs

pre print link will be posted soon. 

# Data

## Synthetic data

The synthetic data (S1-S5) are generated with the included scripts:

`geom_fcns.py`- Functions to define a baseline geometry of sarcomere chains. 

`render_fcns.py` - Functions to render the baseline geometry as a two dimensional movie. 

This figure explains schematically how the synthetic data is generated:
https://github.com/elejeune11/Sarc-Graph/blob/main/explanatory_figures/render_synthetic_data.png

This figure shows the results of analyzing the synthetic data with SarcGraph:
https://github.com/elejeune11/Sarc-Graph/blob/main/explanatory_figures/validate_synth.pdf

## Real data

The real data (real_data_E1 and real_data_E2) was originally published with the paper ``An Adaptable Software Tool for Efficient Large-Scale Analysis of Sarcomere Function in hiPSC-Cardiomyocytes'' found at https://www.ahajournals.org/doi/full/10.1161/CIRCRESAHA.118.314505 . 

This figure shows the results of analyzing the real data with SarcGraph:
https://github.com/elejeune11/Sarc-Graph/blob/main/explanatory_figures/Expt_res.pdf

# Code

## Python packages
All code is written in python -- the following packages are required:

The most extensive testing has been done in python 3.6.10

* av
* collections
* csv
* cv2
* glob
* imageio
* matplotlib
* networkx
* numpy
* os 
* pandas
* perlin-noise
* pickle
* PIL 
* scipy
* skimage
* sklearn.gaussian_process
* sys
* trackpy

## Scripts

`run_code.py` - this file calls all other functions. 

Core image and time series data processing steps: 

`file_pre_processing()` -  convert the input data (movies) into one 2D numpy array per frame. 

`segmentation_all()` - segment both z-disks and sarcomeres. Key points of the segmentation algorithms are shown here:
https://github.com/elejeune11/Sarc-Graph/blob/main/explanatory_figures/segment_track_figure.png

`run_all_tracking()` - track both z-disks and sarcomeres using the trackpy package:
http://soft-matter.github.io/trackpy/v0.4.2/

`create_spatial_graph()` - create a spatial graph of the sarcomere chains in the field of view. 

`timeseries_all()` - use the results of tracking to make a timeseries plot for each tracked sarcomere. 

Functions for data visualization and analysis:

Schematic of computing the approximate deformation gradient F:
https://github.com/elejeune11/Sarc-Graph/blob/main/explanatory_figures/process_data.png

Example output:
https://github.com/elejeune11/Sarc-Graph/blob/main/explanatory_figures/overview_outputs.png
  
 `visualize_segmentation()`: 
 https://github.com/elejeune11/Sarc-Graph/blob/main/real_data_E1_results/visualize_segmentation_0001.png
    
 `visualize_contract_anim_movie()`:
 https://github.com/elejeune11/Sarc-Graph/blob/main/real_data_E1_results/contract_anim.gif
    
 `cluster_timeseries_plot_dendrogram()`:
 https://github.com/elejeune11/Sarc-Graph/blob/main/real_data_E1_results/dendrogram_DTW.pdf
    
 `plot_normalized_tracked_timeseries()`:
 https://github.com/elejeune11/Sarc-Graph/blob/main/real_data_E1_results/timeseries_tracked_normalized.png
  
 `plot_untracked_absolute_timeseries()`:
 https://github.com/elejeune11/Sarc-Graph/blob/main/real_data_E1_results/absolute_sarc_length_untracked.png
    
 `compute_timeseries_individual_parameters()`:
 https://github.com/elejeune11/Sarc-Graph/blob/main/real_data_E1_results/histogram_time_constants.png
    
 `compare_tracked_untracked()`:
 https://github.com/elejeune11/Sarc-Graph/blob/main/real_data_E1_results/length_compare_box_plots.png
    
 `preliminary_spatial_temporal_correlation_info()`:
 https://github.com/elejeune11/Sarc-Graph/blob/main/real_data_E1_results/preliminary_spatial_analysis.png
    
 `compute_F_whole_movie()`:
 https://github.com/elejeune11/Sarc-Graph/blob/main/real_data_E1_results/recovered_F_plot.png
 
 `analyze_J_full_movie()': 
 https://github.com/elejeune11/Sarc-Graph/blob/main/real_data_E1_results/recovered_F_plot_timeseries.png
 
 `visualize_F_full_movie()': 
 https://github.com/elejeune11/Sarc-Graph/blob/main/real_data_E1_results/F_anim.gif


## Contact
Don't hesitate to get in touch for additional information 

## Acknowledgements 
This work was supported by the CELL-MET Engineering Research Center NSF ECC-1647837.
