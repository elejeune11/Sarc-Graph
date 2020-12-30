# Sarc-Graph
Segmentation, tracking, and analysis of sarcomeres in hiPSC-CMs

# Data

## Synthetic data

The synthetic data (synthetic_data_1, synthetic_data_2, and synthetic_data_3) are generated with the included scripts:

`geom_fcns.py`- Functions to define a baseline geometry of sarcomere chains. 

`render_fcns.py` - Functions to render the baseline geometry as a two dimensional movie. 

`synthetic_data_1.py`:
https://github.com/elejeune11/Sarc-Graph/blob/main/ALL_MOVIES_RAW/synthetic_data_1/frame_000.png

`synthetic_data_2.py`:
https://github.com/elejeune11/Sarc-Graph/blob/main/ALL_MOVIES_RAW/synthetic_data_2/frame_000.png

`synthetic_data_3.py`:
https://github.com/elejeune11/Sarc-Graph/blob/main/ALL_MOVIES_RAW/synthetic_data_3/frame_000.png

This figure explains schematically how the synthetic data is generated:
https://github.com/elejeune11/Sarc-Graph/blob/main/explanatory_figures/render_synthetic_data.png

## Real data

The real data (real_data_Sample_1 and real_data_Sample_2) was originally published with the paper ``An Adaptable Software Tool for Efficient Large-Scale Analysis of Sarcomere Function in hiPSC-Cardiomyocytes'' found at https://www.ahajournals.org/doi/full/10.1161/CIRCRESAHA.118.314505 . 

# Code

## Python packages
All code is written in python -- the following packages are required:
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
* pickle
* PIL 
* scipy
* skimage
* sklearn.gaussian_process
* sys
* trackpy

## Scripts

`run_code.py` - this file calls all other functions. 

`file_pre_processing.py` -  convert the input data (movies) into one 2D numpy array per frame. 

`segmentation.py` - segment both z-disks and sarcomeres. Key points of the segmentation algorithms are shown here:
https://github.com/elejeune11/Sarc-Graph/blob/main/explanatory_figures/segment_track_figure.png

`tracking.py` - track both z-disks and sarcomeres using the trackpy package:
http://soft-matter.github.io/trackpy/v0.4.2/

`spatial_graph.py` - create a spatial graph of the sarcomere chains in the field of view. 

`timeseries.py` - use the results of tracking to make a timeseries plot for each tracked sarcomere. 

`analysis_tools.py` - several functions for visualization and data analysis:

Example output:
https://github.com/elejeune11/Sarc-Graph/blob/main/explanatory_figures/overview_outputs.png
  
 `visualize_segmentation()`:
     https://github.com/elejeune11/Sarc-Graph/blob/main/code_output_synthetic_data_3/analysis/visualize_segmentation.png
    
 `visualize_contract_anim_movie()`:
     https://github.com/elejeune11/Sarc-Graph/blob/main/code_output_synthetic_data_3/analysis/contract_anim/contract_anim.gif
    
 `cluster_timeseries_plot_dendrogram()`:
    https://github.com/elejeune11/Sarc-Graph/blob/main/code_output_synthetic_data_3/analysis/dendrogram_DTW.pdf
    
 `plot_normalized_tracked_timeseries()`:
    https://github.com/elejeune11/Sarc-Graph/blob/main/code_output_synthetic_data_3/analysis/timeseries_tracked_normalized.png
    
 `plot_untracked_absolute_timeseries()`:
    https://github.com/elejeune11/Sarc-Graph/blob/main/code_output_synthetic_data_3/analysis/absolute_sarc_length_untracked.png
    
 `compute_timeseries_individual_parameters()`:
    https://github.com/elejeune11/Sarc-Graph/blob/main/code_output_synthetic_data_3/analysis/timeseries_parameters_info.xlsx
    
 `compare_tracked_untracked()`:
    https://github.com/elejeune11/Sarc-Graph/blob/main/code_output_synthetic_data_3/analysis/length_compare_box_plots.png
    
 `preliminary_spatial_temporal_correlation_info()`:
    https://github.com/elejeune11/Sarc-Graph/blob/main/code_output_synthetic_data_3/analysis/preliminary_spatial_analysis.png
    
 `compute_F_whole_movie()`:
    https://github.com/elejeune11/Sarc-Graph/blob/main/code_output_synthetic_data_3/analysis/recovered_F_plot.png


