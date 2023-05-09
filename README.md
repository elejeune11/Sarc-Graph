# THIS REPOSITORY IS FOR LEGACY SARC-GRAPH 

Please use the updated version: https://github.com/Sarc-Graph/sarcgraph


# Sarc-Graph
Segmentation, tracking, and analysis of sarcomeres in hiPSC-CMs

manuscript link: 
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009443

presentation slides:
https://docs.google.com/presentation/d/1GwJxtIdyIpoed4Asnn68aFbQIn92cGqY_fT7-BPT2GI/edit?usp=sharing

**create a virtual environment with Anaconda and install all packages:**

conda create -n sarc_graph_env python=3.6.10

conda activate sarc_graph_env

pip install -r requirements.txt

then, as a demo, you can try:

python -i run_code.py

(note: on Windows, "conda activate sarc_graph_env" may need to be replaced with "CALL conda.bat activate sarc_graph_env")

(note: to create a virtual environment you will first need to install Anaconda https://docs.anaconda.com/anaconda/install/, and, on Windows, it may also be necessary to install visual studio https://visualstudio.microsoft.com/downloads/)

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

Additional real data (real_data_E3, real_data_E4, real_data_E5) can be found here:
https://github.com/elejeune11/Sarc-Graph/tree/main/ALL_MOVIES_RAW/experimental_data

This figure shows example results of analyzing the real data with SarcGraph:
https://github.com/elejeune11/Sarc-Graph/blob/main/explanatory_figures/Expt_res.pdf

The supplementary movies to the paper are also included here:
https://github.com/elejeune11/Sarc-Graph/tree/main/Supplementary_Information


# Code

## Python packages
All code is written in python -- the following packages are required:

The most extensive testing has been done in python 3.6.10

* av:
https://pypi.org/project/av/
* collections:
https://docs.python.org/3/library/collections.html
* csv:
https://docs.python.org/3.6/library/csv.html
* cv2:
https://pypi.org/project/opencv-python/
* glob:
https://docs.python.org/3.6/library/glob.html
* imageio:
https://pypi.org/project/imageio/
* matplotlib:
https://matplotlib.org/3.1.1/users/installing.html
* networkx:
https://networkx.org/documentation/stable/install.html
* numpy:
https://numpy.org/install/
* os:
https://docs.python.org/3.6/library/os.path.html
* pandas:
https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html
* perlin-noise:
https://pypi.org/project/perlin-noise/
* pickle:
https://docs.python.org/3.6/library/pickle.html
* PIL:
https://pypi.org/project/Pillow/
* scipy:
https://www.scipy.org/install.html
* skimage:
https://scikit-image.org/docs/dev/install.html
* sklearn.gaussian_process:
https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
* sys:
https://docs.python.org/3/library/sys.html
* trackpy:
http://soft-matter.github.io/trackpy/v0.4.2/

## Scripts

`run_code.py` - this file calls all other functions. To set up a movie `filename.avi` do the following:
* create a directory named `filename` in the folder `ALL_MOVIES_RAW`
* place `filename.avi`in the `ALL_MOVIES_RAW/filename` folder 
* add `filename` to the list `folder_name_list` in `run_code.py`
Note: for .mov files, .mov must be specified in the call to `file_pre_processing()`, for arbitrary file types, alternative functions can be added to the `file_pre_processing()` script -- an example template is included in the code. 

Core image and time series data processing steps: 

`file_pre_processing()` -  convert the input data (movies) into one 2D numpy array per frame. 

`segmentation_all()` - segment both z-disks and sarcomeres. Key points of the segmentation algorithms are shown here:
https://github.com/elejeune11/Sarc-Graph/blob/main/explanatory_figures/segment_track_figure.png

`run_all_tracking()` - track both z-disks and sarcomeres using the trackpy package:
http://soft-matter.github.io/trackpy/v0.4.2/

`create_spatial_graph()` - create a spatial graph of the sarcomere chains in the field of view. 

`timeseries_all()` - use the results of tracking to make a timeseries plot for each tracked sarcomere. 

`compute_metrics()` - compute functional metrics for each movie, defined in our forthcoming paper.

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
 
 `analyze_J_full_movie()`: 
 https://github.com/elejeune11/Sarc-Graph/blob/main/real_data_E1_results/recovered_F_plot_timeseries.png
 
 `visualize_F_full_movie()`: 
 https://github.com/elejeune11/Sarc-Graph/blob/main/real_data_E1_results/F_anim.gif


## Contact
Don't hesitate to get in touch for additional information 

## Acknowledgements 
This work was supported by the CELL-MET Engineering Research Center NSF ECC-1647837.
