# Sarc-Graph
segmentation, tracking, and analysis of sarcomeres in hiPSC-CMs

# Data

## Synthetic data

## Real data

The real data (real_data_Sample_1 and real_data_Sample_2) was originally published with the paper ``An Adaptable Software Tool for Efficient Large-Scale Analysis of Sarcomere Function in hiPSC-Cardiomyocytes'' found at https://www.ahajournals.org/doi/full/10.1161/CIRCRESAHA.118.314505 . 

# Code



## Python packages
All code is written in python -- the following packages are required:
* import numpy as np
import matplotlib.pyplot as plt
import geom_fcns as geo
import render_fcns as ren
import os 
import sys
import av
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as pilimage
from skimage import io
import os 
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  
import trackpy as tp
import os
import trackpy.predict
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.filters import threshold_otsu, threshold_local
from skimage import measure
from scipy import ndimage
from scipy.spatial import distance
from collections import Counter
import os 
import pickle
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os 
import pickle
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os 
import pickle
from matplotlib import cm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, WhiteKernel, ConstantKernel as C, ExpSineSquared
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
import sys
import cv2
import glob
import pickle
import cv2
from skimage.filters import threshold_otsu, threshold_local
from skimage import measure
from scipy import ndimage
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import imageio
from scipy import signal
from skimage.feature import peak_local_max
from scipy.signal import find_peaks
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from scipy.signal import find_peaks
import csv
import pandas as pd
import random 
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, leaves_list
import networkx as nx
import scipy

