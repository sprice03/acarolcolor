import cv2
from bigcrittercolor import inferMasks
from bigcrittercolor import clusterExtract
from bigcrittercolor import createBCCDataFolder
import torch
import torchvision
import torch.utils
import os
import warnings

warnings.filterwarnings('ignore')

# Make bigcrittercolor folder
createBCCDataFolder("C:/a_carol_data", "county_name")
# Use R to download images from county into "images" subfolder of BCC folder

# Infer the masks of the images
inferMasks(img_ids=None, gpu=True, skip_existing=True,
           text_prompt="animal", erode_kernel_size=0, remove_islands=True,
           show_indv=False, print_steps=True, print_details=True,
           data_folder="C:/path_to_BCC_folder")

# Extract masks from the segments
filterExtractSegments(feature_extractor="inceptionv3", 
                      cluster_params_dict={'algo':"affprop",'preference': -2000},
                      data_folder='C:/path_to_BCC_folder/masks')
