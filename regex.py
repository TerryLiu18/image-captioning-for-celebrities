import os 
import re
import torch

import numpy as np



# read in the name_list data, and use regex to replace

# caps, alphas = get_caps_from(img.unsqueeze(0))
# print(f"this is {name_list}")

def regex(caps, alphas, name_list):
    for name in name_list:
         