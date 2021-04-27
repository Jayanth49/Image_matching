# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 12:38:40 2021

@author: Jayanth
"""


import sys
# !{sys.executable} -m pip uninstall --quiet --yes onnxruntime-gpu
# !{sys.executable} -m pip install --quiet onnxruntime-gpu
# !{sys.executable} -m pip install --quiet --upgrade transformers
# !{sys.executable} -m pip install --quiet --upgrade onnxconverter_common
# !{sys.executable} -m pip install --quiet --upgrade onnxruntime-tools
# !{sys.executable} -m pip install --quiet wget netron pandas

import onnxruntime
import psutil
import numpy

assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
device_name = 'gpu'
print(onnxruntime.get_available_providers())
    
sess_options = onnxruntime.SessionOptions()

sess_options.intra_op_num_threads=psutil.cpu_count(logical=True)

print("session")
session = onnxruntime.InferenceSession(r"D:\GitHub\Image_matching\Untitled Folder\sys_2.onnx")
print("omg")