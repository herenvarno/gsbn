#!/usr/bin/env python

import os
import sys
import re
import glob
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as misc
#from mnist import MNIST

from google.protobuf import text_format
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../build")
import gsbn_pb2

def crop_center(img, cropx, cropy):
	x = img.shape[0]
	y = img.shape[1]
	startx = x//2-(cropx//2)
	starty = y//2-(cropy//2)
	return img[starty:starty+cropy,startx:startx+cropx]
	
	
if len(sys.argv) < 6:
	print("Arguments wrong! Please retry with command :")
	print("python "+os.path.realpath(__file__)+" <image file directory> <Height> <Width> <Channel> <Output filename>")
	exit(-1)

#mndata = MNIST('./mnist')
#images, labels = mndata.load_training()

#for i,image in enumerate(images):
#	if i>50 :
#		break
#	img = np.asarray(image).reshape((28,28))
#	misc.imsave("./mnist/"+str(i)+".png", img)
#	
#	
#exit()
files = []
for f in glob.glob( os.path.join(sys.argv[1], '*.png') ):
	files.append(f)

files=sorted(files)

patterns = []
color_map = set()

H = int(sys.argv[2])
W = int(sys.argv[3])
C = int(sys.argv[4])
DIM_HCU = H*W*C
DIM_MCU = 256


for f in files:
	img0 = misc.imresize(misc.imread(f), (H,W))
	img0 = crop_center(img0, H, W)
	for h in range(H):
		for w in range(W):
			c = 0
			c = c | img0[h][w][0]
			c = c<<8 | img0[h][w][1]
			c = c<<8 | img0[h][w][2]
			color_map.add(c)

color_list = list(color_map)
color_list.sort()
print(len(color_list))
for c in color_list:
	print(c)
